# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if len(repo_list) <= 1:
        datasets_out_lst = [construct_lerobot(repo_id, config) for repo_id in repo_list]
    else:
        construct_func = partial(
            construct_lerobot,
            config=config,
        )
        with Pool(num_init_worker) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)
    return datasets_out_lst

def _safe_quat(quat):
    """将零范数四元数替换为单位四元数 [0,0,0,1]（xyzw），避免 scipy R.from_quat 报错。"""
    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim == 1:
        quat = quat.reshape(1, -1)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    # identity (1,4) 才能与 (N,4) 正确广播
    identity = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=quat.dtype)
    quat = np.where(norm < 1e-8, identity, quat / np.maximum(norm, 1e-8))
    return quat.squeeze(0) if quat.shape[0] == 1 else quat

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    quat = _safe_quat(pose[:, 3:7])
    first_quat = _safe_quat(pose[:1, 3:7])
    rot = R.from_quat(quat)
    first_rot = R.from_quat(np.tile(first_quat.reshape(1, -1), (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

import numpy as np

def pad_to_dim(arr, target_dim, pad_value, add_batch_dim=True):
    """
    Pad (or truncate) a 1D array to target_dim.

    Args:
        arr: list or np.ndarray
        target_dim: int, target dimension
        pad_value: value used for padding
        add_batch_dim: whether to add leading batch dim (shape -> [1, D])

    Returns:
        np.ndarray
    """
    arr = np.asarray(arr, dtype=float)

    # truncate if longer
    if arr.shape[0] > target_dim:
        arr = arr[:target_dim]

    # pad if shorter
    elif arr.shape[0] < target_dim:
        arr = np.pad(
            arr,
            (0, target_dim - arr.shape[0]),
            constant_values=pad_value
        )

    if add_batch_dim:
        arr = arr[None]

    return arr

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=8,
    ):
        self._datasets = construct_lerobot_multi_processor(config, num_init_worker)
        self.item_id_to_dataset_id, self.acc_dset_num = self._get_item_id_to_dataset_id()

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = pad_to_dim(config.norm_stat['q01'], 30, pad_value=0.0)
        self.q99 = pad_to_dim(config.norm_stat['q99'], 30, pad_value=1.0)
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        expected_tokens = None
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
            # 校验各摄像头的 latent 实际 token 数与元数据一致，且互相匹配
            try:
                data = torch.load(latent_file, weights_only=False, map_location='cpu')
                actual_tokens = data["latent"].shape[0]
                claimed_tokens = (
                    data["latent_num_frames"]
                    * data["latent_height"]
                    * data["latent_width"]
                )
                if actual_tokens != claimed_tokens:
                    print(
                        f"[_check_meta] skip ep={episode_index} key={key}: "
                        f"latent shape {actual_tokens} != claimed {claimed_tokens}",
                        flush=True,
                    )
                    return False
                if expected_tokens is None:
                    expected_tokens = actual_tokens
                elif actual_tokens != expected_tokens:
                    print(
                        f"[_check_meta] skip ep={episode_index} key={key}: "
                        f"latent token mismatch across cameras "
                        f"({actual_tokens} vs {expected_tokens})",
                        flush=True,
                    )
                    return False
            except Exception as e:
                print(
                    f"[_check_meta] skip ep={episode_index} key={key}: "
                    f"failed to load latent file: {e}",
                    flush=True,
                )
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        wrist_latent = torch.cat(latent_lst[1:], dim=2)
        cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict

    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        # n = action.shape[0]
        # if action.shape[1] == 7:
        #     #  单臂轴角格式
        #     from scipy.spatial.transform import Rotation as R
        #     xyz = action[:, :3]
        #     aa = action[:, 3:6]   # axis-angle
        #     gripper = action[:, 6:7]
        #     # 轴角 -> 四元数，传给 get_relative_pose
        #     quat = R.from_rotvec(aa).as_quat()  # (n, 4) xyzw
        #     pose7 = np.concatenate([xyz, quat], axis=1)  # (n, 7): xyz + quat
        #     left_action = get_relative_pose(pose7)
        #     left_gripper = gripper  # (n, 1)
        # else:
        #     left_action = get_relative_pose(action[:, :7])
        #     left_gripper = action[:, 7:8]
        # left_action = np.asarray(left_action, dtype=np.float32)
        # if action.shape[1] >= 15:
        #     right_action = get_relative_pose(action[:, 8:15])
        #     right_action = np.asarray(right_action, dtype=np.float32)
        #     right_gripper = action[:, 15:16]
        # else:
        #     # 单臂：右臂用零填充
        #     right_action = np.zeros((n, 7), dtype=np.float32)
        #     right_gripper = np.zeros((n, 1), dtype=action.dtype)
        # action = np.concatenate([left_action, left_gripper, right_action, right_gripper], axis=1)
        # action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)
        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4
        # action = action[:required_action_num]
        # action_mask = np.ones_like(action, dtype='bool')
        current_len = action.shape[0]
        if current_len >= required_action_num:
            action = action[:required_action_num]
            action_mask = np.ones((required_action_num, action.shape[1]), dtype=bool)
        else:
            pad_len = required_action_num - current_len
            action_mask = np.ones((current_len, action.shape[1]), dtype=bool)
            action = np.pad(
                action,
                ((0, pad_len), (0, 0)),
                mode='constant',
                constant_values=0
            )
            action_mask = np.pad(
                action_mask,
                ((0, pad_len), (0, 0)),
                mode='constant',
                constant_values=False
            )
        assert action.shape[0] == required_action_num

        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        # action_aligned = (action_aligned - self.q01) / (
        #         self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)
        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])
        
        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        out_dict['episode_index'] = episode_index
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['robotwin_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
