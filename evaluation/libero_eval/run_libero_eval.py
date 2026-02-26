"""
LIBERO evaluation client for lingbot-va.

Evaluation logic follows reasoningvla/libero_eval: one server (wan_va_server with
libero_spatial config), one client that runs LIBERO benchmark and queries the server.

Usage:
  Terminal 1: bash evaluation/libero_eval/launch_server.sh
  Terminal 2: bash evaluation/libero_eval/launch_client.sh [task_suite_name] [port]

  Or:
  python -m evaluation.libero_eval.run_libero_eval --host 127.0.0.1 --port 29536 \\
      --task_suite_name libero_spatial --replan_steps 10
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import pathlib

import numpy as np
import tqdm
import tyro

# Set before importing libero
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from . import image_tools
from .websocket_client_policy import WebsocketClientPolicy

# Server obs keys (must match wan_va config obs_cam_keys for libero_spatial)
OBS_IMAGE_KEY = "observation.images.image"
OBS_WRIST_KEY = "observation.images.wrist_image"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

def _get_libero_env(task, resolution: int, seed: int):
    """Create LIBERO env and task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _server_action_to_env_actions(raw_action: np.ndarray, replan_steps: int):
    """
    Convert server action output to a list of 7-dim LIBERO env actions.

    Server's postprocess_action already denormalizes the action and selects
    used_action_channel_ids, so raw_action is already in original scale.
    Shape: (7, frame_chunk, action_per_frame).
    Channel layout: [dx, dy, dz, daa1, daa2, daa3, gripper].

    LIBERO env expects [dx, dy, dz, daa1, daa2, daa3, gripper].
    """
    c, f, h = raw_action.shape                  # (7, frame_chunk, action_per_frame)
    steps = raw_action.reshape(c, -1).T         # (total_steps, 7)

    env_actions = list(steps)                   # each element: (7,) already in original scale

    if replan_steps is not None:
        env_actions = env_actions[:replan_steps]
    return env_actions

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 29536
    resize_size: int = 256
    replan_steps: int | None = None

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    video_out_path: str = "data/libero/videos"
    seed: int = 7

    frame_chunk_size: int = 2
    action_per_frame: int = 4


def eval_libero(cfg: Args) -> None:
    np.random.seed(cfg.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", cfg.task_suite_name)

    pathlib.Path(cfg.video_out_path).mkdir(parents=True, exist_ok=True)

    if cfg.task_suite_name == "libero_spatial":
        max_steps = 280
    elif cfg.task_suite_name == "libero_object":
        max_steps = 280
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 300
    elif cfg.task_suite_name == "libero_10":
        max_steps = 520
    elif cfg.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError("Unknown task suite: %s" % cfg.task_suite_name)

    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="task"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, cfg.seed
        )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_task), desc="episode", leave=False
        ):
            env.reset()
            action_plan = collections.deque()
            last_raw_action = None
            key_frame_list = []        # frames at VIDEO rate (one per action_per_frame steps)
            action_step_in_chunk = 0  # counts action steps within current chunk

            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            done = False

            # Reset server for new episode (set prompt)
            client.infer({"reset": True, "prompt": task_description})

            def _obs_to_frame(ob):
                img = np.ascontiguousarray(
                    ob["agentview_image"][::-1, ::-1]
                )
                wrist_img = np.ascontiguousarray(
                    ob["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        img, cfg.resize_size, cfg.resize_size
                    )
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        wrist_img, cfg.resize_size, cfg.resize_size
                    )
                )
                return img, wrist_img

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img, wrist_img = _obs_to_frame(obs)
                    replay_images.append(img)

                    if not action_plan:
                        if (
                            last_raw_action is not None
                            and len(key_frame_list) > 0
                        ):
                            client.infer(
                                {
                                    "obs": key_frame_list,
                                    "compute_kv_cache": True,
                                    "state": last_raw_action,
                                    "prompt": task_description,
                                }
                            )
                        obs_dict = {
                            "obs": [
                                {
                                    OBS_IMAGE_KEY: img,
                                    OBS_WRIST_KEY: wrist_img,
                                }
                            ],
                            "prompt": task_description,
                        }
                        resp = client.infer(obs_dict)
                        raw_action = resp["action"]
                        last_raw_action = raw_action
                        # reset: collect video-rate frames for the next kv_cache call
                        key_frame_list = []
                        action_step_in_chunk = 0
                        chunk = _server_action_to_env_actions(
                            raw_action, cfg.replan_steps
                        )
                        action_plan.extend(chunk)

                    action = list(action_plan.popleft())
                    obs, reward, done, info = env.step(action)
                    action_step_in_chunk += 1
                    if not done:
                        new_img, new_wrist = _obs_to_frame(obs)
                        # only keep one frame per action_per_frame steps (video rate)
                        # sample the LAST frame of each latent-frame window
                        if action_step_in_chunk % cfg.action_per_frame == 0:
                            key_frame_list.append(
                                {
                                    OBS_IMAGE_KEY: new_img,
                                    OBS_WRIST_KEY: new_wrist,
                                }
                            )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = (
                pathlib.Path(cfg.video_out_path)
                / f"rollout_{task_segment}_{suffix}_{timestamp}.mp4"
            )
            try:
                imageio.mimwrite(
                    out_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            except Exception as e:
                logging.warning("Video save failed: %s", e)

            logging.info(
                "Success: %s | Episodes: %d | Successes: %d (%.1f%%)",
                done,
                total_episodes,
                total_successes,
                100.0 * total_successes / total_episodes,
            )

        logging.info(
            "Task success rate: %.3f | Total success rate: %.3f",
            float(task_successes) / max(1, task_episodes),
            float(total_successes) / float(total_episodes),
        )

    logging.info(
        "Total success rate: %.3f (%d / %d)",
        float(total_successes) / float(total_episodes),
        total_successes,
        total_episodes,
    )


def main():
    logging.basicConfig(level=logging.INFO)
    # 用 tyro.cli(Args) 直接解析为 Args，得到 --host / --port 等扁平参数；
    # 若用 tyro.cli(eval_libero) 会因参数名为 args 而得到 --cfg.host
    cfg = tyro.cli(Args)
    eval_libero(cfg)


if __name__ == "__main__":
    main()
