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
import datetime
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


def _server_action_to_env_actions(
    raw_action: np.ndarray,
    replan_steps: int,
    *,
    skip_first_frame: bool = False,
):
    """
    Convert server action output to a list of 7-dim LIBERO env actions.

    Server's postprocess_action already denormalizes the action and selects
    used_action_channel_ids, so raw_action is already in original scale.
    Shape: (7, frame_chunk, action_per_frame).
    Channel layout: [dx, dy, dz, daa1, daa2, daa3, gripper].

    LIBERO env expects [dx, dy, dz, daa1, daa2, daa3, gripper].

    Returns:
        env_actions: list[np.ndarray] at control frequency.
        steps_per_latent_frame: action_per_frame from server output.
    """
    c, f, h = raw_action.shape                  # (7, frame_chunk, action_per_frame)
    steps = raw_action.reshape(c, -1).T         # (total_steps, 7)

    if skip_first_frame:
        # Match server-side conditioning at frame_st_id == 0:
        # first latent-frame actions are fixed to action_cond (zeros) and should not be executed.
        env_actions = list(steps[h:])
    else:
        env_actions = list(steps)

    if replan_steps is not None:
        env_actions = env_actions[:replan_steps]

    # KV-cache updates consume observations at latent-video rate (1 frame per h actions).
    # Ensure we only execute full latent-frame windows to avoid obs/state length mismatch.
    full_windows = len(env_actions) // h
    env_actions = env_actions[: full_windows * h]
    return env_actions, h

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
    
    def _obs_to_frame(ob):
        # Only flip to correct OpenGL upside-down rendering.
        # Resize (256x256 -> 256x320) is done server-side via F.interpolate,
        # matching preprocessing (extract_latents_wan22_robotwin.py) exactly.
        img = np.ascontiguousarray(ob["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(ob["robot0_eye_in_hand_image"][::-1, ::-1])
        return {
            OBS_IMAGE_KEY: img,
            OBS_WRIST_KEY: wrist_img,
        }
    
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
            # Reset server for new episode (set prompt)
            client.infer({"reset": True, "prompt": task_description})

            first = True
            full_obs_list = []
            full_action_history = []
            key_frame_list = []
            obs = env.set_init_state(initial_states[episode_idx])
            full_obs_list.append(_obs_to_frame(obs))
            t = 0
            replay_images = []
            done = False
            first_obs = None
            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    if first:
                        first_obs = _obs_to_frame(obs)
                    ret = client.infer(dict(obs=first_obs, prompt=task_description)) #(TASK_ENV, model, observation)
                    action = ret['action']
                    key_frame_list = []

                    assert action.shape[2] % 4 == 0
                    action_per_frame = action.shape[2] // 4

                    start_idx = 1 if first else 0
                    # print(action.shape)
                    for i in range(start_idx, action.shape[1]):
                        for j in range(action.shape[2]):
                            raw_action_step = action[:, i, j].flatten() 
                            full_action_history.append(raw_action_step)
                            ee_action = action[:, i, j]
                            ee_action = ee_action[:8]
                            obs, _, done, _ = env.step(ee_action)
                            if (j+1) % action_per_frame == 0:
                                key_frame_list.append(_obs_to_frame(obs))
                                full_obs_list.append(_obs_to_frame(obs))
                                replay_images.append(_obs_to_frame(obs)[OBS_IMAGE_KEY])

                    first = False
                    client.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += action.shape[1] * action.shape[2]

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
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