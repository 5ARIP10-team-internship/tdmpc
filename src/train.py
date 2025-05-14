import warnings

warnings.filterwarnings("ignore")
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
import random
import time
from pathlib import Path

import numpy as np
import torch

import logger
from algorithm.helper import Episode, ReplayBuffer
from algorithm.tdmpc import TDMPC
from cfg import parse_cfg
from env import make_env

torch.backends.cudnn.benchmark = True
__CONFIG__ = "cfgs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step):
    """Evaluate a trained agent."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done, ep_reward, t = False, 0, 0
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t == 0)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            ep_reward += reward
            t += 1
        episode_rewards.append(ep_reward)
    return np.nanmean(episode_rewards)


def train(cfg):
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / cfg.checkpoint_dir
    env, agent, buffer = make_env(cfg, seed=cfg.seed), TDMPC(cfg), ReplayBuffer(cfg)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # Collect trajectory
        obs, _ = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=episode.first)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            episode += (obs, action, reward, done)
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            "episode": episode_idx,
            "step": step,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "episode_reward": episode.cumulative_reward,
        }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category="train")

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            common_metrics["episode_reward"] = evaluate(env, agent, cfg.eval_episodes, step, env_step)
            L.log(common_metrics, category="eval")

    L.finish(agent)
    print("Training completed successfully")


if __name__ == "__main__":
    train(parse_cfg(Path().cwd() / __CONFIG__))
