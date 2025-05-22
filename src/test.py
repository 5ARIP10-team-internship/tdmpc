import os
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm.tdmpc import TDMPC
from cfg import parse_cfg
from env import make_env

__CONFIG__ = "cfgs"


def get_csv_filename(model_path):
    path = Path(model_path)

    if "train_" in path.parent.name:
        identifier = path.parent.name
        filename = f"{identifier}"
    elif "array_" in path.parent.name:
        identifier = path.parent.name
        experiment = path.name
        filename = f"{identifier}_{experiment}"
    else:
        raise ValueError("Unexpected model path format.")

    return filename


class Test:
    def __init__(self, cfg, agent, env):
        self.env = env
        self.agent = agent
        self.num_episodes = 100
        self.step = cfg.train_steps

    def run(self):
        episode_rewards = []
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            done, ep_reward, t = False, 0, 0
            while not done:
                action = self.agent.plan(obs, eval_mode=True, step=self.step, t0=t == 0)
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                ep_reward += reward
                t += 1
        episode_rewards.append(ep_reward)

        print(f"Average Episode Reward: {np.nanmean(episode_rewards):.2f}")


class TestPMSM(Test):
    def __init__(self, cfg, agent, env):
        super().__init__(cfg, agent, env)

        # Create output directory if it does not exist
        self.out_dir = Path().cwd() / "results"
        self.out_dir.mkdir(exist_ok=True)

    def run(self):
        data_records = []
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()

            done, t = False, 0
            while not done:
                action = self.agent.plan(obs, eval_mode=True, step=self.step, t0=t == 0)
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated

                action = action.cpu().numpy()
                record = {
                    "episode": episode,
                    "step": t,
                    "Id": obs[0],
                    "Iq": obs[1],
                    "Id_ref": obs[2],
                    "Iq_ref": obs[3],
                    "action_d": action[0],
                    "action_q": action[1],
                    "reward": reward,
                    "speed": 200 * 2 * np.pi * obs[4],
                }
                data_records.append(record)
                t += 1

        df = pd.DataFrame(data_records)
        model_name = get_csv_filename(cfg.checkpoint_dir)
        out_csv = os.path.join(self.out_dir, f"{model_name}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved test data to {out_csv}")


class TestTCPMSM(Test):
    def __init__(self, cfg, agent, env):
        super().__init__(cfg, agent, env)

        # Create output directory if it does not exist
        self.out_dir = Path().cwd() / "results"
        self.out_dir.mkdir(exist_ok=True)

    def run(self):
        data_records = []
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()

            done, t = False, 0
            while not done:
                action = self.agent.plan(obs, eval_mode=True, step=self.step, t0=t == 0)
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated

                action = action.cpu().numpy()
                record = {
                    "episode": episode,
                    "step": t,
                    "Te": obs[0],
                    "Te_ref": obs[1],
                    "Id_ref": obs[2],
                    "Iq_ref": obs[3],
                    "action_d": action[0],
                    "action_q": action[1],
                    "reward": reward,
                    "speed": 200 * 2 * np.pi * obs[4],
                }
                data_records.append(record)
                t += 1

        df = pd.DataFrame(data_records)
        model_name = get_csv_filename(cfg.checkpoint_dir)
        out_csv = os.path.join(self.out_dir, f"{model_name}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved test data to {out_csv}")


if __name__ == "__main__":
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    env = make_env(cfg, seed=42, render_mode=None)
    agent = TDMPC(cfg)

    # Load the model
    model_dir = Path().cwd() / cfg.checkpoint_dir / "models"
    agent.load(model_dir / "model.pth")

    if cfg.task == "PMSM-v0":
        test = TestPMSM(cfg, agent, env)
    elif cfg.task == "TC_PMSM-v0":
        test = TestTCPMSM(cfg, agent, env)
    else:
        test = Test(cfg, agent, env)
    test.run()
