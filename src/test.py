from algorithm.tdmpc import TDMPC
from pathlib import Path
from cfg import parse_cfg
from env import make_env
import matplotlib.pyplot as plt
import numpy as np

__CONFIG__, __LOGS__ = 'cfgs', 'logs'

class Test():
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
                action = self.agent.plan(obs, eval_mode=True, step=self.step, t0=t==0)
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                ep_reward += reward
                t += 1
        episode_rewards.append(ep_reward)

        print(f"Average Episode Reward: {np.nanmean(episode_rewards):.2f}")

class TestPMSM(Test):
    def __init__(self, cfg, agent, env):
        super().__init__(cfg, agent, env)

    # Create plots directory if it does not exist
    plots_dir = Path().cwd() / 'plots'
    plots_dir.mkdir(exist_ok=True)

    def run(self):
        episode_rewards = []
        ss_errors = np.zeros((self.num_episodes, 2))
        for episode in range(self.num_episodes):
            obs, _ = self.env.reset() 
            action_list = []
            reward_list = []
            state_list  = [obs[0:4]]

            done, ep_reward, t = False, 0, 0
            while not done:
                action = self.agent.plan(obs, eval_mode=True, step=self.step, t0=t==0)
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated

                if not done:
                    action_list.append(action.cpu().numpy())
                    state_list.append(obs[0:4])
                    reward_list.append(reward)
                ep_reward += reward
                t += 1

            episode_rewards.append(ep_reward)
            Inorm = [state[0:2] for state in state_list[-10:]]
            Iref = obs[2:4]
            ss_errors[episode,:] = np.abs(np.mean(Inorm, axis=0) - Iref)

            if episode < 10:
                self.plot_three_phase(episode, state_list, action_list, reward_list,
                                        "TDMPC", "absolute", 200*2*np.pi*obs[4])

        self.plot_error(ss_errors)

        print(f"Average Episode Reward: {np.nanmean(episode_rewards):.2f}")

    def plot_three_phase(self, idx, observations, actions, reward, env_name, reward_type, speed=None):
        plt.figure(idx, figsize=(10, 6))
        if speed is not None:
            plt.suptitle(f"Reward: {reward_type}\nSpeed = {speed} [rad/s]")
        # Plot State
        ax = plt.subplot(131)
        ax.set_title("State vs step")
        ax.plot(observations, label=['Id', 'Iq', 'Idref', 'Iqref'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot action
        ax = plt.subplot(132)
        ax.set_title("Action vs step")
        ax.plot(actions, label=['Vd', 'Vq'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        plt.savefig(f"plots/{env_name}_{idx}.png", bbox_inches='tight')
        plt.close()

    def plot_error(self, errors):
        # Remove outliers from the error list
        q1 = np.percentile(errors, 25, axis=0)
        q3 = np.percentile(errors, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        error_list = errors[
            np.all((errors >= lower_bound) & (errors <= upper_bound), axis=1)
        ]

        # Create a box plot of the error
        plt.figure(figsize=(8, 6))
        plt.boxplot(error_list, labels=['Id error', 'Iq error'])
        plt.ylabel("Absolute error")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("plots/TDMPC_error.png", bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    cfg = parse_cfg(Path().cwd() / __CONFIG__)
    env = make_env(cfg, seed=42, render_mode=None)
    agent = TDMPC(cfg)

    # Load the model
    model_dir = Path().cwd() / __LOGS__ / cfg.task / 'models'
    agent.load(model_dir / 'model.pth')

    if cfg.task == 'PMSM-v0':
        test = TestPMSM(cfg, agent, env)
    else:
        test = Test(cfg, agent, env)
    test.run()