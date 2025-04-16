from algorithm.tdmpc import TDMPC
from pathlib import Path
from cfg import parse_cfg
from env import make_env
import matplotlib.pyplot as plt
import numpy as np

__CONFIG__, __LOGS__ = 'cfgs', 'logs'

class PlotTest():
    def __init__(self):
        return

    def plot_three_phase(self, idx, observations, actions, reward, env_name, reward_type, speed=None):
        plt.clf()
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
        plt.pause(0.001)  # pause a bit so that plots are updated

        # plt.show()

def test(cfg):
    env, agent = make_env(cfg, render_mode=None), TDMPC(cfg)
    model_dir = Path().cwd() / __LOGS__ / cfg.task / 'models'
    agent.load(model_dir / 'model.pt')

    # Create plots directory if it does not exist
    plots_dir = Path().cwd() / 'plots'
    plots_dir.mkdir(exist_ok=True)
    plot = PlotTest()

    num_episodes = 10
    episode_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset(options={"Idref":0, "Iqref":100}) 
        action_list = []
        reward_list = []
        state_list  = [obs[0:4]]

        plt.figure(episode, figsize=(10, 6))
        done, ep_reward, t = False, 0, 0
        while not done:
            action = agent.plan(obs, eval_mode=True, step=2501, t0=t==0)
            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            if not done:
                action_list.append(action.cpu().numpy())
                state_list.append(obs[0:4])
                reward_list.append(reward)
            ep_reward += reward
            t += 1

        episode_rewards.append(ep_reward)
        plot.plot_three_phase(episode, state_list, action_list, reward_list,
                                "TDMPC", "absolute", 200*2*np.pi*obs[4])
        plt.close()
    print(f"Average Episode Reward: {np.nanmean(episode_rewards):.2f}")

if __name__ == '__main__':
    test(parse_cfg(Path().cwd() / __CONFIG__))