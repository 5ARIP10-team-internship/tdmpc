from algorithm.tdmpc import TDMPC
from pathlib import Path
from cfg import parse_cfg
from env import make_env
__CONFIG__, __LOGS__ = 'cfgs', 'logs'

def test(cfg):
    env, agent = make_env(cfg, render_mode="human"), TDMPC(cfg)
    model_dir = Path().cwd() / __LOGS__ / cfg.task / 'models'
    agent.load(model_dir / 'model.pt')
    obs, _ = env.reset()
    done, ep_reward, t = False, 0, 0
    while not done:
        action = agent.plan(obs, eval_mode=True, step=50000, t0=t==0)
        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = terminated or truncated
        ep_reward += reward
        t += 1
    print(ep_reward)

if __name__ == '__main__':
    test(parse_cfg(Path().cwd() / __CONFIG__))