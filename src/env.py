import gymnasium as gym

def make_env(cfg, render_mode=None):
    if render_mode:
        env = gym.make('Pendulum-v1', render_mode=render_mode, g=9.81)
    else:
        env = gym.make('Pendulum-v1', render_mode="rgb_array", g=9.81)
    env._max_episode_steps = 200//cfg.action_repeat

    # Convenience
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env