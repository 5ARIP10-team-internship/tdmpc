import gymnasium as gym

class ActionRepeatWrapper(gym.Wrapper):
	def __init__(self, env, num_repeats):
		super().__init__(env)
		self._num_repeats = num_repeats

	def step(self, action):
		total_reward = 0.0
		done = False

		for _ in range(self._num_repeats):
			obs, reward, terminated, truncated, info = self.env.step(action)
			done = terminated or truncated
			total_reward += reward
			if done:
				break

		return obs, total_reward, terminated, truncated, info

def make_env(cfg, render_mode=None):
    domain, task = cfg.task.replace('-', '_').split('_', 1)

    if render_mode:
        env = gym.make('Pendulum-v1', render_mode=render_mode, g=9.81)
    else:
        env = gym.make('Pendulum-v1', render_mode="rgb_array", g=9.81)

    env = ActionRepeatWrapper(env, cfg.action_repeat)
    env = gym.wrappers.TimeLimit(env, cfg.episode_length)

    # Convenience
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env