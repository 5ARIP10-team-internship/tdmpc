import inspect

import gymnasium as gym

import envs  # noqa: F401


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


def make_env(cfg, seed=None, render_mode=None):
    domain = cfg.task

    # Get the environment class from gym
    env_spec = gym.envs.registry.get(domain)
    if env_spec is None:
        raise ValueError(f"Environment '{domain}' is not registered in gym.")

    env_class = env_spec.entry_point
    module_name, class_name = env_class.split(":")

    env_module = __import__(module_name, fromlist=[class_name])
    env_cls = getattr(env_module, class_name)

    # Get valid arguments from the environment's __init__ method
    valid_params = set(inspect.signature(env_cls.__init__).parameters.keys())

    # Keep only valid parameters
    filtered_kwargs = {k: v for k, v in cfg.items() if k in valid_params}

    env = gym.make(domain, render_mode=render_mode, **filtered_kwargs)
    env = ActionRepeatWrapper(env, cfg.action_repeat)
    env = gym.wrappers.TimeLimit(env, cfg.episode_length)
    env.reset(seed=seed)

    # Convenience
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]

    return env
