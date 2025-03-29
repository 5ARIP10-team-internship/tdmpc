from gymnasium.envs.registration import register

register(
     id='PMSM-v0',
     entry_point='envs.pmsm:EnvPMSM',
     max_episode_steps=200,
)