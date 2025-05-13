from gymnasium.envs.registration import register

register(
     id='PMSM-v0',
     entry_point='envs.pmsm:EnvPMSM',
     max_episode_steps=200,
)
register(
     id='TC_PMSM-v0',
     entry_point = 'envs.tc_pmsm:EnvPMSMTC',
     max_episode_steps = 200,
)