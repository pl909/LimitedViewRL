from .envs.pursuit_evasion_env import PursuitEvasionEnv
# from gym.envs.registration import register
import gym
gym.envs.register(
    id='PursuitEvasion-v0',
    entry_point='pursuit_evasion.envs:PursuitEvasionEnv'
)