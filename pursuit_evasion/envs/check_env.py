from stable_baselines3.common.env_checker import check_env
from pursuit_evasion_env import PursuitEvasionEnv

env = PursuitEvasionEnv()

check_env(env)