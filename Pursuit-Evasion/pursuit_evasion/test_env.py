import gym
import numpy as np
from envs.pursuit_evasion_env import PursuitEvasionEnv

if __name__ == '__main__':
    env = gym.make('PursuitEvasion-v0')
    nSteps = 500000

    for i in range(nSteps):
        done = False
        score = 0

        while not done:

            action = np.array([8, 0, 0.5, 0.5])
            observation_, reward, done, _ = env.step(action)
            score += reward
