import gym
import numpy as np

if __name__ == '__main__':
    # env = gym.make('PursuitEvasion-v0')
    # nSteps = 500

    # for i in range(nSteps):
    #     done = False
    #     score = 0

    #     while not done:

    #         action = np.array([8, 0, 0.5, 0.5])
    #         observation_, reward, done, _ = env.step(action)
    #         score += reward

    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    c_array = np.concatenate((array1, array2), axis=None)
    print(array1)
    print(array1)
    print(c_array)