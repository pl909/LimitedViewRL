from tabnanny import verbose
import gym
import numpy as np
from pursuit_evasion.envs.pursuit_evasion_env import PursuitEvasionEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

if __name__ == '__main__':

    env = gym.make('PursuitEvasion-v0')
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG('MlpPolicy', env, action_noise=action_noise , verbose=1)
    model.learn(total_timesteps=1000)

    episodes = 10

    for ep in range(episodes):

        done = False
        obs = env.reset()
        score = 0

        # Temporary
        # action = np.array([8, 0, 0.5, 0.5])
        # obs, rewards, done, _ = env.step(action)

        while not done:

            action, _steps = model.predict(obs)
            obs, rewards, done, _ = env.step(action)
            # score += rewards
            print(obs)

    print("done!")

