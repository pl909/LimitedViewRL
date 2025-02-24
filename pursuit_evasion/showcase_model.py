import gym
from envs.pursuit_evasion_env import PursuitEvasionEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = PursuitEvasionEnv(trainingMode=False)

modelFile = "ddpg_single_drone"
episodes = 10


def main():
    model = DDPG.load(modelFile)
    # env = model.get_env()

    for ep in range(episodes):

        print("Episode", ep)
        
        done = False
        obs = env.reset()
        # print("Initial position:", obs)
        score = 0
        solved = 0
        step = 0
        while not done:

            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            # print("Current position: ", obs)
            if step % 100 == 0:
                print("Reward: ", reward)
                step = 0

            if reward >= 100:
                solved += 1

            score += reward
            step += 1

    if done:
        # env.close()
        print("Done!")
        print("Final position: ", obs)
        print("Score: ", score)
        print("Solved", solved, "times")

if __name__ == '__main__':
    main()