from pursuit_evasion_env import PursuitEvasionEnv

env = PursuitEvasionEnv()
episodes = 50

# Simple gym loop to make a deeper checkenv
for episode in range(episodes):
    done = False
    obs = env.reset()
    print("Inicial obs:", obs)
    
    while not done:
        random_action = env.action_space.sample()
        print("action:", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward:", reward)