import gym
import numpy as np
from envs.pursuit_evasion_env import PursuitEvasionEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import os

def train_agent(env_type, total_timesteps=10000):
    try:
        env = PursuitEvasionEnv(drone_type=env_type, trainingMode=True)
        
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), 
                                       sigma=0.1 * np.ones(n_actions))

        model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
        model.learn(total_timesteps=total_timesteps, log_interval=10)
        
        # Save model in models directory
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', f"ddpg_{env_type}")
        model.save(model_path)
        
        return model
    
    except Exception as e:
        print(f"Error training {env_type}: {str(e)}")
        return None

def evaluate_agents(pursuer_model, evader_model, episodes=10):
    if pursuer_model is None or evader_model is None:
        print("Cannot evaluate - one or both models failed to train")
        return
        
    try:
        env = PursuitEvasionEnv(trainingMode=False)
        
        pursuer_wins = 0
        evader_wins = 0
        
        for ep in range(episodes):
            done = False
            obs = env.reset()
            
            while not done:
                # Get actions from both models
                pursuer_action, _ = pursuer_model.predict(obs[:12])
                evader_action, _ = evader_model.predict(obs[12:])
                
                # Apply actions and get new state
                obs, reward, done, info = env.step((pursuer_action, evader_action))
                
                if done:
                    if info['distance'] < 0.5:
                        pursuer_wins += 1
                    else:
                        evader_wins += 1
        
        print(f"\nResults after {episodes} episodes:")
        print(f"Pursuer wins: {pursuer_wins}")
        print(f"Evader wins: {evader_wins}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    
    finally:
        env.close()

def main():
    # Train pursuer
    print("Training pursuer...")
    pursuer_model = train_agent('pursuer')
    
    # Train evader
    print("\nTraining evader...")
    evader_model = train_agent('evader')
    
    # Evaluate both agents
    print("\nEvaluating agents...")
    evaluate_agents(pursuer_model, evader_model)

if __name__ == '__main__':
    main() 