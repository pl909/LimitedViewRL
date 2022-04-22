import gym
import numpy as np
import pybullet as pb
import time

from pursuit_evasion.resources import utils as r
from pursuit_evasion.resources import Quadrotor

# PursuitEvasion-v0

class PursuitEvasionEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):

        # Action Space of 4 actions
        # (force z, torque x, torque y, torque z)
        self.action_space = gym.spaces.box.Box(
            low = np.array([-10, -1, -1, -1]),
            high = np.array([10, 1, 1, 1]))

        # Observation Space of 6 elements: 2 arrays of 3 elements
        # (Robot Position(x,y,z), Robot Orientation(x,y,z))
        self.observation_space = gym.spaces.box.Box(
            low = np.array([-20, -20, -20, -10, -10, -10]),
            high = np.array([20, 20, 20, 10, 10, 10]))

        # Random seed generator
        self.np_random, _ = gym.utils.seeding.np_random()

        # PyBullet server connection (change to DIRECT later)
        self.pbClient = r.initializeGUI(enable_gui=True)

        # Add plane and robot models
        self.planeId = pb.loadURDF("plane.urdf")
        # Drone 1:
        self.drone1 = Quadrotor(urdf='C:/DEV/Pursuit-Evasion/Pursuit-Evasion-Quadcopter/Pursuit-Evasion/pursuit_evasion/resources/robot_models/quadrotor.urdf',
                    startPosition=[0, 1, 1], client=self.pbClient)


    def step(self, action):
        # Action is an array with 4 elements
        # Returns observation, reward, done, info
        # observation is an array with 6 elements
        # done returns True if goal is reached or time expired
        # info is null, assign to _
        
        reward = 0
        goal = np.array([1, 3, 1])

        # Apply action:
        self.drone1.apply_action(action)

        pb.stepSimulation()
        time.sleep(1 / 240)

        # Get observation and position
        obs, pos = self.drone1.get_observation()

        # Detected done:
        # Goal: (1, 3, 1)
        if pos.any() == goal.any():
            done = True
            reward = 10
        else:
            done == False
        
        info = {}

        return obs, reward, done, info

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pb.disconnect(self.client)
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]