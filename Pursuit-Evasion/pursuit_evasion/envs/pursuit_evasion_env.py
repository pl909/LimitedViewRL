from turtle import position
import gym
import numpy as np
import pybullet as pb
import time

from .resources import utils as r
from .resources import Quadrotor

# PursuitEvasion-v0

class PursuitEvasionEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        
        self.done = False
        self.timer = 0

        # Action Space of 4 actions
        # (force z, torque x, torque y, torque z)
        self.action_space = gym.spaces.box.Box(
            low=np.array([-10, -1, -1, -1]),
            high=np.array([10, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32)

        # Observation Space of 6 elements: 2 arrays of 3 elements
        # (Robot Position(x,y,z), Robot Orientation(x,y,z))
        self.observation_space = gym.spaces.box.Box( # Change to np.inf and 2*np.pi
            low=np.array([-200, -200, -200, -100, -100, -100, -100, -100, -100, -100, -100, -100]),
            high=np.array([200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
            shape=(12,),
            dtype=np.float32)

        # Random seed generator
        self.np_random, _ = gym.utils.seeding.np_random()

        # PyBullet server connection (change to DIRECT later)
        self.pbClient = r.initializeGUI(enable_gui=True, connection='GUI')


        # Add plane and robot models
        self.planeId = pb.loadURDF("plane.urdf")
        # Drone 1:
        self.startPosition1 = np.array([0, 1, 1])
        self.drone1 = Quadrotor(urdf='C:/DEV/Pursuit-Evasion/Pursuit-Evasion-Quadcopter/Pursuit-Evasion/pursuit_evasion/resources/robot_models/quadrotor.urdf',
                    startPosition=self.startPosition1, client=self.pbClient)
        # Inicial distance
        posDiff = np.array([1, 3, 1]) - self.startPosition1
        sumSq = np.dot(posDiff.T, posDiff)
        self.distance1 = np.sqrt(sumSq)



    def step(self, action):
        # Action is an array with 4 elements
        # Returns observation, reward, done, info
        # observation is an array with 6 elements
        # done returns True if goal is reached or time expired
        # info is null, assign to _
        
        reward = 0
        self.done = False
        goal = np.array([1, 3, 1])
        # Apply action:
        self.drone1.apply_action(action)

        pb.stepSimulation()
        time.sleep(1 / 240)
        self.timer += 1

        # Get observation and position
        position, orientation, linearVel, angularVel = self.drone1.get_observation()
        observation = np.concatenate((position, orientation, linearVel, angularVel))
        # observation = position
        # Calculate new distance
        posDiff = goal - position
        sumSq = np.dot(posDiff.T, posDiff)
        newDistance1 = np.sqrt(sumSq)

        # Reward rules
        # Goal: (1, 3, 1)
        if (position == goal).all():
            self.done = True
            reward = 100
        elif self.timer >= 500:
            self.done = True
            reward = 0
            self.timer = 0
        elif newDistance1 < self.distance1:
            reward = 10
        elif newDistance1 > self.distance1:
            reward = -10
        # elif position.any == goal.any():
        #     reward = 1
        
        self.distance1 = newDistance1
        info = {}

        return observation, reward, self.done, info

    def reset(self):
        # pass
        # reset flags
        self.done = False

        # reset morphology
        pb.resetBasePositionAndOrientation(self.drone1.quadrotor, self.startPosition1,
                                            pb.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        # Get observation and position
        position, orientation, linearVel, angularVel = self.drone1.get_observation()
        _observation = np.concatenate((position, orientation, linearVel, angularVel))

        return _observation

    def render(self):
        pass

    def close(self):
        pb.disconnect(self.client)
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]