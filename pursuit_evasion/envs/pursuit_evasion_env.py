from turtle import position
import gym
import numpy as np
import pybullet as pb
import time
import os

from resources import utils as r
from resources import Quadrotor

# PursuitEvasion-v0

class PursuitEvasionEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self, drone_type='pursuer', trainingMode=False):
        self.done = False
        self.timer = 0
        self.drone_type = drone_type
        
        # Action space remains same for both drones
        self.action_space = gym.spaces.box.Box(
            low=np.array([-10, -1, -1, -1]),
            high=np.array([10, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32)

        # Observation space now includes both drones' positions and states
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-np.inf] * 24),  # 12 states for each drone
            high=np.array([np.inf] * 24),
            shape=(24,),
            dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        # Initialize PyBullet
        if trainingMode:
            connectionType = pb.DIRECT
            draw_debug = False
        else:
            connectionType = pb.GUI
            draw_debug = True
            
        self.pbClient = pb.connect(connectionType)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.pbClient)
        pb.setRealTimeSimulation(0, physicsClientId=self.pbClient)

        # Get the absolute path to the resources directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resources_dir = os.path.join(os.path.dirname(current_dir), 'resources')
        
        # Load plane
        plane_path = os.path.join(resources_dir, 'robot_models', 'plane.urdf')
        self.planeId = pb.loadURDF(plane_path, physicsClientId=self.pbClient)
        
        # Initialize drone positions
        self.pursuer_start = np.array([0, 1, 1])
        self.evader_start = np.array([5, 5, 1])
        
        # Fix the path to the quadrotor URDF
        quadrotor_path = os.path.join(resources_dir, 'robot_models', 'quadrotor.urdf')
        
        if not os.path.exists(quadrotor_path):
            raise FileNotFoundError(f"Could not find quadrotor URDF at {quadrotor_path}")
            
        # Create drones with correct paths
        self.pursuer = Quadrotor(
            urdf=quadrotor_path,
            startPosition=self.pursuer_start,
            client=self.pbClient,
            draw_debug=draw_debug
        )
        
        self.evader = Quadrotor(
            urdf=quadrotor_path,
            startPosition=self.evader_start,
            client=self.pbClient,
            draw_debug=draw_debug
        )

        # Initial distance between drones
        self.distance = self._get_distance()

    def _get_distance(self):
        pursuer_pos, _, _, _ = self.pursuer.get_observation()
        evader_pos, _, _, _ = self.evader.get_observation()
        return np.linalg.norm(pursuer_pos - evader_pos)

    def _get_full_state(self):
        p_pos, p_ori, p_vel, p_ang = self.pursuer.get_observation()
        e_pos, e_ori, e_vel, e_ang = self.evader.get_observation()
        return np.concatenate((p_pos, p_ori, p_vel, p_ang, e_pos, e_ori, e_vel, e_ang))

    def step(self, action):
        reward = 0
        self.done = False
        
        # Apply action to the controlled drone
        if self.drone_type == 'pursuer':
            self.pursuer.apply_action(action)
            # Evader uses simple escape policy
            evader_action = self._simple_escape_policy()
            self.evader.apply_action(evader_action)
        else:
            self.evader.apply_action(action)
            # Pursuer uses simple chase policy
            pursuer_action = self._simple_chase_policy()
            self.pursuer.apply_action(pursuer_action)

        pb.stepSimulation()
        time.sleep(1/240)
        self.timer += 1

        # Get new distance
        new_distance = self._get_distance()
        
        # Calculate rewards based on drone type
        if self.drone_type == 'pursuer':
            if new_distance < 0.5:  # Capture threshold
                reward = 100
                self.done = True
            elif new_distance < self.distance:  # Getting closer
                reward = 1
            else:  # Getting further
                reward = -1
        else:  # evader
            if new_distance < 0.5:  # Captured
                reward = -100
                self.done = True
            elif new_distance > self.distance:  # Escaping
                reward = 1
            else:  # Getting caught
                reward = -1

        # Time limit
        if self.timer >= 500:
            self.done = True
            self.timer = 0

        self.distance = new_distance
        observation = self._get_full_state()
        info = {'distance': self.distance}

        return observation, reward, self.done, info

    def _simple_chase_policy(self):
        # Simple pursuing policy when training evader
        pursuer_pos, _, _, _ = self.pursuer.get_observation()
        evader_pos, _, _, _ = self.evader.get_observation()
        direction = evader_pos - pursuer_pos
        return np.array([8, *direction[:3]/np.linalg.norm(direction)])

    def _simple_escape_policy(self):
        # Simple evading policy when training pursuer
        pursuer_pos, _, _, _ = self.pursuer.get_observation()
        evader_pos, _, _, _ = self.evader.get_observation()
        direction = -(pursuer_pos - evader_pos)
        return np.array([8, *direction[:3]/np.linalg.norm(direction)])

    def reset(self):
        self.done = False
        self.timer = 0
        
        # Reset both drones
        pb.resetBasePositionAndOrientation(self.pursuer.quadrotor, self.pursuer_start,
                                         pb.getQuaternionFromEuler([0, 0, 0]))
        pb.resetBasePositionAndOrientation(self.evader.quadrotor, self.evader_start,
                                         pb.getQuaternionFromEuler([0, 0, 0]))
        
        self.distance = self._get_distance()
        return self._get_full_state()

    def render(self):
        pass

    def close(self):
        pb.disconnect(self.pbClient)
    
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]