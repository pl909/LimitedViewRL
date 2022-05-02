from turtle import position
import pybullet as pb
import numpy as np
import os

from .utils import *

class Quadrotor:
    def __init__(self, urdf, startPosition, client):
        self.client = client
        # f_name = os.path.join(os.path.dirname(__file__),
        #                       'quadrotor.urdf')
        # self.pbClient = initializeGUI(enable_gui=True)
        self.quadrotor = pb.loadURDF(fileName=urdf,
                              basePosition=startPosition)
                            #   physicsClientId=initializeGUI(enable_gui=True))

        # Draw robot frame
        draw_frame(self.client, self.quadrotor, -1)

    def get_ids(self):
        return self.client, self.quadrotor

    def apply_action(self, action):

        forceZ = action[0]
        torqueX = action[1]
        torqueY = action[2]
        torqueZ = action[3]

        # Apply movement inputs
        controlInput = np.array([forceZ, torqueX, torqueY, torqueZ])
        force_torque_control(self.client, self.quadrotor, controlInput)

    
    def get_observation(self):
        
        # Get observation
        robotObs = get_robot_state(self.client, self.quadrotor)
        position, orientation, linearVel, angularVel = robotObs
        # observation =  np.array([pos[0], pos[1], pos[2], orn[0], orn[1], orn[2]])

        return position, orientation, linearVel, angularVel