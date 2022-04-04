import pybullet as pb
import numpy as np
import os


class Quadrotor:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__),
                              'quadrotor.urdf')
        self.quadrotor = pb.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.1],
                              physicsClientId=client)

    def get_ids(self):
        return self.client, self.car

    def apply_action(self, action):
        pass
    
    def get_observation(self):
        pass