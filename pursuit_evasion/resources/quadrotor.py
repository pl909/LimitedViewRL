from turtle import position
import pybullet as pb
import numpy as np
import os

from .utils import *

class Quadrotor:
    def __init__(self, urdf, startPosition, client, draw_debug=False):
        # Store both the client object and ID
        self.client = client
        self.client_id = client if isinstance(client, int) else client._client
        
        try:
            self.quadrotor = pb.loadURDF(
                fileName=urdf,
                basePosition=startPosition,
                baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.client_id
            )
            
            if self.quadrotor < 0:
                raise ValueError(f"Failed to load URDF file from {urdf}")
                
            # Only draw debug frame if requested and in GUI mode
            if draw_debug and pb.getConnectionInfo(self.client_id)["connectionMethod"] == pb.GUI:
                draw_frame(self.client_id, self.quadrotor, -1)
                
        except Exception as e:
            print(f"Error loading quadrotor URDF: {str(e)}")
            print(f"Attempted to load from path: {urdf}")
            raise

    def get_ids(self):
        return self.client_id, self.quadrotor

    def apply_action(self, action):
        forceZ = action[0]
        torqueX = action[1]
        torqueY = action[2]
        torqueZ = action[3]

        # Apply movement inputs
        controlInput = np.array([forceZ, torqueX, torqueY, torqueZ])
        force_torque_control(self.client_id, self.quadrotor, controlInput)

    def get_observation(self):
        # Get observation
        robotObs = get_robot_state(self.client_id, self.quadrotor)
        position, orientation, linearVel, angularVel = robotObs
        return position, orientation, linearVel, angularVel