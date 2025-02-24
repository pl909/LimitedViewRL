import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

import control


def set_camera(distance, yaw, pitch, position):
    pb.resetDebugVisualizerCamera(cameraDistance=distance,
                                  cameraYaw=yaw,
                                  cameraPitch=pitch,
                                  cameraTargetPosition=position)


def initializeGUI(enable_gui=True, connection=pb.DIRECT):
    """Initialize PyBullet physics server."""
    if enable_gui:
        physicsClient = pb.connect(pb.GUI)
    else:
        physicsClient = pb.connect(connection)
        
    pb.setGravity(0, 0, -9.81, physicsClientId=physicsClient)
    pb.setRealTimeSimulation(0, physicsClientId=physicsClient)
    return physicsClient


def mark_waypoint(pb_client, waypoint, axis_length=.2, line_width=1):
    waypoint = np.array(waypoint)

    pb_client.addUserDebugLine(lineFromXYZ=waypoint,
                               lineToXYZ=waypoint + (axis_length, 0, 0),
                               lineWidth=line_width,
                               lineColorRGB=(1, 0, 0))

    pb_client.addUserDebugLine(lineFromXYZ=waypoint,
                               lineToXYZ=waypoint+(0,axis_length,0),
                               lineWidth=line_width,
                               lineColorRGB=(0,1,0))

    pb_client.addUserDebugLine(lineFromXYZ=waypoint,
                               lineToXYZ=waypoint+(0,0,axis_length),
                               lineWidth=line_width,
                               lineColorRGB=(0,0,1))
    
def draw_frame(pb_client, robot_id, link_id):
    """Draw coordinate frame for debugging."""
    try:
        # Ensure we're using the client ID number
        if isinstance(pb_client, int):
            client_id = pb_client
        else:
            client_id = pb_client._client

        if pb.getConnectionInfo(client_id)["connectionMethod"] != pb.GUI:
            return  # Don't draw in DIRECT mode

        pos, orn = pb.getBasePositionAndOrientation(robot_id, physicsClientId=client_id)
        rot_matrix = pb.getMatrixFromQuaternion(orn)
        
        # Draw three lines for coordinate frame
        axis_length = 0.2
        axis_width = 2
        
        # X axis - Red
        pb.addUserDebugLine(
            pos,
            [pos[0] + rot_matrix[0] * axis_length,
             pos[1] + rot_matrix[1] * axis_length,
             pos[2] + rot_matrix[2] * axis_length],
            [1, 0, 0],
            axis_width,
            physicsClientId=client_id
        )
        
        # Y axis - Green
        pb.addUserDebugLine(
            pos,
            [pos[0] + rot_matrix[3] * axis_length,
             pos[1] + rot_matrix[4] * axis_length,
             pos[2] + rot_matrix[5] * axis_length],
            [0, 1, 0],
            axis_width,
            physicsClientId=client_id
        )
        
        # Z axis - Blue
        pb.addUserDebugLine(
            pos,
            [pos[0] + rot_matrix[6] * axis_length,
             pos[1] + rot_matrix[7] * axis_length,
             pos[2] + rot_matrix[8] * axis_length],
            [0, 0, 1],
            axis_width,
            physicsClientId=client_id
        )
    except Exception as e:
        print(f"Warning: Could not draw debug frame: {str(e)}")

def add_debug_parameters(pb_client, parameter_info):
    debug_parameter_ids = []
    for data in parameter_info:
        debug_parameter_id = pb_client.addUserDebugParameter(paramName=data['name'],
                                                             rangeMin=data['lower_limit'],
                                                             rangeMax=data['upper_limit'],
                                                             startValue=data['start_value'])
        
        debug_parameter_ids.append(debug_parameter_id)

    return debug_parameter_ids

def get_pose(pb_client, robot_id):
    position, orientation = pb_client.getBasePositionAndOrientation(bodyUniqueId=robot_id)
    orientation = pb.getEulerFromQuaternion(orientation)

    return (np.array(position), np.array(orientation))

def get_velocity(pb_client, robot_id):
    position, orientation = get_pose(pb_client, robot_id)
    linear_velocity, angular_velocity = pb_client.getBaseVelocity(bodyUniqueId=robot_id)

    phi, theta, psi = orientation
    rotation_matrix = np.matrix([[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
                                 [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
                                 [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])
    linear_velocity_local_frame = np.squeeze(np.asarray(np.inner(rotation_matrix.T, linear_velocity)))

    T_matrix = np.matrix([[1, 0, -np.sin(theta)],
                          [0, np.cos(phi), np.sin(phi)*np.cos(theta)],
                          [0, -np.sin(phi), np.cos(phi)*np.cos(theta)]])

    angular_velocity_local_frame = np.squeeze(np.asarray(np.inner(T_matrix, angular_velocity)))

    return (linear_velocity_local_frame, angular_velocity_local_frame)

def get_robot_state(pb_client, robot_id):
    """Get robot state from PyBullet."""
    # Ensure we're using the client ID number
    if isinstance(pb_client, int):
        client_id = pb_client
    else:
        client_id = pb_client._client

    position, orientation = pb.getBasePositionAndOrientation(
        robot_id, 
        physicsClientId=client_id
    )
    velocity = pb.getBaseVelocity(
        robot_id, 
        physicsClientId=client_id
    )
    
    linear_vel, angular_vel = velocity
    
    return np.array(position), np.array(orientation), np.array(linear_vel), np.array(angular_vel)

def get_joint_info(pb_client, robot_id):
    number_of_joints = pb_client.getNumJoints(bodyUniqueId=robot_id)
    joint_info = []
    for joint_index in range(number_of_joints):
        return_data = pb_client.getJointInfo(bodyUniqueId=robot_id,
                                             jointIndex=joint_index)
    
        joint_index, joint_name = return_data[:2]
        joint_lower_limit = return_data[8]
        joint_upper_limit = return_data[9]
        joint_info.append({'index': joint_index,
                           'name': joint_name,
                           'limit': (joint_lower_limit, joint_upper_limit)})

    return joint_info

def propeller_control(pb_client, robot_id, propeller_force, arm_length=.1750, drag_coefficient=.01):
    body_force = np.sum(propeller_force)
    body_torque = (arm_length*(-propeller_force[1]+propeller_force[3]),
                   arm_length*(-propeller_force[0]+propeller_force[2]),
                   drag_coefficient*(-propeller_force[0]+propeller_force[1]-propeller_force[2]+propeller_force[3]))
                
    pb_client.applyExternalForce(robot_id,
                                 -1,
                                 forceObj=[0,0,body_force],
                                 posObj=(0,0,0),
                                 flags=pb.LINK_FRAME)

    pb_client.applyExternalTorque(robot_id,
                                  -1,
                                  torqueObj=body_torque,
                                  flags=pb.LINK_FRAME)

def force_torque_control(pb_client, robot_id, control_input):
    """Apply force and torque control to robot."""
    # Ensure we're using the client ID number
    if isinstance(pb_client, int):
        client_id = pb_client
    else:
        client_id = pb_client._client

    force = [0, 0, control_input[0]]  # Only Z force
    torque = control_input[1:]  # X, Y, Z torques
    
    pb.applyExternalForce(
        robot_id,
        -1,  # Link ID (-1 for base)
        force,
        [0, 0, 0],  # Position relative to COM
        pb.LINK_FRAME,
        physicsClientId=client_id
    )
    
    pb.applyExternalTorque(
        robot_id,
        -1,  # Link ID (-1 for base)
        torque,
        pb.LINK_FRAME,
        physicsClientId=client_id
    )

def compute_control_gain():
    mass = .5
    inertial = (0.0023, 0.0023, 0.004)
    gravity = 10

    A = np.matrix([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, gravity, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, -gravity, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    B = np.matrix([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1/mass, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 1/inertial[0], 0, 0],
                   [0, 0, 1/inertial[1], 0],
                   [0, 0, 0, 1/inertial[2]]])

    Q = np.eye(12)
    R = .1*np.eye(4)

    K, S, E = control.lqr(A, B, Q, R)
    return -K
    
def compute_stabilizing_feedback(control_gain, robot_state, destination):
    x = np.concatenate((robot_state[0]-np.array(destination),
                        robot_state[2],
                        robot_state[1],
                        robot_state[3]))
    
    return np.squeeze(np.asarray(np.inner(control_gain, x))) + np.array([5, 0, 0, 0])

