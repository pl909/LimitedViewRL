import pybullet as pb
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np

from utils import *
# from linear_control_exercise import compute_control_gain_exercise

# System parameters
mass = .5
inertial = (0.0023, 0.0023, 0.004)
arm_length = .1750
drag_coefficient = .01
propeller_locations = [(.1750, 0, 0), (0, .1750, 0),
                       (-.1750, 0, 0), (0, -.1750, 0)]

if __name__ == '__main__':
    # Initialize the simulator
    pb_client = initializeGUI(enable_gui=True)

    # Add plane and robot models
    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF('./robot_models/quadrotor.urdf', [0, 1, 1])

    # Draw robot frame
    draw_frame(pb_client, robot_id, -1)

    # Define debug parameters
    parameter_info = []
    parameter_info.append(
        {'name': 'force (z-axis)', 'lower_limit': -10, 'upper_limit': 10, 'start_value': 0})
    parameter_info.append({'name': 'torque (x-axis)',
                           'lower_limit': -1, 'upper_limit': 1, 'start_value': 0})
    parameter_info.append({'name': 'torque (y-axis)',
                           'lower_limit': -1, 'upper_limit': 1, 'start_value': 0})
    parameter_info.append({'name': 'torque (z-axis)',
                           'lower_limit': -1, 'upper_limit': 1, 'start_value': 0})
    debug_parameter_ids = add_debug_parameters(pb_client, parameter_info)

    # Initialize debug texts
    robot_state = get_robot_state(pb_client, robot_id)
    position, orientation, _, _ = robot_state
    debug_text_xyz_id = pb_client.addUserDebugText('x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(*position),
                                                   textPosition=(.1, 0, .4),
                                                   textColorRGB=(0, 0, 0),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=-1)
    debug_text_rpy_id = pb_client.addUserDebugText('r:{:.2f}, p:{:.2f}, y:{:.2f}'.format(*orientation),
                                                   textPosition=(.1, 0, .2),
                                                   textColorRGB=(0, 0, 0),
                                                   parentObjectUniqueId=robot_id,
                                                   parentLinkIndex=-1)

    # Main loop
    for time_step in range(100000):

        # Update debug texts
        robot_state = get_robot_state(pb_client, robot_id)
        position, orientation, _, _ = robot_state
        debug_text_xyz_id = pb_client.addUserDebugText('x:{:.2f}, y:{:.2f}, z:{:.2f}'.format(*position),
                                                        textPosition=(.1, 0, .4),
                                                        textColorRGB=(0, 0, 0),
                                                        parentObjectUniqueId=robot_id,
                                                        parentLinkIndex=-1,
                                                        replaceItemUniqueId=debug_text_xyz_id)

        debug_text_rpy_id = pb_client.addUserDebugText('r:{:.2f}, p:{:.2f}, y:{:.2f}'.format(*orientation),
                                                       textPosition=(.1, 0, .2),
                                                       textColorRGB=(0, 0, 0),
                                                       parentObjectUniqueId=robot_id,
                                                       parentLinkIndex=-1,
                                                       replaceItemUniqueId=debug_text_rpy_id)

        # Apply control
        force_z = pb_client.readUserDebugParameter(
            itemUniqueId=debug_parameter_ids[0])
        torque_x = pb_client.readUserDebugParameter(
            itemUniqueId=debug_parameter_ids[1])
        torque_y = pb_client.readUserDebugParameter(
            itemUniqueId=debug_parameter_ids[2])
        torque_z = pb_client.readUserDebugParameter(
            itemUniqueId=debug_parameter_ids[3])

        control_input = (force_z, torque_x, torque_y, torque_z)
        force_torque_control(pb_client, robot_id, control_input)

        pb.stepSimulation()
        time.sleep(1 / 240)

    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)
    pb.disconnect()
