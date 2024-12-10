from articulate_anything.api.odio_urdf import *

# code of the partnet_{object_id} function


def partnet_100963(intput_dir, links):
    """
    No. masked_links: 5
    Robot Link Summary:
    - base
    - toggle_button
    - toggle_button_2
    - toggle_button_3
    - switch_frame

    Object: a light switch with three toggle buttons
    Affordance: the toggle buttons can be toggled up and down
    """
    pred_robot = Robot(input_dir=intput_dir, name="light_switch")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['switch_frame'])
    pred_robot.add_joint(Joint("base_to_switch_frame",
                         Parent("base"),
                         Child("switch_frame"),
                         type="fixed"),
                         )

    # =============================================
    # partnet mobility peculiarity: need to always call this function
    # after the first joint to orientate the robot correctly
    pred_robot.align_robot_orientation()
    # =============================================

    pred_robot.add_link(links['toggle_button'])
    pred_robot.place_relative_to('toggle_button', 'switch_frame',
                                 placement="inside",
                                 clearance=0.0)

    pred_robot.add_link(links['toggle_button_2'])
    pred_robot.place_relative_to('toggle_button_2', 'switch_frame',
                                 placement="inside",
                                 clearance=0.0)

    pred_robot.add_link(links['toggle_button_3'])
    pred_robot.place_relative_to('toggle_button_3', 'switch_frame',
                                 placement="inside",
                                 clearance=0.0)
    return pred_robot
