from articulate_anything.api.odio_urdf import *


def partnet_41003(intput_dir, links):
    """
    No. masked_links: 6
    Robot Link Summary:
    - base
    - door
    - door_2
    - door_3
    - door_4
    - furniture_body

    Object: a cabinet with 4 doors
    Affordance: the doors can be rotated to open and close
    """
    pred_robot = Robot(input_dir=intput_dir, name="cabinet")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['furniture_body'])
    pred_robot.add_joint(Joint("base_to_furniture_body",
                         Parent("base"),
                         Child("furniture_body"),
                         type="fixed"),
                         )

    # =============================================
    # partnet mobility peculiarity: need to always call this function
    # after the first joint to orientate the robot correctly
    pred_robot.align_robot_orientation()
    # =============================================
    pred_robot.add_link(links['door'])
    # cabinet doors go in front of the furniture body.
    pred_robot.place_relative_to('door', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)

    pred_robot.add_link(links['door_2'])
    pred_robot.place_relative_to('door_2', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)

    pred_robot.add_link(links['door_3'])
    pred_robot.place_relative_to('door_3', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)

    pred_robot.add_link(links['door_4'])
    pred_robot.place_relative_to('door_4', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)
    return pred_robot
