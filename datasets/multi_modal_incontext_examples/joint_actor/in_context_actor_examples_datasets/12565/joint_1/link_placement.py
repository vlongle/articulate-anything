from articulate_anything.api.odio_urdf import *
# code of the partnet_{object_id} function


def partnet_12565(intput_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - dishwasher_body
    - door


    Object: a dishwasher with a door that can be opened and closed
    Affordance: the door can be rotated to open and close
    """
    pred_robot = Robot(input_dir=intput_dir, name="dishwasher")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['dishwasher_body'])
    pred_robot.add_joint(Joint("base_to_dishwasher_body",
                         Parent("base"),
                         Child("dishwasher_body"),
                         type="fixed"),
                         )

    # =============================================
    # partnet mobility peculiarity: need to always call this function
    # after the first joint to orientate the robot correctly
    pred_robot.align_robot_orientation()
    # =============================================

    pred_robot.add_link(links['door'])
    pred_robot.place_relative_to('door', 'dishwasher_body',
                                 placement="front",
                                 clearance=0.0)
    return pred_robot
