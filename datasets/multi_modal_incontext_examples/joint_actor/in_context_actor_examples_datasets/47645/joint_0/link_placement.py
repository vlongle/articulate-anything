from articulate_anything.api.odio_urdf import *
# code of the partnet_{object_id} function


def partnet_47645(intput_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - lid
    - box_body


    Object: a treasure chest
    Affordance: lid can be opened and closed
    """
    pred_robot = Robot(input_dir=intput_dir, name="treasure_chest")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['box_body'])
    pred_robot.add_joint(Joint("base_to_box_body",
                         Parent("base"),
                         Child("box_body"),
                         type="fixed"),
                         )

    # =============================================
    # partnet mobility peculiarity: need to always call this function
    # after the first joint to orientate the robot correctly
    pred_robot.align_robot_orientation()
    # =============================================

    # lid goes on top of the box. No clearance or further adjustment needed
    pred_robot.add_link(links['lid'])
    pred_robot.place_relative_to('lid', 'box_body',
                                 placement="above",
                                 clearance=0.0)
    return pred_robot
