from articulate_anything.api.odio_urdf import *


def partnet_103739(intput_dir, links):
    """
    No. masked_links: 6
    Robot Link Summary:
    - base
    - blade
    - blade_2
    - blade_3
    - blade_4
    - knife_body

    Object: a knife with a blade that can rotate
    """
    pred_robot = Robot(input_dir=intput_dir, name="knife")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['knife_body'])
    pred_robot.add_joint(Joint("base_to_knife_body",
                         Parent("base"),
                         Child("knife_body"),
                         type="fixed"),
                         )

    # =============================================
    # partnet mobility peculiarity: need to always call this function
    # after the first joint to orientate the robot correctly
    pred_robot.align_robot_orientation()
    # =============================================

    for blade_link in ['blade', 'blade_2', 'blade_3', 'blade_4']:
        pred_robot.add_link(links[blade_link])
        pred_robot.place_relative_to(blade_link, 'knife_body',
                                     placement="inside",
                                     clearance=0.0)
    return pred_robot
