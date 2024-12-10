from articulate_anything.api.odio_urdf import *


def partnet_4590(input_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - screen
    - display_base

    Object: a computer monitor
    Targetted affordance: screen
    """
    pred_robot = Robot(input_dir=input_dir, name="monitor")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['display_base'])
    pred_robot.add_joint(Joint("base_to_display_base",
                               Parent("base"),
                               Child("display_base"),
                               type="fixed"),
                         )

    # screen goes on top of the display_base. No clearance or further adjustment needed
    pred_robot.add_link(links['screen'])
    pred_robot.place_relative_to('screen', 'display_base',
                                 placement="above",
                                 clearance=0.0)
    # ====================JOINT PREDICTION====================
    # The screen tilts up and down.
    # The axis of motion is along the y-axis.
    pred_robot.make_revolute_joint(
        "screen",
        "display_base",
        global_axis=[0, 1, 0],
        lower_angle_deg=-45,
        upper_angle_deg=45,
    )
    return pred_robot
