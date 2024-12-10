from articulate_anything.api.odio_urdf import *


def partnet_100963(intput_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 5
    Robot Link Summary:
    - base
    - toggle_button
    - toggle_button_2
    - toggle_button_3
    - switch_frame

    Object: a light switch with three toggle buttons
    Targetted affordance: "toggle_button"
    """
    pred_robot = Robot(input_dir=intput_dir, name="light_switch")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["switch_frame"])
    pred_robot.add_joint(
        Joint("base_to_switch_frame", Parent("base"),
              Child("switch_frame"), type="fixed"),
    )

    pred_robot.add_link(links["toggle_button"])
    pred_robot.place_relative_to(
        "toggle_button", "switch_frame", placement="inside", clearance=0.0
    )

    pred_robot.add_link(links["toggle_button_2"])
    pred_robot.place_relative_to(
        "toggle_button_2", "switch_frame", placement="inside", clearance=0.0
    )

    pred_robot.add_link(links["toggle_button_3"])
    pred_robot.place_relative_to(
        "toggle_button_3", "switch_frame", placement="inside", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # the button switch always rotates along the center of the switch frame
    targetted_affordance = "toggle_button"
    switch_frame_bb = pred_robot.get_bounding_boxes(
        ["switch_frame"], include_dim=False)["switch_frame"]
    aabb_min, aabb_max = switch_frame_bb
    # get the center of the switch frame
    pivot_point = [(aabb_min[0] + aabb_max[0]) / 2, (aabb_min[1] +
                                                     aabb_max[1]) / 2, (aabb_min[2] + aabb_max[2]) / 2]

    pred_robot.make_revolute_joint(
        targetted_affordance,
        "switch_frame",
        global_axis=[
            0,
            # pivot-axis relationship:
            1,
            0,
        ],  # the toggle_button moves up and down so rotates along the left-right axis,
        # which is the y-axis
        pivot_point=pivot_point,
        lower_angle_deg=0,
        upper_angle_deg=24,  # move just a little bit
    )

    return pred_robot
