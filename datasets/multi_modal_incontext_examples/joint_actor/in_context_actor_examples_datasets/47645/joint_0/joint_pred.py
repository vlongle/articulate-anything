from articulate_anything.api.odio_urdf import *


def partnet_47645(intput_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - lid
    - box_body


    Object: a treasure chest
    Targetted affordance: "lid"
    """
    pred_robot = Robot(input_dir=intput_dir, name="treasure_chest")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["box_body"])
    pred_robot.add_joint(
        Joint("base_to_box_body", Parent("base"),
              Child("box_body"), type="fixed"),
    )

    # lid goes on top of the box. No clearance or further adjustment needed
    pred_robot.add_link(links["lid"])
    pred_robot.place_relative_to(
        "lid", "box_body", placement="above", clearance=0.0)
    # ========================================================

    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # joint_type: 'revolute', the lid rotates in an arc
    # joint_axis: the lid rotates along the y-axis
    # joint_origin: the lid is attached to the body at the **back** and bottom of the lid. Bottom is optional
    # but BACK is a must.
    # so pivot point needs to either be Back-Left-Bottom (BLB) or Back-Right-Bottom (BRB)
    # joint_limit: the lid opens upward
    lid_bb = pred_robot.get_bounding_boxes(["lid"], include_dim=False)["lid"]
    lid_vertices = compute_aabb_vertices(*lid_bb)
    pivot_point = lid_vertices[0]  # Back-Left-Bottom (BLB)

    pred_robot.make_revolute_joint(
        "lid",
        "box_body",
        global_axis=[
            0,
            # pivot-axis relationship:
            # In our convention, **back** is negative and **front** is positive,
            # and our pivot point is back so set axis to negative
            # for the lid to open upward
            -1,
            0,
        ],  # the lid of the treasure chest opens up and down so rotates along the left-right axis,
        # which is the y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,
        pivot_point=pivot_point,
    )

    return pred_robot
