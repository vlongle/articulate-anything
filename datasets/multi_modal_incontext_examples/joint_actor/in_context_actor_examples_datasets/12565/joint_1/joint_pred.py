from articulate_anything.api.odio_urdf import *


def partnet_12565(intput_dir, links, joint_id="joint_1"):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - dishwasher_body
    - door


    Object: a dishwasher
    Targetted affordance: "door"
    """
    pred_robot = Robot(input_dir=intput_dir, name="dishwasher")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["dishwasher_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_dishwasher_body",
            Parent("base"),
            Child("dishwasher_body"),
            type="fixed",
        ),
    )

    pred_robot.add_link(links["door"])
    pred_robot.place_relative_to(
        "door", "dishwasher_body", placement="front", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # joint_type: 'revolute', the door opens by rotating in an arc
    # joint_axis: the door rotates along the y-axis
    # joint_origin: the door is attached to the body at the **bottom** and front of the door. front is optional
    # but RIGHT is a must. so pivot point needs to either be Front-Right-Bottom (FRB)
    # or Front-Left-Bottom (FLB)
    # joint_limit: the door opens outward
    door_bb = pred_robot.get_bounding_boxes(
        ["door"], include_dim=False)["door"]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[2]  # Front-Left-Bottom (FLB)

    pred_robot.make_revolute_joint(
        "door",
        "dishwasher_body",
        global_axis=[
            0,
            # pivot-axis relationship:
            # In our convention, **front** is positive and **back** is negative.
            # Since the pivot is on the front along the y-axis and we want to open outward, set
            # axis to positive
            1,
            0,
        ],  # The door opens by rotating along the left-right axis, which is y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot
