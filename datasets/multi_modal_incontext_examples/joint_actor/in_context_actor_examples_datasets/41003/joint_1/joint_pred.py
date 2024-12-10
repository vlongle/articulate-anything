from articulate_anything.api.odio_urdf import *


def partnet_41003(intput_dir, links, joint_id="joint_1"):
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
    Targetted affordance: "door_2"
    """
    pred_robot = Robot(input_dir=intput_dir, name="cabinet")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["furniture_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_furniture_body",
            Parent("base"),
            Child("furniture_body"),
            type="fixed",
        ),
    )

    pred_robot.add_link(links["door"])
    # cabinet doors go in front of the furniture body.
    pred_robot.place_relative_to(
        "door", "furniture_body", placement="front", clearance=0.0
    )

    pred_robot.add_link(links["door_2"])
    pred_robot.place_relative_to(
        "door_2", "furniture_body", placement="front", clearance=0.0
    )

    pred_robot.add_link(links["door_3"])
    pred_robot.place_relative_to(
        "door_3", "furniture_body", placement="front", clearance=0.0
    )

    pred_robot.add_link(links["door_4"])
    pred_robot.place_relative_to(
        "door_4", "furniture_body", placement="front", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # joint_type: 'revolute', the top-right door opens by rotating in an arc
    # joint_axis: the door rotates along the z-axis
    # joint_origin: the door is attached to the body at the **right** and front of the door. front is optional
    # but RIGHT is a must. so pivot point needs to either be Front-Right-Bottom (FRB)
    # or Front-Right-Top (FRT)
    # joint_limit: the door opens outward
    target_door_link = "door_2"
    door_bb = pred_robot.get_bounding_boxes([target_door_link], include_dim=False)[
        target_door_link
    ]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[3]  # Front-Right-Bottom (FRB)

    pred_robot.make_revolute_joint(
        target_door_link,
        "furniture_body",
        global_axis=[
            0,
            0,
            # pivot-axis relationship:
            # In our convention, **right** is positive and **left** is negative.
            # Since the pivot is on the right along the z-axis and we want to open outward, set
            # axis to positive
            1,  # opening up
        ],  # The cabinet door swings open and closed around the vertical axis,
        # which is the z-axis.
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot
