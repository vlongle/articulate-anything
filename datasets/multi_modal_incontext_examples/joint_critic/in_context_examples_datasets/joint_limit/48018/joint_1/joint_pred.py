from articulate_anything.api.odio_urdf import *


def partnet_48018(intput_dir, links):
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
    target_door_link = "door_2"
    door_bb = pred_robot.get_bounding_boxes([target_door_link], include_dim=False)[
        target_door_link
    ]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[2]  # Front-Left-Bottom (FLB)

    pred_robot.make_revolute_joint(
        target_door_link,
        "furniture_body",
        global_axis=[
            0,
            0,
            # pivot-axis relationship:
            # since the pivot is on the right and we want to open outward, set
            # axis to positive
            1,  # opening up
        ],  # The cabinet door swings open and closed around the vertical axis,
        # which is the z-axis.
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot
