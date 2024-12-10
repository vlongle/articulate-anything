from articulate_anything.api.odio_urdf import *


def partnet_48167(intput_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - door
    - furniture_body

    Object: a cabinet with a door
    Targetted affordance: "door"
    """
    pred_robot = Robot(input_dir=intput_dir, name="cabinet")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['furniture_body'])
    pred_robot.add_joint(Joint("base_to_furniture_body",
                         Parent("base"),
                         Child("furniture_body"),
                         type="fixed"),
                         )

    pred_robot.add_link(links['door'])
    # cabinet doors go in front of the furniture body.
    pred_robot.place_relative_to('door', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)

    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # in the groundtruth video, the door opens up while its bottom part still
    # attached to the body so the pivot point needs to be Front-Left-Bottom (FLB)
    # or Front-Right-Bottom (FRB)
    door_bb = pred_robot.get_bounding_boxes(
        ['door'], include_dim=False)['door']
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[3]  # Front-Right-Bottom (FRB)

    pred_robot.make_revolute_joint(
        "door",
        "furniture_body",
        global_axis=[
            0,
            0,
            # pivot-axis relationship:
            # since the pivot is on the bottom and we want to open outward, set
            # axis to positive
            1,
        ],  # The door opens by rotating along the left-right axis, which is y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot
