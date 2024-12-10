from articulate_anything.api.odio_urdf import *


def partnet_38516(intput_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - door
    - furniture_body

    Object: A cabinet with a single door on the front
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
    pred_robot.place_relative_to('door', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)
    # ====================JOINT PREDICTION====================
    # we're rotating at an edge of the door instead of the center
    # so we need to specify the pivot_point
    door_bb = pred_robot.get_bounding_boxes(
        ['door'], include_dim=False)['door']
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[3]  # Bottom-Front-Right

    pred_robot.make_revolute_joint(
        'door', 'furniture_body',
        # The cabinet door swings open and closed around the vertical axis,
        global_axis=[0, 0, 1],
        # which is the z-axis.
        lower_angle_deg=0, upper_angle_deg=90,
        pivot_point=pivot_point,
    )
    return pred_robot
