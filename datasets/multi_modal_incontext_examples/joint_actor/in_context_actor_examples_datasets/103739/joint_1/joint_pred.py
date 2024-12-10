from articulate_anything.api.odio_urdf import *


def partnet_103739(intput_dir, links, joint_id="joint_1"):
    """
    No. masked_links: 6
    Robot Link Summary:
    - base
    - blade
    - blade_2
    - blade_3
    - blade_4
    - knife_body

    Object: a swiss army knife
    Targetted affordance: "blade_2"
    """
    pred_robot = Robot(input_dir=intput_dir, name="knife")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['knife_body'])
    pred_robot.add_joint(Joint("base_to_knife_body",
                         Parent("base"),
                         Child("knife_body"),
                         type="fixed"),
                         )

    for blade_link in ['blade', 'blade_2', 'blade_3', 'blade_4']:
        pred_robot.add_link(links[blade_link])
        pred_robot.place_relative_to(blade_link, 'knife_body',
                                     placement="inside",
                                     clearance=0.0)
    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # joint_type: 'revolute', the blade opens by rotating in an arc
    # joint_axis: the blade rotates along the z-axis
    # joint_origin: the blade is attached to the body at the back and right of the blade.
    # the pivot point needs to be either Back-Right-Bottom (BRB) or Back-Right-Top (BRT)
    # joint_limit: the blade opens outward

    blade_bb = pred_robot.get_bounding_boxes(
        ["blade_2"], include_dim=False)["blade_2"]
    blade_vertices = compute_aabb_vertices(*blade_bb)
    pivot_point = blade_vertices[1]  # Back-Right-Bottom (BRB)

    pred_robot.make_revolute_joint(
        "blade_2",
        "knife_body",
        global_axis=[
            0,
            0,
            # pivot-axis relationship:
            # In our convention, **front** is positive and **back** is negative.
            # Since the pivot is on the back along the z-axis and we want to open outward, set
            # axis to negative
            -1,
        ],  # The blade opens by rotating along the left-right axis, which is y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot
