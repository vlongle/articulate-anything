from articulate_anything.api.odio_urdf import *


def partnet_100248(input_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - handle
    - suitcase_body

    Object: a suitcase with a handle
    Targetted affordance: "handle"
    """
    pred_robot = Robot(input_dir=input_dir, name="suitcase")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["suitcase_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_suitcase_body",
            Parent("base"),
            Child("suitcase_body"),
            type="fixed",
        ),
    )

    # we actually want to place the handle inside the suitcase, and not above.
    # so we need to shift the handle down by its height. We first place it above
    # then compute the height and shift it down by that amount using clearance
    # and remove the previous placement
    pred_robot.add_link(links["handle"])
    pred_robot.place_relative_to(
        "handle", "suitcase_body", placement="above", clearance=0.0
    )
    bb = pred_robot.get_bounding_boxes(["handle"], include_dim=True)["handle"]
    bb_dim = bb[1]
    pred_robot.place_relative_to(
        "handle", "suitcase_body", placement="above", clearance=-bb_dim["height"]
    )  # for prismatic joint, we need to specify the lower and upper points corresponding

    # ====================JOINT PREDICTION====================
    upper_point = [
        0,
        0,
        0,
    ]  # the handle is fully extended. IMPORTANT: in the original mesh,
    # the handle is fully extended at resting position. So the upper point
    # is set to [0, 0, 0]
    factor = 0.5 # not fully all the down
    # this handle moves along z-axis
    lower_point = [
        0,
        0,
        -factor * bb_dim["height"],
    ]  # the handle is fully retracted, moving down by a factor of  of its height.
    ## IMPORTANT: we MUST set negative to simulate a downward movement
    pred_robot.make_prismatic_joint(
        "handle", "suitcase_body", lower_point, upper_point)

    return pred_robot


def partnet_47645(input_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - lid
    - box_body


    Object: a treasure chest
    Targetted affordance: "lid"
    """
    pred_robot = Robot(input_dir=input_dir, name="treasure_chest")
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
    # in the groundtruth video, the lid is attached to the body at the BACK and bottom of the lid
    # so pivot point needs to either be Back-Left-Bottom (BLB) or Back-Right-Bottom (BRB)
    lid_bb = pred_robot.get_bounding_boxes(["lid"], include_dim=False)["lid"]
    lid_vertices = compute_aabb_vertices(*lid_bb)
    pivot_point = lid_vertices[0]  # Back-Left-Bottom (BLB)

    pred_robot.make_revolute_joint(
        "lid",
        "box_body",
        global_axis=[
            0,
            # pivot-axis relationship:
            # since the pivot is on the back and we want to open upward, set
            # axis to negative
            -1,
            0,
        ],  # the lid of the treasure chest opens up and down so rotates along the left-right axis,
        # which is the y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,
        pivot_point=pivot_point,
    )

    return pred_robot


def partnet_3398(input_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - lid
    - bottle_body

    Object: a bottle of soap
    Targetted affordance: "lid"
    """
    pred_robot = Robot(input_dir=input_dir, name="soap_bottle")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["bottle_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_bottle_body", Parent("base"), Child("bottle_body"), type="fixed"
        ),
    )

    # lid goes on top of the bottle. No clearance or further adjustment needed
    pred_robot.add_link(links["lid"])
    pred_robot.place_relative_to(
        "lid", "bottle_body", placement="above", clearance=0.0)
    # ========================================================

    # ====================JOINT PREDICTION====================
    # rotate around the center of the lid / bottle body.
    # -- Groundtruth video analysis --
    # in the groundtruth video, the lid around the bottle's body so the lid is rotating
    # around its own center of mass
    # do not set pivot_point
    pred_robot.make_revolute_joint(
        "lid",
        "bottle_body",
        global_axis=[
            0,
            0,
            1,
        ],  # the bottle lid twists open and close, so it rotates around its vertical axis,
        # which is the z-axis
        lower_angle_deg=0,
        upper_angle_deg=360,
    )

    return pred_robot


def partnet_102658(input_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - lid
    - seat
    - toilet_body


    Object: a toilet
    Targetted affordance: "lid"
    """
    pred_robot = Robot(input_dir=input_dir, name="toilet")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["toilet_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_toilet_body", Parent("base"), Child("toilet_body"), type="fixed"
        ),
    )

    # seat goes on top of the toilet body. No clearance or further adjustment needed
    pred_robot.add_link(links["seat"])
    pred_robot.place_relative_to(
        "seat", "toilet_body", placement="above", clearance=0.0
    )
    # ========================================================

    pred_robot.add_link(links["lid"])
    pred_robot.place_relative_to(
        "lid", "toilet_body", placement="above", clearance=0.0)

    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # in the groundtruth video, the toilet lid opens up while its back parts still attached
    # to the body so the pivot point needs to be Back-Left-Bottom (BLB) or Back-Right-Bottom (BRB)
    lid_bb = pred_robot.get_bounding_boxes(["lid"], include_dim=False)["lid"]
    lid_vertices = compute_aabb_vertices(*lid_bb)
    pivot_point = lid_vertices[0]  # Back-Left-Bottom (BLB)
    pred_robot.make_revolute_joint(
        "lid",
        "toilet_body",
        global_axis=[
            0,
            # pivot-axis relationship:
            # since the pivot is on the back and we want to open upward, set
            # axis to negative
            -1,
            0,
        ],  # it rotates along the left-right axis,
        # which is the y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,  # NOTE: might need to tune this by visual feedback
        pivot_point=pivot_point,
    )

    return pred_robot


def partnet_103015(input_dir, links, joint_id="joint_2"):
    """
    No. masked_links: 5
    Robot Link Summary:
    - base
    - window_frame
    - window
    - window_2
    - window_3

    Object: a window
    Targetted affordance: "window_3"
    """
    pred_robot = Robot(input_dir=input_dir, name="window")
    pred_robot.add_link(links["base"])

    pred_robot.add_link(links["window_frame"])
    pred_robot.add_joint(
        Joint(
            "base_to_window_frame", Parent("base"), Child("window_frame"), type="fixed"
        ),
    )

    for window_link in ["window", "window_2", "window_3"]:
        pred_robot.add_link(links[window_link])

        # we actually want to place the window_link inside the window_frame, and not above.
        # so we need to shift the window_frame down by its height. We first place it above
        # then compute the height and shift it down by that amount using clearance
        # and remove the previous placement
        pred_robot.place_relative_to(
            window_link, "window_frame", placement="above", clearance=0.0
        )

        bbox = pred_robot.get_bounding_boxes([window_link], include_dim=True)[
            window_link
        ]
        bbox_dim = bbox[1]

        pred_robot.place_relative_to(
            window_link,
            "window_frame",
            placement="above",
            clearance=-bbox_dim["height"],
        )

    # ========================================================

    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # in the groundtruth video, the window opens up while its top parts still attached
    # to the frame so the pivot point needs
    # to be Back-Left-Top or Back-Right-Top
    target_window_link = "window_3"
    bbox = pred_robot.get_bounding_boxes([target_window_link], include_dim=False)[
        target_window_link
    ]
    vertices = compute_aabb_vertices(*bbox)
    pivot_point = vertices[
        4
    ]  # Back-Left-Top ## rotate around the point attached to the frame
    # so "back" and "top" are important.
    pred_robot.make_revolute_joint(
        target_window_link,
        "window_frame",
        global_axis=[
            0,
            # pivot-axis relationship:
            # since the pivot is on the top and we want to open outward, set
            # axis to negative
            -1,
            0,
        ],  # this window opens up and down so rotates along the left-right axis,
        # which is the y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,  # NOTE: the angle needs to be tuned by visual feedback as well
        pivot_point=pivot_point,
    )

    return pred_robot


def partnet_40417(input_dir, links, joint_id="joint_2"):
    """
    No. masked_links: 8
    Robot Link Summary:
    - base
    - door
    - door_2
    - drawer
    - drawer_2
    - door_3
    - door_4
    - furniture_body

    Object: a kitchen cabinet with 4 doors and 2 drawers
    Targetted affordance: drawer
    """
    pred_robot = Robot(input_dir=input_dir, name="kitchen_cabinet")
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

    pred_robot.add_link(links["drawer"])
    pred_robot.place_relative_to(
        "drawer", "furniture_body", placement="inside", clearance=0.0
    )

    pred_robot.add_link(links["drawer_2"])
    pred_robot.place_relative_to(
        "drawer_2", "furniture_body", placement="inside", clearance=0.0
    )

    # ====================JOINT PREDICTION====================
    target_drawer_link = "drawer"
    lower_point = [
        0,
        0,
        0,
    ]  # the drawer is fully retracted. At the same pos (hide in the box) ## TUNABLE
    bbox = pred_robot.get_bounding_boxes([target_drawer_link], include_dim=True)[
        target_drawer_link
    ]
    bbox_dim = bbox[1]
    # x, y, z corresponds to length, width, height
    # this drawer moves along x-axis
    upper_point = [
        bbox_dim["length"],
        0,
        0,
    ]  # the drawer is fully extended. At the same pos as the drawer_width ## TUNABLE
    # it slides forward and backward so x-axis

    pred_robot.make_prismatic_joint(
        target_drawer_link, "furniture_body", lower_point, upper_point
    )
    return pred_robot


def partnet_41003(input_dir, links, joint_id="joint_1"):
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
    pred_robot = Robot(input_dir=input_dir, name="cabinet")
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
    # in the groundtruth video, the top-right door opens up while its right part of the door is still
    # attached to the body so the pivot point needs to be Back-Right-Bottom (BRB)
    # or Back-Right-Top (BRT)

    target_door_link = "door_2"
    door_bb = pred_robot.get_bounding_boxes([target_door_link], include_dim=False)[
        target_door_link
    ]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[1]  # Back-Right-Bottom (BRB)

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


def partnet_41003(input_dir, links, joint_id="joint_3"):
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
    Targetted affordance: "door_4"
    """
    pred_robot = Robot(input_dir=input_dir, name="cabinet")
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
    # in the groundtruth video, the top-left door opens up while its left part of the door is still
    # attached to the body so the pivot point needs to be Back-Left-Bottom (BLB)
    # or Back-Left-Top (BLT)
    target_door_link = "door_4"
    door_bb = pred_robot.get_bounding_boxes([target_door_link], include_dim=False)[
        target_door_link
    ]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[0]  # Back-Left-Bottom (BLB)

    pred_robot.make_revolute_joint(
        target_door_link,
        "furniture_body",
        global_axis=[
            0,
            0,
            # pivot-axis relationship:
            # since the pivot is on the left and we want to open outward, set
            # axis to negative
            -1,
        ],  # The cabinet door swings open and closed around the vertical axis,
        # which is the z-axis. Negative z-axis because the pivot point is on the left
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot


def partnet_46768(input_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - door
    - door_2
    - furniture_body

    Object: a cabinet with two sliding doors
    Targetted affordance: "door"
    """
    pred_robot = Robot(input_dir=input_dir, name="cabinet")
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

    # door 1
    door_link = "door"
    pred_robot.add_link(links[door_link])
    pred_robot.place_relative_to(
        door_link, "furniture_body", placement="inside", clearance=0.0
    )

    # door 2
    door_link = "door_2"
    pred_robot.add_link(links[door_link])
    pred_robot.place_relative_to(
        door_link, "furniture_body", placement="inside", clearance=0.0
    )

    # ====================JOINT PREDICTION====================
    target_door_link = "door"
    lower_point = [0, 0, 0]
    bbox = pred_robot.get_bounding_boxes([target_door_link], include_dim=True)[
        target_door_link
    ]
    bbox_dim = bbox[1]
    upper_point = [
        0,
        -bbox_dim["width"],
        0,
    ]  # this door slides left and right along the y-axis
    pred_robot.make_prismatic_joint(
        target_door_link, "furniture_body", lower_point, upper_point
    )

    return pred_robot


def partnet_44817(input_dir, links, joint_id="joint_2"):
    """
    No. masked_links: 6
    Robot Link Summary:
    - base
    - drawer
    - drawer_2
    - drawer_3
    - drawer_4
    - furniture_body

    Object: a chest of drawers with 4 drawers
    Targetted afffordance: "drawer_3"
    """
    pred_robot = Robot(input_dir=input_dir, name="chest_of_drawers")
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

    pred_robot.add_link(links["drawer"])
    pred_robot.place_relative_to(
        "drawer", "furniture_body", placement="inside", clearance=0.0
    )
    # ========================================================
    pred_robot.add_link(links["drawer_2"])
    pred_robot.place_relative_to(
        "drawer_2", "furniture_body", placement="inside", clearance=0.0
    )
    # ========================================================
    pred_robot.add_link(links["drawer_3"])
    pred_robot.place_relative_to(
        "drawer_3", "furniture_body", placement="inside", clearance=0.0
    )
    # ========================================================
    pred_robot.add_link(links["drawer_4"])
    pred_robot.place_relative_to(
        "drawer_4", "furniture_body", placement="inside", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    target_drawer_link = "drawer_3"
    lower_point = [
        0,
        0,
        0,
    ]  # the drawer is fully retracted. At the same pos (hide in the box) ## TUNABLE
    bbox = pred_robot.get_bounding_boxes(
        [target_drawer_link], include_dim=True
    )  # get all bounding boxes for all links

    bbox_dim = bbox[target_drawer_link][1]
    upper_point = [
        bbox_dim["length"],
        0,
        0,
    ]  # the drawer is fully extended. At the same pos as the drawer_width ## TUNABLE
    # the drawer slides forward and backward so x-axis
    pred_robot.make_prismatic_joint(
        target_drawer_link, "furniture_body", lower_point, upper_point
    )
    return pred_robot


def partnet_103619(input_dir, links, joint_id="joint_3"):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - handle
    - dispenser_body
    - lid

    Object: a dispenser bottle
    Targetted affordance: "lid" can be lifted up
    """
    pred_robot = Robot(input_dir=input_dir, name="dispenser_bottle")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["dispenser_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_dispenser_body",
            Parent("base"),
            Child("dispenser_body"),
            type="fixed",
        ),
    )

    # lid goes on top of the dispenser_body. No clearance or further adjustment needed
    pred_robot.add_link(links["lid"])
    pred_robot.place_relative_to(
        "lid", "dispenser_body", placement="above", clearance=0.0
    )
    # handle goes on top of the dispenser_body. No clearance or further adjustment needed
    pred_robot.add_link(links["handle"])
    pred_robot.place_relative_to(
        "handle", "dispenser_body", placement="above", clearance=0.0
    )
    # ====================JOINT PREDICTION====================

    # this lid moves along z-axis
    lower_point = [0, 0, 0]
    # the lid move a very small amount up ## TUNABLE
    upper_point = [0, 0, 0.025]
    pred_robot.make_prismatic_joint(
        "lid", "dispenser_body", lower_point, upper_point)

    return pred_robot


def partnet_103619(input_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - handle
    - dispenser_body
    - lid

    Object: a dispenser bottle
    Targetted affordance: "handle" can be twisted to dispense liquid by rotating
    """
    pred_robot = Robot(input_dir=input_dir, name="dispenser_bottle")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["dispenser_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_dispenser_body",
            Parent("base"),
            Child("dispenser_body"),
            type="fixed",
        ),
    )

    # lid goes on top of the dispenser_body. No clearance or further adjustment needed
    pred_robot.add_link(links["lid"])
    pred_robot.place_relative_to(
        "lid", "dispenser_body", placement="above", clearance=0.0
    )
    # handle goes on top of the dispenser_body. No clearance or further adjustment needed
    pred_robot.add_link(links["handle"])
    pred_robot.place_relative_to(
        "handle", "dispenser_body", placement="above", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # in the groundtruth video, the handle rotates up and down while its TOP part still
    # attached to the lid so the pivot point needs to be Back-Right-Top (BRT) or Back-Left-Top (BLT)

    lid_bb = pred_robot.get_bounding_boxes(
        ["handle"], include_dim=False)["handle"]
    lid_vertices = compute_aabb_vertices(*lid_bb)
    pivot_point = lid_vertices[5]  # Back-Right-Top (BRT)

    pred_robot.make_revolute_joint(
        "handle",
        "dispenser_body",
        global_axis=[
            0,
            # pivot-axis relationship:
            # the handle is squeezed in (i.e. open inward) and the pivot is on the top.
            # so set the axis to positive
            1,
            0,
        ],  # the handle twists open and close, so rotates along the left-right axis,
        lower_angle_deg=0,
        upper_angle_deg=45,  # slightly open inward
        pivot_point=pivot_point,
    )

    return pred_robot


def partnet_102318(input_dir, links):
    """
    No. masked_links: 17
    Robot Link Summary:
    - base
    - door
    - button
    - button_2
    - button_3
    - button_4
    - button_5
    - button_6
    - button_7
    - button_8
    - button_9
    - button_10
    - button_11
    - button_12
    - knob
    - safe_body


    Object: a safe with a door, a knob, and buttons
    Affordance: buttons can be pushed
    """
    pred_robot = Robot(input_dir=input_dir, name="safe")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["safe_body"])
    pred_robot.add_joint(
        Joint("base_to_safe_body", Parent("base"),
              Child("safe_body"), type="fixed"),
    )

    pred_robot.add_link(links["door"])
    pred_robot.place_relative_to(
        "door", "safe_body", placement="front", clearance=0.0)
    pred_robot.add_link(links["knob"])
    pred_robot.place_relative_to(
        "knob", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button"])
    pred_robot.place_relative_to(
        "button", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_2"])
    pred_robot.place_relative_to(
        "button_2", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_3"])
    pred_robot.place_relative_to(
        "button_3", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_4"])
    pred_robot.place_relative_to(
        "button_4", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_5"])
    pred_robot.place_relative_to(
        "button_5", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_6"])
    pred_robot.place_relative_to(
        "button_6", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_7"])
    pred_robot.place_relative_to(
        "button_7", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_8"])
    pred_robot.place_relative_to(
        "button_8", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_9"])
    pred_robot.place_relative_to(
        "button_9", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_10"])
    pred_robot.place_relative_to(
        "button_10", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_11"])
    pred_robot.place_relative_to(
        "button_11", "door", placement="front", clearance=0.0)
    pred_robot.add_link(links["button_12"])
    pred_robot.place_relative_to(
        "button_12", "door", placement="front", clearance=0.0)

    # # ====================JOINT PREDICTION====================
    pushing_direction = pred_robot.compute_push_direction("button")  # compute
    # the normal vector to the button mesh, which is the pushing direction
    num_buttons = 12
    for link_idx in range(num_buttons):
        link = f"button" if link_idx == 0 else f"button_{link_idx+1}"
        lower_point = [0, 0, 0]
        pred_robot.make_prismatic_joint(
            link, "door", lower_point, pushing_direction)
    return pred_robot


def partnet_36280(input_dir, links, joint_id="joint_17"):
    """
    No. masked_links: 18
    Robot Link Summary:
    - base
    - caster
    - wheel
    - wheel_2
    - caster_2
    - wheel_3
    - wheel_4
    - caster_3
    - wheel_5
    - wheel_6
    - caster_4
    - wheel_7
    - wheel_8
    - caster_5
    - wheel_9
    - wheel_10
    - seat
    - chair_leg


    Object: an office chair
    Targetted affordance: "seat" can be pushed down
    """
    pred_robot = Robot(input_dir=input_dir, name="office_chair")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["chair_leg"])
    pred_robot.add_joint(
        Joint("base_to_chair_leg", Parent("base"),
              Child("chair_leg"), type="fixed"),
    )

    pred_robot.add_link(links["seat"])
    pred_robot.place_relative_to(
        "seat", "chair_leg", placement="above", clearance=0.0)
    pred_robot.add_link(links["caster"])
    pred_robot.place_relative_to(
        "caster", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel"])
    pred_robot.place_relative_to(
        "wheel", "caster", placement="below", clearance=0.0)
    pred_robot.add_link(links["wheel_2"])
    pred_robot.place_relative_to(
        "wheel_2", "caster", placement="below", clearance=0.0)
    pred_robot.add_link(links["caster_2"])
    pred_robot.place_relative_to(
        "caster_2", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_3"])
    pred_robot.place_relative_to(
        "wheel_3", "caster_2", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_4"])
    pred_robot.place_relative_to(
        "wheel_4", "caster_2", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["caster_3"])
    pred_robot.place_relative_to(
        "caster_3", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_5"])
    pred_robot.place_relative_to(
        "wheel_5", "caster_3", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_6"])
    pred_robot.place_relative_to(
        "wheel_6", "caster_3", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["caster_4"])
    pred_robot.place_relative_to(
        "caster_4", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_7"])
    pred_robot.place_relative_to(
        "wheel_7", "caster_4", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_8"])
    pred_robot.place_relative_to(
        "wheel_8", "caster_4", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["caster_5"])
    pred_robot.place_relative_to(
        "caster_5", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_9"])
    pred_robot.place_relative_to(
        "wheel_9", "caster_5", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_10"])
    pred_robot.place_relative_to(
        "wheel_10", "caster_5", placement="below", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # The seat is moving down.
    # The axis of motion is along the z-axis.
    pred_robot.make_prismatic_joint(
        "seat",
        "chair_leg",
        global_lower_point=[0, 0, 0],
        global_upper_point=[0, 0, -0.10],  # chair leg moves downward as
        # per groundtruth video
    )
    return pred_robot


def partnet_36280(input_dir, links, joint_id="joint_15"):
    """
    No. masked_links: 18
    Robot Link Summary:
    - base
    - caster
    - wheel
    - wheel_2
    - caster_2
    - wheel_3
    - wheel_4
    - caster_3
    - wheel_5
    - wheel_6
    - caster_4
    - wheel_7
    - wheel_8
    - caster_5
    - wheel_9
    - wheel_10
    - seat
    - chair_leg


    Object: an office chair
    Targetted affordance: "seat" can be rotated
    """
    pred_robot = Robot(input_dir=input_dir, name="office_chair")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["chair_leg"])
    pred_robot.add_joint(
        Joint("base_to_chair_leg", Parent("base"),
              Child("chair_leg"), type="fixed"),
    )

    pred_robot.add_link(links["seat"])
    pred_robot.place_relative_to(
        "seat", "chair_leg", placement="above", clearance=0.0)
    pred_robot.add_link(links["caster"])
    pred_robot.place_relative_to(
        "caster", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel"])
    pred_robot.place_relative_to(
        "wheel", "caster", placement="below", clearance=0.0)
    pred_robot.add_link(links["wheel_2"])
    pred_robot.place_relative_to(
        "wheel_2", "caster", placement="below", clearance=0.0)
    pred_robot.add_link(links["caster_2"])
    pred_robot.place_relative_to(
        "caster_2", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_3"])
    pred_robot.place_relative_to(
        "wheel_3", "caster_2", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_4"])
    pred_robot.place_relative_to(
        "wheel_4", "caster_2", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["caster_3"])
    pred_robot.place_relative_to(
        "caster_3", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_5"])
    pred_robot.place_relative_to(
        "wheel_5", "caster_3", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_6"])
    pred_robot.place_relative_to(
        "wheel_6", "caster_3", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["caster_4"])
    pred_robot.place_relative_to(
        "caster_4", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_7"])
    pred_robot.place_relative_to(
        "wheel_7", "caster_4", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_8"])
    pred_robot.place_relative_to(
        "wheel_8", "caster_4", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["caster_5"])
    pred_robot.place_relative_to(
        "caster_5", "chair_leg", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_9"])
    pred_robot.place_relative_to(
        "wheel_9", "caster_5", placement="below", clearance=0.0
    )
    pred_robot.add_link(links["wheel_10"])
    pred_robot.place_relative_to(
        "wheel_10", "caster_5", placement="below", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # the seat can rotate along its our center of mass (no pivot point)
    # along the z-axis (vertical axis)
    pred_robot.make_revolute_joint(

        "seat",
        "chair_leg",
        global_axis=[0, 0, 1],
        lower_angle_deg=0,
        upper_angle_deg=360,
    )
    return pred_robot


def partnet_3615(input_dir, links, joint_id="joint_2"):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - lid
    - bottle_body

    Object: a bottle of liquid
    Targetted affordance: "lid" can be pushed down
    """
    pred_robot = Robot(input_dir=input_dir, name="bottle")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["bottle_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_bottle_body", Parent("base"), Child("bottle_body"), type="fixed"
        ),
    )

    # lid goes on top of the bottle. No clearance or further adjustment needed
    pred_robot.add_link(links["lid"])
    pred_robot.place_relative_to(
        "lid", "bottle_body", placement="above", clearance=0.0)
    # ====================JOINT PREDICTION====================
    # the lid moves along z-axis (vertical)
    lower_point = [0, 0, 0]
    upper_point = [0, 0, -0.1]
    pred_robot.make_prismatic_joint(
        "lid", "bottle_body", lower_point, upper_point)

    return pred_robot


def partnet_12565(input_dir, links, joint_id="joint_1"):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - dishwasher_body
    - door


    Object: a dishwasher
    Targetted affordance: "door"
    """
    pred_robot = Robot(input_dir=input_dir, name="dishwasher")
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
    # in the groundtruth video, the door opens up while its BOTTOM part still
    # attached to the body so the pivot point needs to be Back-Left-Bottom (BLB)
    # or Back-Right-Bottom (BRB)
    door_bb = pred_robot.get_bounding_boxes(
        ["door"], include_dim=False)["door"]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[0]  # Back-Left-Bottom (BLB)

    pred_robot.make_revolute_joint(
        "door",
        "dishwasher_body",
        global_axis=[
            0,
            # pivot-axis relationship:
            # since the pivot is at the bottom and we want to open outward, set
            # axis to positive
            1,
            0,
        ],  # The door opens by opening up down: i.e., rotating along the left-right axis, which is y-axis
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot


def partnet_10900(input_dir, links, joint_id="joint_0"):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - door
    - door_2
    - refrigerator_body

    Object: a refrigerator with two doors
    Targetted affordance: "door"
    """
    pred_robot = Robot(input_dir=input_dir, name="refrigerator")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["refrigerator_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_refrigerator_body",
            Parent("base"),
            Child("refrigerator_body"),
            type="fixed",
        ),
    )

    pred_robot.add_link(links["door"])
    # refrigerator doors go in front of the refrigerator body.
    pred_robot.place_relative_to(
        "door", "refrigerator_body", placement="front", clearance=0.0
    )

    pred_robot.add_link(links["door_2"])
    pred_robot.place_relative_to(
        "door_2", "refrigerator_body", placement="front", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # in the groundtruth video, the left door opens up while its right part still
    # attached to the body so the pivot point needs to be Back-Right-Bottom (BRB)
    # or Back-Right-Top (BRT)
    target_door_link = "door"
    door_bb = pred_robot.get_bounding_boxes([target_door_link], include_dim=False)[
        target_door_link
    ]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[1]  # Back-Right-Bottom (BRB)

    pred_robot.make_revolute_joint(
        target_door_link,
        "refrigerator_body",
        global_axis=[
            0,
            0,
            # pivot-axis relationship:
            # since the pivot is on the right and we want to open outward, set
            # axis to positive
            1,  # opening up
        ],  # The refrigerator door swings open and closed around the vertical axis,
        # which is the z-axis.
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot


def partnet_7366(input_dir, links, joint_id="joint_1"):
    """
    No. masked_links: 5
    Robot Link Summary:
    - base
    - tray
    - door
    - button
    - microwave_body

    Object: a microwave
    Targetted affordance: "door"
    """
    pred_robot = Robot(input_dir=input_dir, name="microwave")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['microwave_body'])
    pred_robot.add_joint(Joint("base_to_microwave_body",
                         Parent("base"),
                         Child("microwave_body"),
                         type="fixed"),
                         )

    # =============================================
    # partnet mobility peculiarity: need to always call this function
    # after the first joint to orientate the robot correctly
    pred_robot.align_robot_orientation()
    # =============================================

    pred_robot.add_link(links['tray'])
    pred_robot.place_relative_to('tray', 'microwave_body',
                                 placement="inside",
                                 clearance=0.0)

    pred_robot.add_link(links['door'])
    pred_robot.place_relative_to('door', 'microwave_body',
                                 placement="front",
                                 clearance=0.0)

    pred_robot.add_link(links['button'])
    pred_robot.place_relative_to('button', 'microwave_body',
                                 placement="front",
                                 clearance=0.0)
    # ====================JOINT PREDICTION====================
    # -- Groundtruth video analysis --
    # in the groundtruth video, the door opens up while its left part still
    # attached to the body so the pivot point needs to be Back-Left-Bottom (BLB)
    # or Back-Left-Top (BLT)
    door_bb = pred_robot.get_bounding_boxes(
        ['door'], include_dim=False)['door']
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[0]  # Back-Left-Bottom (BLB)

    pred_robot.make_revolute_joint(
        "door",
        "microwave_body",
        global_axis=[
            0,
            0,
            # pivot-axis relationship:
            # since the pivot is on the left and we want to open outward, set
            # axis to negative
            -1,
        ],  # The microwave door swings open and closed around the vertical axis,
        # which is the z-axis. Negative z-axis because the pivot point is on the left
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )
    return pred_robot


def partnet_10383(input_dir, links, joint_id="joint_1") -> Robot:
    pred_robot = Robot(input_dir=input_dir)
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["laptop_base"])
    pred_robot.add_joint(
        Joint(
            "base_to_laptop_base",
            Parent("base"),
            Child("laptop_base"),
            type="fixed",
        )
    )
    pred_robot.add_link(links["screen"])
    pred_robot.place_relative_to(
        "screen", "laptop_base", placement="above", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    # The screen is hinged at the bottom and opens up and down.
    # The axis of motion is along the y-axis.
    screen_bb = pred_robot.get_bounding_boxes(
        ["screen"], include_dim=False)["screen"]
    screen_vertices = compute_aabb_vertices(*screen_bb)
    pivot_point = screen_vertices[2]  # Front-Left-Bottom (FLB)
    # laptop is special case where we attach the screen at Front instead of Back
    # This is because the default mesh of the laptop is standing up slightly tilted backward
    # so the front part is attached to the base
    pred_robot.make_revolute_joint(
        "screen",
        "laptop_base",
        # the laptop screen rotate along y-axis (left-right axis)
        global_axis=[0, 1, 0],
        lower_angle_deg=0,
        upper_angle_deg=135,  # open up
        pivot_point=pivot_point,
    )
    return pred_robot
