from articulate_anything.api.odio_urdf import *


def partnet_100248(input_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - handle
    - suitcase_body

    Object: a suitcase with a handle
    Affordance: sliding the handle up and down
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

    pred_robot.add_link(links["handle"])
    pred_robot.place_relative_to(
        "handle", "suitcase_body", placement="above", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    bb = pred_robot.get_bounding_boxes(["handle"], include_dim=True)["handle"]
    bb_dim = bb[1]  # length of the handle. bb: [x, y, z]
    # the handle starts at the extended position
    # per the link placement code, the handle is correctly `above` the suitcase_body
    # fully extended.

    # Moving the suitcase down involves moving the handle down by its length along the z-axis
    lower_point = [
        0,
        0,
        -bb_dim["height"],
    ]  # the handle is fully retracted. At the same pos (hide in the box) ## TUNABLE
    # this handle moves along z-axis
    upper_point = [
        0,
        0,
        0,
    ]  # the handle is fully extended. At the same pos as the handle_height ## TUNABLE
    pred_robot.make_prismatic_joint(
        "handle", "suitcase_body", lower_point, upper_point)

    return pred_robot


def partnet_47645(input_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - lid
    - box_body


    Object: a treasure chest
    Affordance: lid can be opened and closed
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
    # we're rotating at an edge of the lid instead of the center
    lid_bb = pred_robot.get_bounding_boxes(["lid"], include_dim=False)["lid"]
    lid_vertices = compute_aabb_vertices(*lid_bb)
    pivot_point = lid_vertices[0]  # Bottom-Back-Left

    pred_robot.make_revolute_joint(
        "lid",
        "box_body",
        global_axis=[
            0,
            1,
            0,
        ],  # the lid of the treasure chest opens up and down so rotates along the left-right axis,
        # which is the y-axis
        lower_angle_deg=-90,
        upper_angle_deg=0,  # opening upward
        pivot_point=pivot_point,
    )

    return pred_robot


def partnet_3398(input_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - lid
    - bottle_body

    Object: a bottle of soap
    Affordance: bottle mouth rotating around
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


def partnet_102630(input_dir, links):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - lid
    - seat
    - toilet_body


    Object: a toilet
    Affordance: lid and seat can be rotated up and down
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
    pred_robot.make_revolute_joint(
        "lid",
        "toilet_body",
        global_axis=[
            0,
            1,
            0,
        ],  # the toilet lid opens up and down, so it rotates along the left-right axis,
        # which is the y-axis
        # the current lid is
        lower_angle_deg=-20,
        upper_angle_deg=70,
    )

    pred_robot.make_revolute_joint(
        "seat",
        "toilet_body",
        global_axis=[
            0,
            1,
            0,
        ],  # the toilet seat opens up and down, so it rotates along the left-right axis,
        # which is the y-axis
        lower_angle_deg=-90,
        upper_angle_deg=0,  # opening upward
    )

    return pred_robot


def partnet_103015(input_dir, links):
    """
    No. masked_links: 5
    Robot Link Summary:
    - base
    - window_frame
    - window
    - window_2
    - window_3

    Object: a window
    Affordance: windows can be opened and closed
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
    # ## rotate around an edge of the window instead of the center
    # ## so we need to specify the pivot_point
    for window_link in ["window", "window_2", "window_3"]:
        bbox = pred_robot.get_bounding_boxes([window_link], include_dim=False)[
            window_link
        ]
        vertices = compute_aabb_vertices(*bbox)
        pivot_point = vertices[
            6
        ]  # Top-Front-Left ## NOTE: need to tune this by visual feedback
        pred_robot.make_revolute_joint(
            window_link,
            "window_frame",
            global_axis=[
                0,
                1,
                0,
            ],  # this window opens up and down so rotates along the left-right axis,
            # which is the y-axis
            lower_angle_deg=-90,
            upper_angle_deg=90,  # NOTE: the angle needs to be tuned by visual feedback as well
            pivot_point=pivot_point,
        )

    return pred_robot


def partnet_40417(input_dir, links):
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
    Affordance: drawers can be opened and closed via prismatic joints
    Doors can be rotated open and closed via revolute joints
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

    # ========================================================

    # ====================JOINT PREDICTION====================

    bbox = pred_robot.get_bounding_boxes(include_dim=True)  # get for all links
    for drawer_link in ["drawer", "drawer_2"]:
        lower_point = [
            0,
            0,
            0,
        ]  # the drawer is fully retracted. At the same pos (hide in the box) ## TUNABLE
        bbox_dim = bbox[drawer_link][1]
        # x, y, z corresponds to length, width, height
        # this drawer moves along x-axis
        upper_point = [
            bbox_dim["length"],
            0,
            0,
        ]  # the drawer is fully extended. At the same pos as the drawer_width ## TUNABLE
        # sliding front and back so along x-axis

        pred_robot.make_prismatic_joint(
            drawer_link, "furniture_body", lower_point, upper_point
        )

    for door in ["door", "door_2", "door_3", "door_4"]:
        door_bb = pred_robot.get_bounding_boxes(
            [door], include_dim=False)[door]
        door_vertices = compute_aabb_vertices(*door_bb)
        pivot_point = door_vertices[3]
        pred_robot.make_revolute_joint(
            door,
            "furniture_body",
            global_axis=[
                0,
                0,
                1,
            ],  # The cabinet door swings open and closed around the vertical axis,
            # which is the z-axis.
            lower_angle_deg=0,
            upper_angle_deg=90,
            pivot_point=pivot_point,
        )

    return pred_robot


def partnet_41003(input_dir, links):
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
    Affordance: the doors can be rotated open and closed
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
    # we're rotating at an edge of the door instead of the center
    # so we need to specify the pivot_point
    for door_idx in range(4):
        door_link = "door" if door_idx == 0 else f"door_{door_idx+1}"
        door_bb = pred_robot.get_bounding_boxes([door_link], include_dim=False)[
            door_link
        ]
        door_vertices = compute_aabb_vertices(*door_bb)
        pivot_point = door_vertices[3]  # Bottom-Front-Right

        pred_robot.make_revolute_joint(
            door_link,
            "furniture_body",
            global_axis=[
                0,
                0,
                1,
            ],  # The cabinet door swings open and closed around the vertical axis,
            # which is the z-axis.
            lower_angle_deg=0,
            upper_angle_deg=90,
            pivot_point=pivot_point,
        )
    return pred_robot


def partnet_46768(input_dir, links):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - door
    - door_2
    - furniture_body

    Object: a cabinet with two sliding doors
    Affordance: the doors can be slid open and closed
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
    # The right door slides from right to left
    lower_point = [0, 0, 0]
    bbox = pred_robot.get_bounding_boxes(["door"], include_dim=True)["door"]
    bbox_dim = bbox[1]
    upper_point = [0, -bbox_dim["width"], 0]
    pred_robot.make_prismatic_joint(
        "door", "furniture_body", lower_point, upper_point)

    # The left door slides from left to right
    lower_point = [0, 0, 0]
    bbox = pred_robot.get_bounding_boxes(
        ["door_2"], include_dim=True)["door_2"]
    bbox_dim = bbox[1]
    upper_point = [0, bbox_dim["width"], 0]
    pred_robot.make_prismatic_joint(
        "door_2", "furniture_body", lower_point, upper_point
    )
    return pred_robot


def partnet_44817(input_dir, links):
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
    # drawer 1
    lower_point = [
        0,
        0,
        0,
    ]  # the drawer is fully retracted. At the same pos (hide in the box) ## TUNABLE
    bbox = pred_robot.get_bounding_boxes(
        include_dim=True
    )  # get all bounding boxes for all links
    num_drawers = 4

    for drawer in range(num_drawers):
        drawer_name = "drawer" if drawer == 0 else f"drawer_{drawer+1}"
        bbox_dim = bbox[drawer_name][1]
        upper_point = [
            bbox_dim["length"],
            0,
            0,
        ]  # the drawer is fully extended. At the same pos as the drawer_width ## TUNABLE
        pred_robot.make_prismatic_joint(
            drawer_name, "furniture_body", lower_point, upper_point
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
    Affordance: the door can be opened and closed
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
    # # we're rotating at an edge of the door instead of the center
    # # so we need to specify the pivot_point
    door_bb = pred_robot.get_bounding_boxes(
        ["door"], include_dim=False)["door"]
    door_vertices = compute_aabb_vertices(*door_bb)
    pivot_point = door_vertices[1]  # Front-Right-Bottom

    pred_robot.make_revolute_joint(
        "door",
        "safe_body",
        global_axis=[0, 0, 1],  # The safe door swings open and closed
        # to the front and back like a door so along the fixed z-axis
        lower_angle_deg=0,
        upper_angle_deg=90,  # open outward
        pivot_point=pivot_point,
    )

    pushing_direction = pred_robot.compute_push_direction("button")  # compute
    # the normal vector to the button mesh, which is the pushing direction
    num_buttons = 12
    for link_idx in range(num_buttons):
        link = f"button" if link_idx == 0 else f"button_{link_idx+1}"
        lower_point = [0, 0, 0]
        pred_robot.make_prismatic_joint(
            link, "door", lower_point, pushing_direction)
    return pred_robot


def partnet_103048(input_dir, links):
    """
    No. masked_links: 6
    Robot Link Summary:
    - base
    - container
    - lid
    - rotor
    - button
    - coffee_machine_body

    Object: a blender with a lid, a container, a rotor, a button and a body
    """
    pred_robot = Robot(input_dir=input_dir, name="blender")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["coffee_machine_body"])
    pred_robot.add_joint(
        Joint(
            "base_to_coffee_machine_body",
            Parent("base"),
            Child("coffee_machine_body"),
            type="fixed",
        ),
    )

    pred_robot.add_link(links["container"])
    pred_robot.place_relative_to(
        "container", "coffee_machine_body", placement="inside", clearance=0.0
    )

    pred_robot.add_link(links["lid"])
    pred_robot.place_relative_to(
        "lid", "container", placement="above", clearance=0.0)
    pred_robot.add_link(links["rotor"])
    pred_robot.place_relative_to(
        "rotor", "container", placement="inside", clearance=0.0
    )
    pred_robot.add_link(links["button"])
    pred_robot.place_relative_to(
        "button", "coffee_machine_body", placement="front", clearance=0.0
    )
    # ====================JOINT PREDICTION====================
    pred_robot.make_revolute_joint(
        "lid",
        "container",
        global_axis=[
            0,
            1,
            0,
        ],  # the lid of the blender opens up and down so rotates along the left-right axis,
        # which is the y-axis
        lower_angle_deg=-90,
        upper_angle_deg=0,  # opening upward
    )

    num_buttons = 1
    for link_idx in range(num_buttons):
        link = f"button" if link_idx == 0 else f"button_{link_idx+1}"
        pushing_direction = pred_robot.compute_push_direction(link)  # compute
        # the normal vector to the button mesh, which is the pushing direction
        lower_point = [0, 0, 0]
        pred_robot.make_prismatic_joint(
            link, "coffee_machine_body", lower_point, pushing_direction
        )
    return pred_robot


def partnet_13120(input_dir, links):
    """
    No. masked_links: 109
    Robot Link Summary:
    - base
    - key
    - key_2
    - key_3
    - key_4
    - key_5
    - key_6
    - key_7
    - key_8
    - key_9
    - key_10
    - key_11
    - key_12
    - key_13
    - key_14
    - key_15
    - key_16
    - key_17
    - key_18
    - key_19
    - key_20
    - key_21
    - key_22
    - key_23
    - key_24
    - key_25
    - key_26
    - key_27
    - key_28
    - key_29
    - key_30
    - key_31
    - key_32
    - key_33
    - key_34
    - key_35
    - key_36
    - key_37
    - key_38
    - key_39
    - key_40
    - key_41
    - key_42
    - key_43
    - key_44
    - key_45
    - key_46
    - key_47
    - key_48
    - key_49
    - key_50
    - key_51
    - key_52
    - key_53
    - key_54
    - key_55
    - key_56
    - key_57
    - key_58
    - key_59
    - key_60
    - key_61
    - key_62
    - key_63
    - key_64
    - key_65
    - key_66
    - key_67
    - key_68
    - key_69
    - key_70
    - key_71
    - key_72
    - key_73
    - key_74
    - key_75
    - key_76
    - key_77
    - key_78
    - key_79
    - key_80
    - key_81
    - key_82
    - key_83
    - key_84
    - key_85
    - key_86
    - key_87
    - key_88
    - key_89
    - key_90
    - key_91
    - key_92
    - key_93
    - key_94
    - key_95
    - key_96
    - key_97
    - key_98
    - key_99
    - key_100
    - key_101
    - key_102
    - key_103
    - key_104
    - key_105
    - key_106
    - key_107
    - key_108
    - keyboard_base


    Object: a keyboard with keys
    """
    pred_robot = Robot(input_dir=input_dir, name="keyboard")
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links["keyboard_base"])
    pred_robot.add_joint(
        Joint(
            "base_to_keyboard_base",
            Parent("base"),
            Child("keyboard_base"),
            type="fixed",
        ),
    )

    for key_link in [
        "key",
        "key_2",
        "key_3",
        "key_4",
        "key_5",
        "key_6",
        "key_7",
        "key_8",
        "key_9",
        "key_10",
        "key_11",
        "key_12",
        "key_13",
        "key_14",
        "key_15",
        "key_16",
        "key_17",
        "key_18",
        "key_19",
        "key_20",
        "key_21",
        "key_22",
        "key_23",
        "key_24",
        "key_25",
        "key_26",
        "key_27",
        "key_28",
        "key_29",
        "key_30",
        "key_31",
        "key_32",
        "key_33",
        "key_34",
        "key_35",
        "key_36",
        "key_37",
        "key_38",
        "key_39",
        "key_40",
        "key_41",
        "key_42",
        "key_43",
        "key_44",
        "key_45",
        "key_46",
        "key_47",
        "key_48",
        "key_49",
        "key_50",
        "key_51",
        "key_52",
        "key_53",
        "key_54",
        "key_55",
        "key_56",
        "key_57",
        "key_58",
        "key_59",
        "key_60",
        "key_61",
        "key_62",
        "key_63",
        "key_64",
        "key_65",
        "key_66",
        "key_67",
        "key_68",
        "key_69",
        "key_70",
        "key_71",
        "key_72",
        "key_73",
        "key_74",
        "key_75",
        "key_76",
        "key_77",
        "key_78",
        "key_79",
        "key_80",
        "key_81",
        "key_82",
        "key_83",
        "key_84",
        "key_85",
        "key_86",
        "key_87",
        "key_88",
        "key_89",
        "key_90",
        "key_91",
        "key_92",
        "key_93",
        "key_94",
        "key_95",
        "key_96",
        "key_97",
        "key_98",
        "key_99",
        "key_100",
        "key_101",
        "key_102",
        "key_103",
        "key_104",
        "key_105",
        "key_106",
        "key_107",
        "key_108",
    ]:
        pred_robot.add_link(links[key_link])
        pred_robot.place_relative_to(
            key_link, "keyboard_base", placement="inside", clearance=0.0
        )

    # ====================JOINT PREDICTION====================
    # All keys are fixed on the keyboard.
    for key_link in [
        "key",
        "key_2",
        "key_3",
        "key_4",
        "key_5",
        "key_6",
        "key_7",
        "key_8",
        "key_9",
        "key_10",
        "key_11",
        "key_12",
        "key_13",
        "key_14",
        "key_15",
        "key_16",
        "key_17",
        "key_18",
        "key_19",
        "key_20",
        "key_21",
        "key_22",
        "key_23",
        "key_24",
        "key_25",
        "key_26",
        "key_27",
        "key_28",
        "key_29",
        "key_30",
        "key_31",
        "key_32",
        "key_33",
        "key_34",
        "key_35",
        "key_36",
        "key_37",
        "key_38",
        "key_39",
        "key_40",
        "key_41",
        "key_42",
        "key_43",
        "key_44",
        "key_45",
        "key_46",
        "key_47",
        "key_48",
        "key_49",
        "key_50",
        "key_51",
        "key_52",
        "key_53",
        "key_54",
        "key_55",
        "key_56",
        "key_57",
        "key_58",
        "key_59",
        "key_60",
        "key_61",
        "key_62",
        "key_63",
        "key_64",
        "key_65",
        "key_66",
        "key_67",
        "key_68",
        "key_69",
        "key_70",
        "key_71",
        "key_72",
        "key_73",
        "key_74",
        "key_75",
        "key_76",
        "key_77",
        "key_78",
        "key_79",
        "key_80",
        "key_81",
        "key_82",
        "key_83",
        "key_84",
        "key_85",
        "key_86",
        "key_87",
        "key_88",
        "key_89",
        "key_90",
        "key_91",
        "key_92",
        "key_93",
        "key_94",
        "key_95",
        "key_96",
        "key_97",
        "key_98",
        "key_99",
        "key_100",
        "key_101",
        "key_102",
        "key_103",
        "key_104",
        "key_105",
        "key_106",
        "key_107",
        "key_108",
    ]:
        pushing_direction = pred_robot.compute_push_direction(
            key_link)  # compute
        # the normal vector to the button mesh, which is the pushing direction
        lower_point = [0, 0, 0]
        pred_robot.make_prismatic_joint(
            key_link, "coffee_machine_body", lower_point, pushing_direction
        )

    return pred_robot


def partnet_36280(input_dir, links):
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
