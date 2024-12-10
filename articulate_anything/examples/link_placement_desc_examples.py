from articulate_anything.api.odio_urdf import *


def partnet_100248(input_dir, links):
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

    # top of the handle is at the top of the suitcase body
    # (i.e. when the handle is retracted, the handle hides inside the suitcase)
    pred_robot.place_relative_to(
        "handle", "suitcase_body", placement="above_inside", clearance=0.0
    )


def partnet_45397(input_dir, links):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - door_1
    - door_2
    - furniture_body
    Object: a cabinet with two doors
    """
    pred_robot = Robot(input_dir=input_dir, name="cabinet")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['furniture_body'])
    pred_robot.add_joint(Joint("base_to_furniture_body",
                         Parent("base"),
                         Child("furniture_body"),
                         type="fixed"),
                         )

    # Add left door
    pred_robot.add_link(links['left_door'])
    pred_robot.place_relative_to(
        "door_1", "furniture_body", placement="left_inside", clearance=0.0
    )  # Description says door_1 is left_door.
    # left side of `left_door` is aligned with left side of `furniture_body`

    # Add right door
    pred_robot.add_link(links['right_door'])
    pred_robot.place_relative_to(
        "door_2", "furniture_body", placement="right_inside", clearance=0.0
    )  # Description says door_2 is right_door.
    # right side of `right_door` is aligned with right side of `furniture_body`
    return pred_robot


def partnet_47645(input_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - lid
    - box_body

    Object: a treasure chest
    """
    pred_robot = Robot(input_dir=input_dir, name="treasure_chest")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['box_body'])
    pred_robot.add_joint(Joint("base_to_box_body",
                         Parent("base"),
                         Child("box_body"),
                         type="fixed"),
                         )

    # lid goes on top of the box. No clearance or further adjustment needed
    pred_robot.add_link(links['lid'])
    pred_robot.place_relative_to('lid', 'box_body',
                                 placement="above",
                                 clearance=0.0)
    return pred_robot


def partnet_3398(input_dir, links):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - lid
    - bottle_body

    Object: a bottle of soap
    """
    pred_robot = Robot(input_dir=input_dir, name="soap_bottle")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['bottle_body'])
    pred_robot.add_joint(Joint("base_to_bottle_body",
                         Parent("base"),
                         Child("bottle_body"),
                         type="fixed"),
                         )

    # lid goes on top of the bottle. No clearance or further adjustment needed
    pred_robot.add_link(links['lid'])
    pred_robot.place_relative_to('lid', 'bottle_body',
                                 placement="above",
                                 clearance=0.0)
    return pred_robot


def partnet_44962(input_dir, links):
    """
    No. masked_links: 5
    Robot Link Summary:
    - base
    - drawer_1
    - drawer_2
    - drawer_3
    - furniture_body

    Object: a cabinet with 3 drawers
    """
    pred_robot = Robot(input_dir=input_dir, name="cabinet")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['furniture_body'])
    pred_robot.add_joint(Joint("base_to_furniture_body",
                         Parent("base"),
                         Child("furniture_body"),
                         type="fixed"),
                         )

    pred_robot.add_link(links['drawer_1'])
    # description says drawer_1 is the middle drawer
    pred_robot.place_relative_to('drawer_1', 'furniture_body',
                                 placement="inside",  # in the middle
                                 # origin of the middle drawer is aligned with the origin of the furniture body
                                 clearance=0.0)
    # ========================================================
    # description says drawer_2 is the top drawer
    pred_robot.add_link(links['drawer_2'])
    pred_robot.place_relative_to('drawer_2', 'furniture_body',
                                 placement="above_inside",  # top has to above
                                 # but align the top part of the drawer with the top part of the furniture body
                                 clearance=0.0)
    # ========================================================
    # description says drawer_3 is the bottom drawer
    pred_robot.add_link(links['drawer_3'])
    pred_robot.place_relative_to('drawer_3', 'furniture_body',
                                 placement="below_inside",
                                 # bottom has to be below but align the bottom part of the drawer with the bottom part of the furniture body
                                 clearance=0.0)
    return pred_robot


def partnet_41510(input_dir, links):
    """
    No. masked_links: 5
    Robot Link Summary:
    - base
    - drawer
    - door
    - furniture_body

    Object: a cabinet with a top drawer and a bottom door
    """
    pred_robot = Robot(input_dir=input_dir, name="cabinet")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['furniture_body'])
    pred_robot.add_joint(Joint("base_to_furniture_body",
                         Parent("base"),
                         Child("furniture_body"),
                         type="fixed"),
                         )

    # description says drawer is the top drawer
    pred_robot.add_link(links['drawer'])
    pred_robot.place_relative_to('drawer', 'furniture_body',
                                 placement="above_inside",
                                 clearance=0.0)
    # ========================================================
    # description says door is the bottom door
    pred_robot.add_link(links['door'])
    pred_robot.place_relative_to('door', 'furniture_body',
                                 placement="below_inside",
                                 clearance=0.0)
    return pred_robot


def partnet_39015(input_dir, links):
    """
    No. masked_links: 13
    Robot Link Summary:
    - base
    - wheel_1
    - wheel_2
    - wheel_3
    - wheel_4
    - seat 
    - chair_leg

    Object: a chair with five wheels
    Affordance: rotating the seat
    """
    pred_robot = Robot(input_dir=input_dir, name="chair")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['chair_leg'])
    pred_robot.add_joint(Joint("base_to_chair_leg",
                         Parent("base"),
                         Child("chair_leg"),
                         type="fixed"),
                         )
    pred_robot.add_link(links['seat'])
    pred_robot.place_relative_to('seat', 'chair_leg',
                                 placement="above",
                                 clearance=0.0)
    for wheel in ['wheel_1', 'wheel_2', 'wheel_3', 'wheel_4']:
        pred_robot.add_link(links[wheel])
        pred_robot.place_relative_to(wheel, 'chair_leg',
                                     placement="below",
                                     clearance=0.0)
    return pred_robot
