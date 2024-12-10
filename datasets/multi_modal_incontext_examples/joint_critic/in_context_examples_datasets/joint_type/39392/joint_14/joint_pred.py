from articulate_anything.api.odio_urdf import *


def partnet_39392(input_dir, links):
    """
    No. masked_links: 16
    Robot Link Summary:
    - base
    - lever
    - knob
    - wheel
    - wheel_2
    - wheel_3
    - wheel_4
    - wheel_5
    - seat
    - caster
    - caster_2
    - caster_3
    - caster_4
    - caster_5
    - chair_leg

    Object: a office chair
    Affordance: the chair can rotate around the chair leg
    """
    pred_robot = Robot(input_dir=input_dir, name="office_chair")
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
    pred_robot.add_link(links['lever'])
    pred_robot.place_relative_to('lever', 'seat',
                                 placement="right",
                                 clearance=0.0)
    pred_robot.add_link(links['caster'])
    pred_robot.place_relative_to('caster', 'chair_leg',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['caster_2'])
    pred_robot.place_relative_to('caster_2', 'chair_leg',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['caster_3'])
    pred_robot.place_relative_to('caster_3', 'chair_leg',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['caster_4'])
    pred_robot.place_relative_to('caster_4', 'chair_leg',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['caster_5'])
    pred_robot.place_relative_to('caster_5', 'chair_leg',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['knob'])
    pred_robot.place_relative_to('knob', 'seat',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['wheel'])
    pred_robot.place_relative_to('wheel', 'caster',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['wheel_2'])
    pred_robot.place_relative_to('wheel_2', 'caster_2',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['wheel_3'])
    pred_robot.place_relative_to('wheel_3', 'caster_3',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['wheel_4'])
    pred_robot.place_relative_to('wheel_4', 'caster_4',
                                 placement="below",
                                 clearance=0.0)
    pred_robot.add_link(links['wheel_5'])
    pred_robot.place_relative_to('wheel_5', 'caster_5',
                                 placement="below",
                                 clearance=0.0)
    # ====================JOINT PREDICTION====================
    pred_robot.make_revolute_joint(
        'seat', 'chair_leg',
        # the seat rotates around the chair leg, so it rotates along the left-right axis,
        global_axis=[0, 1, 0],
        # which is the y-axis.
        lower_angle_deg=0, upper_angle_deg=360,
    )
    return pred_robot
