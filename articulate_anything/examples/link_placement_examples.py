from articulate_anything.api.odio_urdf import *


def partnet_100248(input_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - handle
    - suitcase_body

    Object: a suitcase with a handle
    """
    pred_robot = Robot(input_dir=input_dir, name="suitcase")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['suitcase_body'])
    pred_robot.add_joint(Joint("base_to_suitcase_body",
                         Parent("base"),
                         Child("suitcase_body"),
                         type="fixed"),
                         )
    pred_robot.add_link(links['handle'])
    pred_robot.place_relative_to('handle', 'suitcase_body',
                                 placement="above",
                                 clearance=0.0)
    return pred_robot


def partnet_45397(input_dir, links):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - door
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

    pred_robot.add_link(links['door'])
    # cabinet doors go in front of the furniture body.
    pred_robot.place_relative_to('door', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)

    pred_robot.add_link(links['door_2'])
    pred_robot.place_relative_to('door_2', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)
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


def partnet_102630(input_dir, links):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - lid
    - seat
    - toilet_body

    Object: a toilet
    """
    pred_robot = Robot(input_dir=input_dir, name="toilet")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['toilet_body'])
    pred_robot.add_joint(Joint("base_to_toilet_body",
                         Parent("base"),
                         Child("toilet_body"),
                         type="fixed"),
                         )

    # seat goes on top of the toilet body. No clearance or further adjustment needed
    pred_robot.add_link(links['seat'])
    pred_robot.place_relative_to('seat', 'toilet_body',
                                 placement="above",
                                 clearance=0.0)
    # ========================================================

    pred_robot.add_link(links['lid'])
    pred_robot.place_relative_to('lid', 'toilet_body',
                                 placement="above",
                                 clearance=0.0)
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
    """
    pred_robot = Robot(input_dir=input_dir, name="window")
    pred_robot.add_link(links['base'])

    pred_robot.add_link(links['window_frame'])
    pred_robot.add_joint(Joint("base_to_window_frame",
                         Parent("base"),
                         Child("window_frame"),
                         type="fixed"),
                         )

    for window_idx in range(3):
        window_link = "window" if window_idx == 0 else f"window_{window_idx+1}"
        pred_robot.add_link(links[window_link])

        pred_robot.place_relative_to(window_link, 'window_frame',
                                     placement="inside",
                                     clearance=0.0)

    return pred_robot


def partnet_101840(input_dir, links):
    """
    No. masked_links: 4
    Robot Link Summary:
    - base
    - leg
    - leg_2
    - glasses_body

    Object: a pair of glasses
    """
    pred_robot = Robot(input_dir=input_dir, name="glasses")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['glasses_body'])
    pred_robot.add_joint(Joint("base_to_glasses_body",
                         Parent("base"),
                         Child("glasses_body"),
                         type="fixed"),
                         )

    pred_robot.add_link(links['leg'])
    # leg goes on the right side of the glasses_body.
    pred_robot.place_relative_to('leg', 'glasses_body',
                                 placement="right",
                                 clearance=0.0,
                                 snap_to_place=False)  # avoid snapping
    # so that the leg is fixed at the correct position

    pred_robot.add_link(links['leg_2'])
    pred_robot.place_relative_to('leg_2', 'glasses_body',
                                 placement="left",
                                 clearance=0.0,
                                 snap_to_place=False)
    return pred_robot


def partnet_100783(input_dir, links):
    """
    No. masked_links: 3
    Robot Link Summary:
    - base
    - sphere
    - globe_frame

    Object: a globe with a stand
    """
    pred_robot = Robot(input_dir=input_dir, name="globe")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['globe_frame'])
    pred_robot.add_joint(Joint("base_to_globe_frame",
                         Parent("base"),
                         Child("globe_frame"),
                         type="fixed"),
                         )

    # sphere goes on top of the globe frame. Shift the sphere down
    # by its height so it is pretty much inside the globe frame
    # and not just hanging by the tip of the stick from the globe frame

    pred_robot.add_link(links['sphere'])
    pred_robot.place_relative_to('sphere', 'globe_frame',
                                 placement="above",
                                 clearance=0)

    bb = pred_robot.get_bounding_boxes(['sphere'], include_dim=True)['sphere']
    bb_dim = bb[1]
    pred_robot.place_relative_to('sphere', 'globe_frame',
                                 placement="above",
                                 clearance=-bb_dim['height'])
    return pred_robot


def partnet_45238(input_dir, links):
    """
    No. masked_links: 6
    Robot Link Summary:
    - base
    - door
    - door_2
    - drawer
    - drawer_2
    - furniture_body

    Object: A brown cabinet with two doors and two drawers.
    """
    pred_robot = Robot(input_dir=input_dir, name="cabinet")
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
    # ========================================================
    pred_robot.add_link(links['door_2'])
    pred_robot.place_relative_to('door_2', 'furniture_body',
                                 placement="front",
                                 clearance=0.0)
    # ========================================================
    pred_robot.add_link(links['drawer'])
    pred_robot.place_relative_to('drawer', 'furniture_body',
                                 placement="inside",
                                 clearance=0.0)
    # ========================================================
    pred_robot.add_link(links['drawer_2'])
    pred_robot.place_relative_to('drawer_2', 'furniture_body',
                                 placement="inside",
                                 clearance=0.0)
    return pred_robot


def partnet_100462(input_dir, links):
    """                                                     
    No. masked_links: 3                                     
    Robot Link Summary:                                     
    - base                                                  
    - handle                                                
    - bucket_body                                           

    Object: a bucket with a handle                          
    """
    pred_robot = Robot(input_dir=input_dir, name="bucket")
    pred_robot.add_link(links['base'])
    pred_robot.add_link(links['bucket_body'])
    pred_robot.add_joint(Joint("base_to_bucket_body",
                               Parent("base"),
                               Child("bucket_body"),
                               type="fixed"),
                         )

    pred_robot.add_link(links['handle'])
    pred_robot.place_relative_to('handle', 'bucket_body',
                                 placement="inside",
                                 clearance=0.0)
    return pred_robot




                                                               
def partnet_101946(input_dir, links):                                  
    """                                                                
    No. masked_links: 9                                                
    Robot Link Summary:                                                
    - base                                                             
    - door                                                             
    - knob                                                             
    - knob_2                                                           
    - knob_3                                                           
    - knob_4                                                           
    - knob_5                                                           
    - knob_6                                                           
    - oven_body                                                        
                                                                    
    Object: A silver oven with a door and six knobs.                   
    Targetted affordance: "door"                                       
    """                                                                
    pred_robot = Robot(input_dir=input_dir, name="oven")               
    pred_robot.add_link(links['base'])                                 
    pred_robot.add_link(links['oven_body'])                            
    pred_robot.add_joint(Joint("base_to_oven_body",                    
                        Parent("base"),                               
                        Child("oven_body"),                           
                        type="fixed"),                                
                        )                                             
                                                                    
    pred_robot.add_link(links['door'])                                 
    pred_robot.place_relative_to('door', 'oven_body',                  
                                placement="front",                    
                                clearance=0.0)                        
                                                                    
    for knob_idx in range(6):                                          
        knob_link = "knob" if knob_idx == 0 else f"knob_{knob_idx+1}"  
        pred_robot.add_link(links[knob_link])                          
        pred_robot.place_relative_to(knob_link, 'oven_body',           
                                    placement="front",                
                                    clearance=0.0)  