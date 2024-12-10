from typing import List, Tuple, Union
import cv2
import pybullet as p
import pybullet_data
import numpy as np
from typing import List, Tuple, Optional
from articulate_anything.utils.utils import (
    save_json,
    join_path,
    HideOutput,
)
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import os


def compute_box_dimensions(
    aabb_min: List[float], aabb_max: List[float]
) -> Dict[str, float]:
    """Computes the dimensions (length, width, height) of a box given its AABB."""
    length = aabb_max[0] - aabb_min[0]
    width = aabb_max[1] - aabb_min[1]
    height = aabb_max[2] - aabb_min[2]

    return {"length": length, "width": width, "height": height}


def get_aabb(
    robot_id: int, link_names: Optional[List[str]] = None, include_dim: bool = False
) -> Dict[str, Any]:
    """Gets AABBs for multiple links given their names."""
    link_aabbs = {}
    if link_names is None:
        link_names = get_all_link_names(robot_id)

    for link_name in link_names:
        link_index = get_link_index_by_name(robot_id, link_name)
        if link_index is not None:
            link_aabbs[link_name] = p.getAABB(robot_id, linkIndex=link_index)
        else:
            raise ValueError(f"Link '{link_name}' not found.")
    if include_dim:
        for link_name, aabb in link_aabbs.items():
            link_aabbs[link_name] = (
                aabb, compute_box_dimensions(aabb[0], aabb[1]))
    return link_aabbs


def get_bounding_box_center(aabb_min, aabb_max):
    """Computes the center of a bounding box given its min and max coordinates."""
    return [(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)]


def connect_pybullet(headless: bool = True) -> int:
    """Connects to PyBullet and sets up the environment.

    Args:
        headless (bool): If True, connects to PyBullet in DIRECT mode; otherwise, GUI mode.

    Returns:
        int: The client ID of the connected PyBullet simulation.
    """
    client = p.connect(p.DIRECT if headless else p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    return client


def get_link_bounding_box(
    robot_id: int, link_index: int
) -> Tuple[List[float], List[float]]:
    """Gets the bounding box of a specific link of a robot in PyBullet.

    Args:
        robot_id (int): The unique ID of the robot in PyBullet.
        link_index (int): The index of the link within the robot.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
            - min_bounds: The minimum x, y, z coordinates of the bounding box.
            - max_bounds: The maximum x, y, z coordinates of the bounding box.
    """
    aabbMin, aabbMax = p.getAABB(robot_id, linkIndex=link_index)
    return aabbMin, aabbMax


def get_all_link_names(robot_id: int) -> List[str]:
    """Gets the names of all links of a robot in PyBullet.

    Args:
        robot_id (int): The unique ID of the robot in PyBullet.

    Returns:
        List[str]: A list of the names of all links of the robot.
    """
    return [
        p.getJointInfo(robot_id, i)[12].decode("utf-8")
        for i in range(p.getNumJoints(robot_id))
    ]


def get_link_index_by_name(robot_id: int, link_name: str) -> Optional[int]:
    """Gets the index of a link within a robot in PyBullet based on its name.

    Args:
        robot_id (int): The unique ID of the robot in PyBullet.
        link_name (str): The name of the link.

    Returns:
        Optional[int]: The index of the link, or None if the link name is not found.
    """
    if p.getBodyInfo(robot_id)[0].decode("utf-8") == link_name:
        return -1  # Base link has index -1

    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if (
            joint_info[12].decode("utf-8") == link_name
        ):  # Compare with decoded link name
            return i
    return None  # Link name not found


def get_link_name_by_index(robot_id: int, link_index: int) -> Optional[str]:
    """Gets the name of a link within a robot in PyBullet based on its index.

    Args:
        robot_id (int): The unique ID of the robot in PyBullet.
        link_index (int): The index of the link.

    Returns:
        Optional[str]: The name of the link, or None if the link index is invalid.
    """
    if link_index == -1:
        return p.getBodyInfo(robot_id)[0].decode("utf-8")  # Base link name

    if 0 <= link_index < p.getNumJoints(robot_id):
        joint_info = p.getJointInfo(robot_id, link_index)
        return joint_info[12].decode("utf-8")  # Link name

    return None  # Invalid link index


def get_manipulatable_joints(
    robot_id: int, return_joint_idx: bool = False
) -> List[Union[Tuple[str, int, List[float]], Tuple[int, str, int, List[float]]]]:
    """Returns a list of (joint_index, joint_name, joint_type, joint_limits) or (joint_name, joint_type, joint_limits) tuples for manipulatable joints.

    Args:
        robot_id (int): The unique ID of the robot in PyBullet.
        return_joint_idx (bool): If True, includes the joint index in the returned tuples.

    Returns:
        List[Union[Tuple[str, int, List[float]], Tuple[int, str, int, List[float]]]]: A list of tuples, each containing the joint name, joint type, and joint limits,
        optionally including the joint index.
    """
    manipulatable_joints = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_limits = (joint_info[8], joint_info[9])
            joint_limits = [
                min(joint_limits),
                max(joint_limits),
            ]  # Ensure lower < upper
            joint_name = joint_info[1].decode(
                "utf-8"
            )  # Decode joint name from bytes to string
            if return_joint_idx:
                manipulatable_joints.append((i, joint_info[2], joint_limits))
            else:
                manipulatable_joints.append(
                    (joint_name, joint_info[2], joint_limits))
    return manipulatable_joints


def raise_link_above_ground(
    robot_id: int, link_index: int, clearance: float = 0.0
) -> None:
    """Raises a link of the robot so that its lowest point is a specified distance above the ground.

    Args:
        robot_id (int): The unique ID of the robot in PyBullet.
        link_index (int): The index of the link to be raised.
        clearance (float): The desired distance between the lowest point of the link and the ground.
    """
    aabbMin, aabbMax = p.getAABB(robot_id, linkIndex=link_index)

    # Get lowest Z coordinate of the bounding box
    lowest_z = aabbMin[2]

    # Calculate the amount to raise the link
    raise_distance = clearance - lowest_z

    if raise_distance > 0:  # Only raise if necessary
        # Get current base position and orientation
        base_position, base_orientation = p.getBasePositionAndOrientation(
            robot_id)

        # Create a new position by adding the raise distance to the z coordinate
        new_position = [
            base_position[0],
            base_position[1],
            base_position[2] + raise_distance,
        ]

        # Set the new position for the robot base
        p.resetBasePositionAndOrientation(
            robot_id, new_position, base_orientation)
    return raise_distance


def setup_pybullet(
    urdf_file: str, headless: bool = True, reset_joints: bool = False,
) -> Tuple[int, int]:
    """Sets up the PyBullet simulation with a robot loaded from a URDF file.

    Args:
        urdf_file (str): Path to the URDF file of the robot.
        headless (bool): If True, connects to PyBullet in DIRECT mode; otherwise, GUI mode.
        reset_joints (bool): If True, resets all manipulatable joints to their lower limits.

    Returns:
        Tuple[int, int]: A tuple containing the physics client ID and the robot ID.
    """
    physics_client = connect_pybullet(headless=headless)

    # Load the plane and robot URDF
    plane_id = p.loadURDF("plane.urdf")
    logging.debug(f"Loading urdf_file: {urdf_file}")
    # suppressing pybullet annoying prints
    with HideOutput():
        robot_id = p.loadURDF(urdf_file)

    raise_distances = []
    # Raise the robot links above the ground
    for link_index in range(p.getNumJoints(robot_id)):
        raise_distance = raise_link_above_ground(robot_id, link_index)
        raise_distances.append(raise_distance)

    raise_distance_file = join_path(
        os.path.dirname(urdf_file), "raise_distances.json")

    save_json(
        raise_distances,
        raise_distance_file,
    )

    if reset_joints:
        manipulatable_joints = get_manipulatable_joints(
            robot_id, return_joint_idx=True)
        for joint_idx, joint_type, joint_limits in manipulatable_joints:
            p.resetJointState(robot_id, joint_idx, joint_limits[0])

    return physics_client, robot_id


def get_joint_index_from_name(robot_id: int, joint_name: str) -> int:
    """Finds the joint index for a given joint name.

    Args:
        robot_id (int): The unique ID of the robot in PyBullet.
        joint_name (str): The name of the joint to find.

    Returns:
        int: The index of the joint.
    """
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[1].decode("utf-8") == joint_name:
            return i
    raise ValueError(f"Joint name '{joint_name}' not found.")


def get_joint_with_child(robot_id, child_link_id):
    """
    Returns the joint_id of the joint that has the specified child_link_id.

    Args:
        robot_id (int): The ID of the robot in the simulation.
        child_link_id (int): The ID of the child link.

    Returns:
        int: The joint_id of the joint that has the specified child link.
    """
    num_joints = p.getNumJoints(robot_id)
    for joint_id in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_id)
        if joint_info[16] == child_link_id:  # Check if the child link ID matches
            return joint_id
    raise ValueError(f"No joint found with child link ID: {child_link_id}")
