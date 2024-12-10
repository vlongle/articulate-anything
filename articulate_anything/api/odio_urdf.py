"""
Adapted from https://github.com/hauptmech/odio_urdf
Original author: Hauptmech
"""

import re
import sys
import importlib
import importlib.util
from typing import List, Dict, Any, Optional, Union, Tuple
import six
import inspect
import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import os
from collections import defaultdict, deque
from functools import wraps
import trimesh
from articulate_anything.api.odio_urdf_utils import (
    ElementMeta,
    NamedElementMeta,
    instantiate_if_class,
    literal_as_str,
    classname,
    eval_macros,
)
import logging
from articulate_anything.utils.utils import (
    string_to_file,
    join_path,
    save_json,
    load_json,
)
from articulate_anything.physics.pybullet_utils import (
    setup_pybullet,
    get_aabb,
    get_link_index_by_name,
    get_bounding_box_center,
)
import pybullet as p


def pybullet_session(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Setup PyBullet
        tmp_file = join_path(self.input_dir, "temp_robot.urdf")
        string_to_file(str(self), tmp_file)
        client, robot_id = setup_pybullet(tmp_file, headless=True,
                                          )

        try:
            # Execute the function
            result = func(self, *args, **kwargs,
                          client=client, robot_id=robot_id)
        finally:
            # Teardown PyBullet
            p.disconnect(client)
            os.remove(tmp_file)

        return result

    return wrapper


def find_joint_with_parent(root, parent_name):
    """Finds the first joint element in the URDF with the specified parent."""
    for joint_element in root.findall(".//joint"):
        parent_element = joint_element.find("parent")
        if (
            parent_element is not None
            and parent_element.attrib.get("link") == parent_name
        ):
            return joint_element
    return None


def extract_rpy_from_joint(joint_element):
    """Extracts the rpy values from a joint element in the URDF,
    or returns the default value "0 0 0" if it's not specified."""
    origin_element = joint_element.find("origin")
    if origin_element is not None:
        # Use get with default value
        return origin_element.attrib.get("rpy", "0 0 0")
    else:
        return "0 0 0"  # Return default if origin element is missing


@six.add_metaclass(ElementMeta)
class Element(list):
    """
    Parent class for all URDF elements.

    All element classes have class attributes that define what sub elements and xml attributes are
    allowed or required.

    required_elements: xml sub-elements that MUST be present.
    allowed_elements: xml sub-elements that MAY be present.
    required_attributes: xml attributes that MUST be present.
    allowed_attributes: xml attributes that MAY be present.
    """

    element_counter = 0
    string_macros = {}
    xacro_tags = ["Xacroproperty", "Xacroinclude", "Xacroif", "Xacrounless"]
    element_name = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.attributes = set()
        self._instantiated = {}
        self.xmltext = ""

        up_frame = inspect.currentframe().f_back
        if up_frame.f_back:
            up_frame = up_frame.f_back
        callers_local_vars = up_frame.f_locals.items()

        # __call__() adds and sets all our attributes and elements
        self._populate_element_(*args, **kwargs)

        # Create defaults for any required elements that have not been created yet.
        for item in type(self).required_elements:
            if item not in self._instantiated:
                new_child = instantiate_if_class(globals()[item])
                self.append(new_child)
                new_child.parent = self

    def get_named_elements(self, element_type: Union[type, str]) -> Dict[str, Any]:
        """
        Retrieves a dictionary of named elements of the specified type from this NamedElement.

        Args:
            element_type (type or str): The type of element to retrieve (e.g., Visual, Collision).
                                        Can be either the class itself or its name as a string.

        Returns:
            dict: A dictionary where keys are element names and values are the corresponding elements.
        """
        if isinstance(element_type, str):
            element_type = globals().get(element_type)  # Get class from global scope
            if element_type is None:
                raise ValueError(f"Invalid element type: {element_type}")

        return [item for item in self if isinstance(item, element_type)]

    def __call__(self, *args: Any, **kwargs: Any) -> "Element":
        """
        For our DSL, allow existing instances to be populated with sub-elements the same way
        we allow during instantiation.

        Example:
            # populate with a link during object creation
            robot = Robot('my_robot_name', Link('my_link_name'))
            robot(Link('My_second_link')) # Add a second link to the instance
        """
        self._populate_element_(*args, **kwargs)
        return self

    def _populate_element_(self, *args: Any, **kwargs: Any) -> None:
        """
        Populate a URDF element with attributes and sub elements.

        *args of type str and int will be assigned one-by-one to the attributes in Class.arm_attributes.
        *args derived from Element will be added to the sub-element list.

        **kwargs will be assigned to the attribute implied by the keyword.

        xmltext = "<somexml/>" will be directly injected into the output.
        """
        if "xmltext" in kwargs:
            self.xmltext = kwargs["xmltext"]
            del kwargs["xmltext"]

        if "xacro_xml" in kwargs:
            xacroroot = ET.fromstring(
                kwargs["xacro_xml"]
            )  # The root should be <Robot/>
            for child in xacroroot:
                # Add the xml to our <Robot>
                if sys.version_info[0] < 3:
                    self.xmltext += ET.tostring(child)
                else:
                    self.xmltext += ET.tostring(child, encoding="unicode")
            del kwargs["xacro_xml"]

        callers_local_vars = inspect.currentframe().f_back.f_locals.items()

        allowed_attributes = (
            type(self).required_attributes + type(self).allowed_attributes
        )
        name = ""
        unlabeled = 0  # Count of unlabeled strings we have encountered so far
        allowed_unlabeled = len(allowed_attributes)
        for arg in args:
            arg_type = type(arg)
            if arg_type in [str, float, int, tuple, list]:
                if unlabeled < allowed_unlabeled:
                    setattr(
                        self, allowed_attributes[unlabeled], literal_as_str(arg))
                    self.attributes.add(allowed_attributes[unlabeled])
                    unlabeled += 1
            elif arg_type is Group:
                for elt in arg:
                    self.append(elt)
                if hasattr(arg, "xmltext"):
                    self.xmltext += arg.xmltext
            else:
                name = classname(arg)

                if name in self.required_elements + self.allowed_elements:
                    new_child = instantiate_if_class(arg)
                    self.append(new_child)
                    new_child.parent = self

                    self._instantiated[name] = (
                        None  # Keep track of Elements we instantiate
                    )

                    # If name is required and not there already, add it using the variable name
                    if (
                        "name" in type(new_child).required_attributes
                        and not "name" in new_child.attributes
                    ):
                        # If we were a named variable, use it
                        name_val_list = [
                            (var_name, var_val)
                            for var_name, var_val in callers_local_vars
                            if var_val is arg
                        ]
                        if len(name_val_list) > 0:
                            # Use most recent label
                            name_val = name_val_list[-1][0]
                            new_child.name = name_val
                            new_child.attributes.add("name")

                elif name in Element.xacro_tags:
                    pass

                else:
                    raise Exception("Illegal element [" + name + "]")

        for key, value in kwargs.items():
            if key in allowed_attributes:
                setattr(self, key, literal_as_str(value))
                self.attributes.add(key)
            else:
                raise Exception(
                    "Attribute ["
                    + key
                    + "] is not in allowed_attributes list of "
                    + str(type(self))
                )

    def __str__(self) -> str:
        return self.urdf()

    def __repr__(self) -> str:
        return self.urdf()

    def urdf(self, depth: int = 0) -> str:
        name = type(self).__name__.lower()
        if self.element_name:
            name = self.element_name
        s = " " * depth + "<" + name + " "
        if hasattr(self, "attributes"):
            for attr in self.attributes:
                to_insert = str(getattr(self, attr))
                if isinstance(to_insert, tuple):
                    to_insert = str(to_insert).strip(
                        "(").strip(")").replace(",", "")

                s += (
                    " "
                    + str(attr)
                    + '="'
                    + eval_macros(to_insert, Element.string_macros)
                    + '" '
                )
            # Flag required but unnamed attributes
            for attr in set(type(self).required_attributes).difference(self.attributes):
                s += (
                    " "
                    + str(attr)
                    + '="'
                    + "UNNAMED_"
                    + str(Element.element_counter)
                    + '" '
                )
                Element.element_counter += 1
        if len(self) == 0 and self.xmltext == "":
            s += "/>\n"
        else:
            s += ">\n"

            for elt in self:
                s += elt.urdf(depth + 1)
            if self.xmltext != "":
                s += " " * (depth + 1) + self.xmltext + "\n"
            s += " " * depth + "</" + type(self).__name__.lower() + ">\n"
        return s


@six.add_metaclass(NamedElementMeta)
class NamedElement(Element):
    pass


############# elements #############


class Xacroinclude(Element):
    allowed_attributes = ["filename"]


class Xacrounless(Element):
    pass


class Xacroif(Element):
    pass


class Xacroproperty(Element):
    def __init__(self, **kwargs: Any) -> None:
        if "name" in kwargs and "value" in kwargs:
            Element.string_macros[kwargs["name"]] = float(kwargs["value"])


class Group(Element):
    """A group of <Robot> top level elements that will be appended to the Robot() that owns this group"""

    allowed_elements = ["Joint", "Link", "Material", "Transmission", "Gazebo"]
    allowed_attributes = ["name"]


def compute_aabb_vertices(
    aabb_min: Union[List[float], np.ndarray], aabb_max: Union[List[float], np.ndarray]
) -> np.ndarray:
    """
    Computes the 8 vertices of an axis-aligned bounding box. x-axis is front back, y-axis is left right, z-axis is up down.

    Args:
    aabb_min (list or numpy.ndarray): A list or array of 3 elements representing the minimum x, y, and z coordinates.
    aabb_max (list or numpy.ndarray): A list or array of 3 elements representing the maximum x, y, and z coordinates.

    Returns:
    numpy.ndarray: An 8x3 array where each row represents the coordinates of a vertex.
    The vertices are ordered as follows:
    0: Back-Left-Bottom (BLB)
    1: Back-Right-Bottom (BRB)
    2: Front-Left-Bottom (FLB)
    3: Front-Right-Bottom (FRB)
    4: Back-Left-Top (BLT)
    5: Back-Right-Top (BRT)
    6: Front-Left-Top (FLT)
    7: Front-Right-Top (FRT)
    """
    x_min, y_min, z_min = aabb_min
    x_max, y_max, z_max = aabb_max

    vertices = np.array(
        [
            [x_min, y_min, z_min],  # Back-Left-Bottom (BLB)
            [x_min, y_max, z_min],  # Back-Right-Bottom (BRB)
            [x_max, y_min, z_min],  # Front-Left-Bottom (FLB)
            [x_max, y_max, z_min],  # Front-Right-Bottom (FRB)
            [x_min, y_min, z_max],  # Back-Left-Top (BLT)
            [x_min, y_max, z_max],  # Back-Right-Top (BRT)
            [x_max, y_min, z_max],  # Front-Left-Top (FLT)
            [x_max, y_max, z_max],  # Front-Right-Top (FRT)
        ]
    )

    return vertices


def world_to_local(
    world_position: List[float],
    reference_orientation: List[float],
    reference_base_pos: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Converts a target position from the world frame to a link's local frame.
    `reference_base_pos` is used e.g, when we want to compute the offset to the parent origin
    (i.e., the origin tag of the joint) such that we're shifted to `world_position`. The more common
    use case is when `world_position` is already just the offset (i.e., for `place_relative_to` function)

    Args:
        world_position (list): The target position in the world frame.
        reference_orientation (list): The orientation of the reference link.
        reference_base_pos (list, optional): The base position of the reference link in the world frame.

    Returns:
        numpy.ndarray: The position in the local frame of the reference link.
    """
    if reference_base_pos is not None:
        world_position = np.array(world_position) - \
            np.array(reference_base_pos)
    ref_rotation = R.from_quat(reference_orientation)
    local_position = ref_rotation.inv().apply(world_position)
    return local_position


class Robot(NamedElement):
    """Robot is the top level element in a URDF"""

    allowed_elements = ["Joint", "Link", "Material", "Transmission", "Gazebo"]
    allowed_attributes = ["input_dir"]

    def __init__(self, input_dir, *args: Any, **kwargs: Any) -> None:
        self.input_dir = input_dir
        self._base_joint_added = False
        super(Robot, self).__init__(*args, **kwargs)

    def urdf(self, depth: int = 0) -> str:
        return '<?xml version="1.0"?>\n' + super(Robot, self).urdf(0)

    def summarize(self) -> str:
        """Prints a summary of the robot's link names."""
        link_names = [link.name for link in self]
        summary = "Robot Link Summary:\n\n"
        for name in link_names:
            summary += f" - {name}\n"
        return summary

    def align_robot_orientation(self):
        """Aligns the robot's base joint orientation with the ground truth from the URDF."""
        joint = self.get_joint_for_parent("base")
        if joint is None:
            raise ValueError(
                "Have to have a joint with parent as base before calling this function"
            )

        urdf_path = join_path(self.input_dir, "mobility.urdf")
        if not os.path.exists(urdf_path):
            return

        root = load_urdf(urdf_path)
        joint_element = find_joint_with_parent(root, "base")

        if joint_element is None:
            raise ValueError("No joint with parent 'base' found in URDF")

        ground_truth_rpy = extract_rpy_from_joint(joint_element)
        origin = Origin(xyz="0 0 0", rpy=ground_truth_rpy)
        origins = joint.get_named_elements("Origin")
        if not origins:
            joint.append(origin)
        else:
            origins[0].rpy = origin.rpy

    def add_joint(self, joint: "Joint") -> None:
        """Adds a Joint to the robot."""
        if not isinstance(joint, Joint):
            raise TypeError("Expected a Joint object")

        self.append(joint)
        if not self._base_joint_added:
            parent_link = joint.get_named_elements("Parent")[0].link
            if parent_link == "base":
                self._base_joint_added = True
                self.align_robot_orientation()

    def get_links(self) -> Dict[str, "Link"]:
        return {link.name: link for link in self if isinstance(link, Link)}

    def get_joints(self) -> Dict[str, "Joint"]:
        return {joint.name: joint for joint in self if isinstance(joint, Joint)}

    def get_manipulatable_joints(self) -> Dict[str, "Joint"]:
        return {
            name: joint
            for name, joint in self.get_joints().items()
            if joint.type in ["revolute", "prismatic"]
        }

    def add_link(self, link: "Link") -> None:
        """
        Adds a Link to the robot if it doesn't already exist.

        Args:
            link (Link): The Link object to be added.

        Raises:
            TypeError: If the provided object is not a Link.
            ValueError: If a link with the same name already exists.
        """
        if not isinstance(link, Link):
            raise TypeError("Expected a Link object")

        # Check if a link with the same name already exists
        existing_links = self.get_links()
        if link.name in existing_links:
            logging.warning(
                f"A link with the name '{link.name}' already exists in the robot"
            )
            return

        self.append(link)

    def rotate_link(self, link_name: str, axis: str, angle_degree: float) -> None:
        """Rotates a link around a specified axis by the given angle in degrees."""
        links = self.get_links()

        if link_name not in links:
            raise ValueError(f"Link '{link_name}' not found in the robot.")

        link = links[link_name]
        for visual in link.get_named_elements("Visual"):
            self._apply_rotation_to_origin(visual, axis, angle_degree)

        for collision in link.get_named_elements("Collision"):
            self._apply_rotation_to_origin(collision, axis, angle_degree)

    def rotate_joint(self, joint_name: str, axis: str, angle_degree: float) -> None:
        """Rotates a joint around a specified axis by the given angle in degrees."""
        joints = self.get_joints()
        if joint_name not in joints:
            raise ValueError(f"Joint '{joint_name}' not found in the robot.")

        joint = joints[joint_name]
        self._apply_rotation_to_origin(joint, axis, angle_degree)

    def _translate_element(self, element: "Element", translation: List[float]) -> None:
        """Helper function to translate a joint or link element."""
        origins = element.get_named_elements("Origin")
        if not origins:
            element.append(Origin(xyz="0 0 0"))
            origins = element.get_named_elements("Origin")

        origin = origins[0]
        current_xyz = [float(x) for x in origin.xyz.split()]
        new_xyz = [current_xyz[i] + translation[i] for i in range(3)]
        origin.xyz = " ".join(map(str, new_xyz))

    def translate_joint(self, joint_name: str, translation: List[float]) -> None:
        """Translates a joint RELATIVE to its parent link."""
        joints = self.get_joints()
        if joint_name not in joints:
            raise ValueError(f"Joint '{joint_name}' not found in the robot.")

        joint = joints[joint_name]
        self._translate_element(joint, translation)

    def get_origin_of_joint(self, joint_name: str) -> "Origin":
        joints = self.get_joints()
        if joint_name not in joints:
            raise ValueError(f"Joint '{joint_name}' not found in the robot.")

        joint = joints[joint_name]
        origins = joint.get_named_elements("Origin")
        if origins:
            origin = origins[0]

            # Check if the origin has the rpy attribute, otherwise default to 0
            if hasattr(origin, "rpy"):
                return origin
            else:
                return Origin(
                    xyz=origin.xyz, rpy="0 0 0"
                )  # Use xyz from the existing origin

        # Default if no origin is specified
        return Origin(xyz="0 0 0", rpy="0 0 0")

    def get_origin_of_link(self, link_name: str) -> "Origin":
        links = self.get_links()
        if link_name not in links:
            raise ValueError(f"Link '{link_name}' not found in the robot.")

        link = links[link_name]
        # get from visual
        for visual in link.get_named_elements("Visual"):
            origins = visual.get_named_elements("Origin")
            if origins:
                return origins[0]
        return Origin(xyz="0 0 0", rpy="0 0 0")

    def translate_link(self, link_name: str, translation: List[float]) -> None:
        """Translates a link RELATIVE to its parent joint."""
        links = self.get_links()

        if link_name not in links:
            raise ValueError(f"Link '{link_name}' not found in the robot.")

        link = links[link_name]
        for visual in link.get_named_elements("Visual"):
            self._translate_element(visual, translation)

        for collision in link.get_named_elements("Collision"):
            self._translate_element(collision, translation)

    @pybullet_session
    def get_bounding_boxes(
        self,
        link_names: List[str] = None,
        include_dim: bool = False,
        client=None,
        robot_id=None,
    ) -> Dict[str, Any]:
        """
        Gets the AABB bounding boxes for specified links on a robot in PyBullet.

        Args:
            link_names (list): A list of strings representing the names of the links to get bounding boxes for.
            include_dim (bool): If True, include the dimensions of the bounding box.

        Returns:
            dict: A dictionary mapping link names to tuples of (aabb_min, aabb_max) coordinates.
        """
        link_aabbs = get_aabb(robot_id, link_names, include_dim=include_dim)
        return link_aabbs

    def get_parent_child_bounding_boxes(
        self, child_link_name: str, parent_link_name: str
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Retrieves the bounding boxes of the child and parent links.

        Args:
            child_link_name (str): The name of the child link.
            parent_link_name (str): The name of the parent link.

        Returns:
            tuple: A tuple containing the bounding boxes of the child and parent links:
                - child_aabb_min (list): Minimum coordinates of the child's AABB.
                - child_aabb_max (list): Maximum coordinates of the child's AABB.
                - parent_aabb_min (list): Minimum coordinates of the parent's AABB.
                - parent_aabb_max (list): Maximum coordinates of the parent's AABB.
        """
        link_aabbs = self.get_bounding_boxes(
            [child_link_name, parent_link_name])

        return (
            link_aabbs[child_link_name][0],
            link_aabbs[child_link_name][1],
            link_aabbs[parent_link_name][0],
            link_aabbs[parent_link_name][1],
        )

    def get_joint_for_child(self, child_link_name: str) -> Optional["Joint"]:
        """Gets the joint for a child link.

        Args:
            child_link_name (str): Name of the child link.

        Returns:
            Joint: The joint object for the child link, or None if not found.
        """
        for joint in self.get_joints().values():
            if joint.get_named_elements("Child")[0].link == child_link_name:
                return joint
        return None

    def get_joint_for_parent(self, parent_link_name: str) -> Optional["Joint"]:
        """Gets the joint for a parent link.

        Args:
            parent_link_name (str): Name of the parent link.

        Returns:
            Joint: The joint object for the parent link, or None if not found.
        """
        for joint in self.get_joints().values():
            if joint.get_named_elements("Parent")[0].link == parent_link_name:
                return joint
        return None

    def get_joint_between(
        self, child_link_name: str, parent_link_name: str
    ) -> Optional["Joint"]:
        """Gets the joint connecting the child and parent links.

        Args:
            child_link_name (str): Name of the child link.
            parent_link_name (str): Name of the parent link.

        Returns:
            Joint: The joint object connecting the links, or None if not found.
        """
        for joint in self.get_joints().values():
            child = joint.get_named_elements("Child")[0]  # Assuming one child
            parent = joint.get_named_elements(
                "Parent")[0]  # Assuming one parent
            if child.link == child_link_name and parent.link == parent_link_name:
                return joint
        return None  # Joint not found

    @pybullet_session
    def get_link_states(
        self, link_names: List[str], client=None, robot_id=None
    ) -> Dict[str, Any]:
        link_states = {}
        for link_name in link_names:
            link_index = get_link_index_by_name(robot_id, link_name)
            if link_index is None:
                raise ValueError(f"Link '{link_name}' not found.")

            if link_index == -1:  # Base link
                pos, orn = p.getBasePositionAndOrientation(robot_id)
                # Create a tuple similar to getLinkState output
                link_states[link_name] = (
                    pos,
                    orn,
                    (0, 0, 0),
                    (0, 0, 0, 1),
                    pos,
                    orn,
                    (0, 0, 0),
                    (0, 0, 0),
                )
            else:
                link_states[link_name] = p.getLinkState(robot_id, link_index)

        return link_states

    def get_parent_link_name(self, child_link_name: str) -> Optional[str]:
        """
        Retrieves the parent link name for a given child link name from existing joints.

        Args:
            child_link_name (str): The name of the child link.

        Returns:
            Optional[str]: The name of the parent link if found, None otherwise.
        """
        for joint in self.get_joints().values():
            child = joint.get_named_elements("Child")[0]
            if child.link == child_link_name:
                parent = joint.get_named_elements("Parent")[0]
                return parent.link
        return None

    def place_relative_to(
        self,
        child_link_name: str,
        parent_link_name: str,
        placement: str = "above",
        clearance: float = 0.0,
        snap_to_place: bool = True,
        overwrite: bool = True,
    ) -> None:
        """
        Places the child link relative to the parent link's bounding box with a specified placement and clearance,
        accounting for the relative orientation of the links.

        Args:
            child_link_name (str): Name of the child link.
            parent_link_name (str): Name of the parent link.
            placement (str): Placement direction. Options are:
                "above", "above_inside", "below", "below_inside",
                "left", "left_inside", "right", "right_inside",
                "front", "front_inside", "back", "back_inside",
                "inside"
            clearance (float): Distance to maintain between the links.
            snap_to_place (bool): Whether to snap the child to place after initial positioning.
            use_bounding_box_center (bool): Whether to use bounding box center for positioning.
            overwrite (bool): Whether to overwrite the existing joint transformation or add to it.
        """
        # Get the joint between child and parent links
        joint = self.get_joint_between(child_link_name, parent_link_name)

        # If no joint exists, create a new fixed joint
        if joint is None:
            self.add_joint(
                Joint(
                    f"{parent_link_name}_to_{child_link_name}",
                    Parent(parent_link_name),
                    Child(child_link_name),
                    type="fixed",
                )
            )
            joint = self.get_joint_between(child_link_name, parent_link_name)

        # Get bounding boxes for both child and parent links
        child_aabb_min, child_aabb_max, parent_aabb_min, parent_aabb_max = (
            self.get_parent_child_bounding_boxes(
                child_link_name, parent_link_name)
        )

        # Get current positions and orientations of child and parent links
        link_states = self.get_link_states([child_link_name, parent_link_name])
        child_link_state = link_states[child_link_name]
        parent_link_state = link_states[parent_link_name]

        child_current_position, child_current_orientation = child_link_state[
            0], child_link_state[1]
        parent_current_position, parent_current_orientation = parent_link_state[
            0], parent_link_state[1]

        # Initialize target position
        target_position = np.array([0.0, 0.0, 0.0])

        if placement == "above":
            # Place bottom of child at top of parent (child outside)
            target_position[2] += parent_aabb_max[2] - \
                child_aabb_min[2] + clearance
            axis, direction = 2, -1
        elif placement == "above_inside":
            # Place top of child at top of parent (child inside)
            target_position[2] += parent_aabb_max[2] - \
                child_aabb_max[2] - clearance
            axis, direction = 2, -1
        elif placement == "below":
            # Place top of child at bottom of parent (child outside)
            target_position[2] += parent_aabb_min[2] - \
                child_aabb_max[2] - clearance
            axis, direction = 2, 1
        elif placement == "below_inside":
            # Place bottom of child at bottom of parent (child inside)
            target_position[2] += parent_aabb_min[2] - \
                child_aabb_min[2] + clearance
            axis, direction = 2, 1
        elif placement == "left":
            # Place right of child at left of parent (child outside)
            target_position[1] += parent_aabb_min[1] - \
                child_aabb_max[1] - clearance
            axis, direction = 1, 1
        elif placement == "left_inside":
            # Place left of child at left of parent (child inside)
            target_position[1] += parent_aabb_min[1] - \
                child_aabb_min[1] + clearance
            axis, direction = 1, 1
        elif placement == "right":
            # Place left of child at right of parent (child outside)
            target_position[1] += parent_aabb_max[1] - \
                child_aabb_min[1] + clearance
            axis, direction = 1, -1
        elif placement == "right_inside":
            # Place right of child at right of parent (child inside)
            target_position[1] += parent_aabb_max[1] - \
                child_aabb_max[1] - clearance
            axis, direction = 1, -1
        elif placement == "front":
            # Place back of child at front of parent (child outside)
            target_position[0] += parent_aabb_max[0] - \
                child_aabb_min[0] + clearance
            axis, direction = 0, -1
        elif placement == "front_inside":
            # Place front of child at front of parent (child inside)
            target_position[0] += parent_aabb_max[0] - \
                child_aabb_max[0] - clearance
            axis, direction = 0, -1
        elif placement == "back":
            # Place front of child at back of parent (child outside)
            target_position[0] += parent_aabb_min[0] - \
                child_aabb_max[0] - clearance
            axis, direction = 0, 1
        elif placement == "back_inside":
            # Place back of child at back of parent (child inside)
            target_position[0] += parent_aabb_min[0] - \
                child_aabb_min[0] + clearance
            axis, direction = 0, 1
        elif placement == "inside":
            # Place center of child at center of parent
            pass
        else:
            raise ValueError(f"Invalid placement: {placement}")

          # Transform target position from world coordinates to parent's local coordinates
        local_target_position = world_to_local(
            target_position, parent_current_orientation)

        # Adjust the target position by subtracting the current movement in local coordinates
        if overwrite:
            # Calculate the current movement (offset) between child and parent
            current_mov = np.array(child_current_position) - \
                np.array(parent_current_position)
            local_target_position = local_target_position - \
                world_to_local(current_mov, parent_current_orientation)

        self.translate_joint(joint.name, local_target_position)

        # Snap the child to place if specified (except for center placement)
        if snap_to_place and placement != "inside":
            self.snap_to_place(
                child_link_name,
                parent_link_name,
                axis,
                direction,
                parent_current_orientation,
            )

    def snap_to_place(
        self,
        child_link_name,
        parent_link_name,
        axis,
        direction,
        parent_current_orientation,
        max_iter=100,
        increment=0.01,
    ):
        distance = 0.0
        collision = self.is_collision(child_link_name, parent_link_name)
        joint = self.get_joint_between(child_link_name, parent_link_name)
        i = 0
        while not collision and i < max_iter:
            target_position = np.array([0.0, 0.0, 0.0])
            target_position[axis] += direction * increment
            local_target_position = world_to_local(
                target_position, parent_current_orientation
            )
            self.translate_joint(joint.name, local_target_position)
            collision = self.is_collision(child_link_name, parent_link_name)
            distance += increment
            i += 1

    @pybullet_session
    def is_collision(
        self,
        child_link_name: str,
        parent_link_name: str,
        distance: float = 0.0,
        client=None,
        robot_id=None,
    ) -> bool:
        child = get_link_index_by_name(robot_id, child_link_name)
        parent = get_link_index_by_name(robot_id, parent_link_name)

        for link in self.get_links().keys():
            if link == "base" or link == child_link_name:
                continue
            link_index = get_link_index_by_name(robot_id, link)
            p.stepSimulation()

            linkIndexA = child
            linkIndexB = link_index
            # linkIndexB = parent

            collisions = p.getClosestPoints(
                bodyA=robot_id,
                bodyB=robot_id,
                linkIndexA=linkIndexA,
                linkIndexB=linkIndexB,
                distance=distance,
            )

            if len(collisions) > 0:
                return True

        return False

    def _create_rotation_matrix(self, axis: str, angle_radians: float) -> np.ndarray:
        """Creates a 3D rotation matrix based on the axis and angle."""
        if axis == "x":
            return R.from_euler("x", angle_radians).as_matrix()
        elif axis == "y":
            return R.from_euler("y", angle_radians).as_matrix()
        elif axis == "z":
            return R.from_euler("z", angle_radians).as_matrix()
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    def _apply_rotation_to_origin(
        self, element: "Element", axis: str, angle_degrees: float
    ) -> None:
        """Applies a rotation to an element's origin (joint or visual)."""
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = self._create_rotation_matrix(axis, angle_radians)

        origins = element.get_named_elements("Origin")

        if not origins:  # Check if any origins exist
            element.append(
                Origin(xyz="0 0 0", rpy="0 0 0")
            )  # Add origin if it doesn't exist
            origins = element.get_named_elements("Origin")

        origin = origins[0]

        current_rpy = np.array([float(x) for x in origin.rpy.split()])
        current_rotation = R.from_euler("xyz", current_rpy).as_matrix()
        new_rotation = np.dot(current_rotation, rotation_matrix)
        new_rpy = R.from_matrix(new_rotation).as_euler("xyz")
        origin.rpy = f"{new_rpy[0]} {new_rpy[1]} {new_rpy[2]}"

    def compute_prismatic_joint(
        self,
        global_lower_point: List[float],
        global_upper_point: List[float],
        parent_orientation: List[float],
    ) -> Dict[str, Any]:
        """
        Computes the prismatic joint parameters.

        When the joint is at the global lower point.
        When the joint is at the global upper point.

        Args:
            global_lower_point (list): The global coordinates of the lower point.
            global_upper_point (list): The global coordinates of the upper point.
            parent_orientation (list): The orientation of the parent link.

        Returns:
            dict: The prismatic joint parameters.
        """
        global_lower_point = np.array(global_lower_point)
        global_upper_point = np.array(global_upper_point)

        # Compute the movement axis and limits
        axis = global_upper_point - global_lower_point
        local_axis = world_to_local(axis, parent_orientation)
        distance = np.linalg.norm(axis)
        local_axis = local_axis / np.linalg.norm(
            local_axis
        )  # Re-normalize the axis in the parent frame
        lower_limit = 0
        upper_limit = distance

        local_lower_point = world_to_local(
            global_lower_point, parent_orientation)
        local_lower_point = local_lower_point.tolist()
        local_lower_point = [str(x) for x in local_lower_point]

        return {
            "axis": Axis(local_axis.tolist()),
            "limit": Limit(
                lower=lower_limit, upper=upper_limit, velocity="5", effort="5"
            ),
            "origin": Origin(xyz=" ".join(local_lower_point)),
        }

    def remove_joint(self, joint_name: str) -> None:
        """
        Removes a joint from the robot.

        NOTE: self.remove(joint) is not working.
        Refer to https://github.com/hauptmech/odio_urdf/tree/master for proper implementation.
        """
        for i, elt in enumerate(self):
            if isinstance(elt, Joint) and elt.name == joint_name:
                del self[i]
                return

    def _ensure_base_helper(self):
        """Ensure that a base_helper link exists and properly connect previously base-connected links."""
        # Find the existing joint connecting to base
        base_joint = self.get_joint_for_parent("base")
        if not any(link.name == "base_helper" for link in self.get_links().values()) and base_joint:
            # Create base_helper link
            base_helper = Link(name="base_helper")
            self.add_link(base_helper)

            # Collect all joints that have 'base' as parent
            joints_to_update = []
            for joint in self.get_joints().values():
                if joint.get_named_elements("Parent")[0].link == "base":
                    joints_to_update.append(joint)

            # Create the base_to_base_helper joint
            base_to_helper_joint = Joint(
                "base_to_base_helper",
                Parent("base"),
                Child("base_helper"),
                type="fixed",
            )

            # Copy properties from the existing base joint
            # to `base_to_base_helper` joint
            for elem in base_joint:
                if not isinstance(elem, (Parent, Child)):
                    base_to_helper_joint.append(elem)

            self.add_joint(base_to_helper_joint)

            # Create new joints for all children of 'base'
            for joint in joints_to_update:
                child_link = joint.get_named_elements("Child")[0].link

                new_joint = Joint(
                    f"base_helper_to_{child_link}",
                    Parent("base_helper"),
                    Child(child_link),
                    type=joint.type
                )

                # Copy all other properties from the original joint
                for elem in joint:
                    if not isinstance(elem, (Parent, Child)):
                        new_joint.append(elem)

                self.add_joint(new_joint)

                # Remove the old joint
                self.remove_joint(joint.name)

    def make_prismatic_joint(
        self,
        child_link_name: str,
        parent_link_name: str,
        global_lower_point: List[float],
        global_upper_point: List[float],
        force_overwrite: bool = True,
    ) -> None:
        """
        Creates or updates a prismatic joint between the specified child and parent links.

        Args:
            child_link_name (str): Name of the child link.
            parent_link_name (str): Name of the parent link.
            global_lower_point (list): The global coordinates of the lower point.
            global_upper_point (list): The global coordinates of the upper point.
            force_overwrite (bool): If True, overwrite existing joint.
        """

        if parent_link_name == "base":
            self._ensure_base_helper()
            parent_link_name = "base_helper"

        joint = self.get_joint_between(child_link_name, parent_link_name)
        if joint is not None and not force_overwrite:
            return  # Joint already exists

        parent_state = self.get_link_states([parent_link_name])[
            parent_link_name]
        parent_pos, parent_orientation = parent_state[0], parent_state[1]
        prismatic_joint = self.compute_prismatic_joint(
            global_lower_point, global_upper_point, parent_orientation
        )

        if joint is not None:
            current_origin = self.get_origin_of_joint(joint.name)
            current_xyz = [float(x) for x in current_origin.xyz.split()]
            prismatic_origin = [float(x)
                                for x in prismatic_joint["origin"].xyz.split()]
            new_origin = [current_xyz[i] + prismatic_origin[i]
                          for i in range(3)]
            prismatic_joint["origin"] = Origin(
                xyz=" ".join(map(str, new_origin)))
            self.remove_joint(joint.name)

        self.add_joint(
            Joint(
                f"{parent_link_name}_to_{child_link_name}",
                Parent(parent_link_name),
                Child(child_link_name),
                type="prismatic",
                *prismatic_joint.values(),
            )
        )

    def make_revolute_joint(
        self,
        child_link_name: str,
        parent_link_name: str,
        global_axis: List[float],
        lower_angle_deg: float,
        upper_angle_deg: float,
        force_overwrite: bool = True,
        pivot_point: Optional[List[float]] = None,
    ) -> None:
        """
        Creates or updates a revolute joint between the specified child and parent links.

        Args:
            child_link_name (str): Name of the child link.
            parent_link_name (str): Name of the parent link.
            global_axis (list): The rotation axis of the joint in the world frame.
            lower_angle_deg (float): The lower joint angle limit in degrees.
            upper_angle_deg (float): The upper joint angle limit in degrees.
            force_overwrite (bool, optional): If True, overwrite existing joint.
            pivot_point (list, optional): The pivot point for the rotation in the world frame.
        """
        if parent_link_name == "base":
            self._ensure_base_helper()
            parent_link_name = "base_helper"

        joint = self.get_joint_between(child_link_name, parent_link_name)
        if joint is not None and not force_overwrite:
            return  # Joint already exists

        parent_state = self.get_link_states([parent_link_name])[
            parent_link_name]
        parent_pos, parent_orientation = np.array(
            parent_state[0]), parent_state[1]

        local_axis = world_to_local(
            np.array(global_axis), parent_orientation).tolist()

        lower_limit = np.radians(lower_angle_deg)
        upper_limit = np.radians(upper_angle_deg)

        if pivot_point is None:
            local_pivot_point = [0, 0, 0]
            if joint is not None:
                current_origin = self.get_origin_of_joint(joint.name)
                local_pivot_point = [float(x)
                                     for x in current_origin.xyz.split()]
        else:
            local_pivot_point = world_to_local(
                np.array(pivot_point), parent_orientation, parent_pos
            )

        local_pivot_point = np.array(local_pivot_point)

        revolute_joint = {
            "axis": Axis(xyz=" ".join(map(str, local_axis))),
            "limit": Limit(
                lower=lower_limit, upper=upper_limit, velocity="5", effort="5"
            ),
            "origin": Origin(xyz=" ".join(map(str, local_pivot_point.tolist()))),
        }

        if pivot_point is not None and joint is not None:
            joint_origin = self.get_origin_of_joint(joint.name)
            joint_origin_xyz = [float(x) for x in joint_origin.xyz.split()]
            self.translate_link(
                child_link_name, joint_origin_xyz - local_pivot_point)

        if joint is not None:
            self.remove_joint(joint.name)

        self.add_joint(
            Joint(
                f"{parent_link_name}_to_{child_link_name}",
                Parent(parent_link_name),
                Child(child_link_name),
                type="revolute",
                *revolute_joint.values(),
            )
        )

    def compute_push_direction(self, button_link_name):
        """
        Compute the push direction by computing the normal vector to the largest face
        of the oriented bounding box of the button mesh
        """
        # Get the link
        links = self.get_links()
        if button_link_name not in links:
            raise ValueError(
                f"Link '{button_link_name}' not found in the robot.")

        link = links[button_link_name]

        # Get the mesh filename from the link's visual element
        visuals = link.get_named_elements("Visual")
        if not visuals:
            raise ValueError(
                f"No visual element found for link '{button_link_name}'")

        geometry = visuals[0].get_named_elements("Geometry")[0]
        mesh = geometry.get_named_elements("Mesh")[0]
        mesh_filename = mesh.filename

        # Construct the full path to the mesh file
        mesh_file_path = join_path(self.input_dir, mesh_filename)

        # Load and rotate the mesh
        rpy = [1.5707963267948966, 2.220446049250313e-16, 1.5707963267948961]
        mesh = load_and_rotate_mesh(mesh_file_path, rpy)

        # Compute the oriented bounding box and largest face
        obb, largest_face = compute_obb_and_largest_face(mesh)

        # Compute the normal vector
        normal_vector = compute_normal_vector(obb.vertices, largest_face)

        return normal_vector

    def compute_topological_order(self) -> List[str]:
        links = self.get_links()
        joints = self.get_joints()

        # Build adjacency list and in-degree count for each link
        adj_list = defaultdict(list)
        in_degree = defaultdict(int)

        for joint in joints.values():
            parent_link = joint.get_named_elements("Parent")[0].link
            child_link = joint.get_named_elements("Child")[0].link
            adj_list[parent_link].append(child_link)
            in_degree[child_link] += 1

        # Initialize the queue with links having no incoming edges (in-degree 0)
        queue = deque([link for link in links if in_degree[link] == 0])
        topological_order = []

        while queue:
            current_link = queue.popleft()
            topological_order.append(current_link)

            for neighbor in adj_list[current_link]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if topological sorting is possible (i.e., no cycles)
        if len(topological_order) != len(links):
            raise ValueError("The robot's kinematic chain contains a cycle.")

        return topological_order

    def get_pose(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Computes the position and orientation of all links in the robot.

        Returns:
            dict: A dictionary with link names as keys and their positions and orientations as values.
                The position is a 3x1 numpy array and the orientation is a 3x3 numpy array.
        """
        links = self.get_links()
        joints = self.get_joints()

        # Dictionary to store positions and orientations
        poses = {}

        # Initialize the position and orientation of the base link
        base_link_name = "base"
        if base_link_name in links:
            poses[base_link_name] = {
                "position": np.array([0.0, 0.0, 0.0]),
                "orientation": np.eye(3),
                "bbox_position": np.array([0.0, 0.0, 0.0]),
            }

        # Get topological order of the joints
        topological_order = self.compute_topological_order()

        bbox_positions = self.get_bounding_boxes(include_dim=False)
        # Traverse the kinematic chain in topological order
        for link_name in topological_order:
            if link_name == base_link_name:
                continue

            joint = next(
                joint
                for joint in joints.values()
                if joint.get_named_elements("Child")[0].link == link_name
            )
            child_link_name = joint.get_named_elements("Child")[0].link
            parent_link_name = joint.get_named_elements("Parent")[0].link
            joint_origin = self.get_origin_of_joint(joint.name)

            # Ensure parent link has been processed
            if parent_link_name not in poses:
                raise ValueError(
                    f"Parent link '{parent_link_name}' must be processed before child link '{child_link_name}'."
                )

            # Compute the joint transformation
            joint_position = np.array([float(x)
                                      for x in joint_origin.xyz.split()])
            joint_rpy = [float(x) for x in joint_origin.rpy.split()]
            joint_orientation = R.from_euler("xyz", joint_rpy).as_matrix()

            # Get the parent's position and orientation
            parent_position = poses[parent_link_name]["position"]
            parent_orientation = poses[parent_link_name]["orientation"]

            # Compute the position and orientation of the child link at the joint
            child_position_at_joint = (
                parent_orientation @ joint_position + parent_position
            )
            child_orientation_at_joint = parent_orientation @ joint_orientation

            # Get the origin of the child's visual element
            child_visual_origin = self.get_origin_of_link(child_link_name)

            # Compute the visual origin transformation
            visual_position = np.array(
                [float(x) for x in child_visual_origin.xyz.split()]
            )
            visual_rpy = [float(x) for x in child_visual_origin.rpy.split()]
            visual_orientation = R.from_euler("xyz", visual_rpy).as_matrix()

            # Compute the final position and orientation of the child link
            final_child_position = (
                child_orientation_at_joint @ visual_position + child_position_at_joint
            )
            final_child_orientation = child_orientation_at_joint @ visual_orientation

            bbox_position = bbox_positions[link_name]
            bbox_position = get_bounding_box_center(
                bbox_position[0], bbox_position[1])

            # Update the child's position and orientation in the poses dictionary
            poses[child_link_name] = {
                "position": final_child_position,
                "orientation": final_child_orientation,
                "bbox_position": bbox_position,
            }

        return poses

    def get_joint_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Saves the state of manipulatable joints (revolute, prismatic, continuous) in the robot.

        Returns:
            dict: A dictionary where each key is a joint name and each value is a dictionary
                with joint type, origin, axis, and limits in the global frame.
        """
        joint_states = {}
        default_origin = {"xyz": "0 0 0", "rpy": "0 0 0"}
        default_axis = "1 0 0"
        default_limit = {"lower": 0, "upper": 0, "effort": 0, "velocity": 0}

        # Compute link poses (already done in get_pose)
        link_poses = self.get_pose()

        # Get topological order of the joints
        topological_order = self.compute_topological_order()

        for link_name in topological_order:
            if link_name == "base":  # Skip the base link
                continue

            # Find the joint associated with the link
            joint = next(
                joint
                for joint in self.get_joints().values()
                if joint.get_named_elements("Child")[0].link == link_name
            )

            if joint.type in ["revolute", "prismatic", "continuous"]:
                # Get joint information from URDF
                joint_origin_elements = joint.get_named_elements("Origin")
                joint_origin = (
                    joint_origin_elements[0]
                    if joint_origin_elements
                    else Origin(**default_origin)
                )

                joint_axis_elements = joint.get_named_elements("Axis")
                joint_axis = (
                    getattr(joint_axis_elements[0], "xyz", default_axis)
                    if joint_axis_elements
                    else default_axis
                )

                joint_limit_elements = joint.get_named_elements("Limit")
                joint_limit = (
                    joint_limit_elements[0]
                    if joint_limit_elements
                    else Limit(**default_limit)
                )

                # Get link names
                child_link_name = joint.get_named_elements("Child")[0].link
                parent_link_name = joint.get_named_elements("Parent")[0].link

                # Transformation logic
                parent_position = link_poses[parent_link_name]["position"]
                parent_orientation = link_poses[parent_link_name]["orientation"]

                # Local joint origin and orientation
                joint_position_local = np.array(
                    [float(x) for x in joint_origin.xyz.split()]
                )

                # Get the origin using the updated get_origin_of_joint
                origin = self.get_origin_of_joint(joint.name)

                # Create rotation matrix from RPY values
                joint_orientation_local = R.from_euler(
                    "xyz", [float(x) for x in origin.rpy.split()]
                ).as_matrix()

                # Transform to global frame
                joint_position = (
                    parent_orientation @ joint_position_local + parent_position
                )
                joint_orientation = parent_orientation @ joint_orientation_local

                # Convert axis from local to global frame
                joint_axis_vector = np.array(
                    [float(x) for x in joint_axis.split()])
                joint_axis_global = joint_orientation @ joint_axis_vector

                # Store the joint state in the dictionary
                joint_states[joint.name] = {
                    "joint_type": joint.type,
                    "joint_origin": {
                        "xyz": " ".join(map(str, joint_position)),
                        # "rpy": ' '.join(map(str, R.from_matrix(joint_orientation).as_euler('xyz'))),
                        "orientation": " ".join(
                            map(str, R.from_matrix(joint_orientation).as_quat())
                        ),
                    },
                    "joint_axis": " ".join(map(str, joint_axis_global)),
                    "joint_limit": {
                        "lower": getattr(joint_limit, "lower", default_limit["lower"]),
                        "upper": getattr(joint_limit, "upper", default_limit["upper"]),
                        "effort": getattr(
                            joint_limit, "effort", default_limit["effort"]
                        ),
                        "velocity": getattr(
                            joint_limit, "velocity", default_limit["velocity"]
                        ),
                    },
                }

        return joint_states

    def get_link_visual_meshes(self, link_name: str) -> List[str]:
        """
        Retrieves all visual mesh filenames associated with a given link.

        Args:
            link_name (str): The name of the link to retrieve meshes for.

        Returns:
            List[str]: A list of mesh filenames associated with the link's visual elements.

        Raises:
            ValueError: If the specified link is not found in the robot.
        """
        links = self.get_links()
        if link_name not in links:
            raise ValueError(f"Link '{link_name}' not found in the robot.")

        link = links[link_name]
        mesh_filenames = []

        for visual in link.get_named_elements("Visual"):
            geometry = visual.get_named_elements("Geometry")
            if geometry:
                mesh = geometry[0].get_named_elements("Mesh")
                if mesh and "filename" in mesh[0].attributes:
                    mesh_filenames.append(mesh[0].filename)

        return mesh_filenames


class Joint(Element):
    required_elements = ["Parent", "Child"]
    allowed_elements = [
        "Origin",
        "Inertial",
        "Visual",
        "Collision",
        "Axis",
        "Calibration",
        "Dynamics",
        "Limit",
        "Mimic",
        "Safety_controller",
    ]
    required_attributes = ["name", "type"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "type" not in kwargs:
            kwargs["type"] = "revolute"

        Joint_types = [
            "revolute",
            "continuous",
            "prismatic",
            "fixed",
            "floating",
            "planar",
        ]
        if kwargs["type"] not in Joint_types:
            raise Exception("Joint type not correct")

        super(Joint, self).__init__(*args, **kwargs)


class Link(NamedElement):
    allowed_elements = [
        "Inertial",
        "Visual",
        "Collision",
        "Self_collision_checking",
        "Contact",
    ]


class Transmission(NamedElement):
    allowed_elements = ["Type", "Transjoint", "Actuator"]


class Type(Element):
    pass


class Transjoint(NamedElement):
    allowed_elements = ["Hardwareinterface"]
    element_name = "joint"


class Hardwareinterface(Element):
    pass


class Mechanicalreduction(Element):
    pass


class Actuator(NamedElement):
    allowed_elements = ["Mechanicalreduction", "Hardwareinterface"]


class Parent(Element):
    required_attributes = ["link"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """If Link type passed in, extract name string"""
        args = [arg for arg in args]
        for i, arg in enumerate(args):
            if isinstance(arg, Link):
                args[i] = arg.name

        super(Parent, self).__init__(*args, **kwargs)


class Child(Parent):
    required_attributes = ["link"]


class Inertia(Element):
    allowed_attributes = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and isinstance(args[0], list):
            if len(args[0]) == 6:
                kwargs["ixx"] = str(args[0][0])
                kwargs["ixy"] = str(args[0][1])
                kwargs["ixz"] = str(args[0][2])
                kwargs["iyy"] = str(args[0][3])
                kwargs["iyz"] = str(args[0][4])
                kwargs["izz"] = str(args[0][5])
                del args[0]
        super(Inertia, self).__init__(*args, **kwargs)


class Visual(Element):
    allowed_elements = ["Origin", "Geometry", "Material"]


class Geometry(Element):
    allowed_elements = ["Box", "Cylinder", "Sphere", "Mesh", "Capsule"]
    allowed_attributes = ["name"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) != 1:
            raise Exception("Can only have one shape!")
        super(Geometry, self).__init__(*args, **kwargs)


class Box(Element):
    allowed_attributes = ["size"]


class Capsule(Element):
    allowed_attributes = ["radius", "length"]


class Cylinder(Element):
    allowed_attributes = ["radius", "length"]


class Sphere(Element):
    allowed_attributes = ["radius"]


class Mesh(Element):
    allowed_attributes = ["filename", "scale"]


class Material(Element):
    allowed_elements = ["Color", "Texture"]
    allowed_attributes = ["name"]


class Color(Element):
    allowed_attributes = ["rgba"]


class Texture(Element):
    allowed_attributes = ["filename"]


class Collision(Element):
    allowed_elements = ["Origin", "Geometry", "Material"]
    allowed_attributes = ["name"]


class Self_collision_checking(Element):
    allowed_elements = ["Origin", "Geometry"]
    allowed_attributes = ["name"]


class Mass(Element):
    allowed_attributes = ["value"]


class Origin(Element):
    allowed_attributes = ["xyz", "rpy"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) > 0 and isinstance(args[0], list):
            if len(args[0]) == 6:
                kwargs["xyz"] = (
                    str(args[0][0]) + " " +
                    str(args[0][1]) + " " + str(args[0][2])
                )
                kwargs["rpy"] = (
                    str(args[0][3]) + " " +
                    str(args[0][4]) + " " + str(args[0][5])
                )
                del args[0]
        super(Origin, self).__init__(*args, **kwargs)


class Axis(Element):
    allowed_attributes = ["xyz"]


class Calibration(Element):
    allowed_attributes = ["rising", "falling"]


class Safety_controller(Element):
    allowed_attributes = [
        "soft_lower_limit",
        "soft_upper_limit",
        "k_position",
        "k_velocity",
    ]


class Limit(Element):
    required_attributes = ["effort", "velocity"]
    allowed_attributes = ["lower", "upper"]


class Dynamics(Element):
    allowed_attributes = ["damping", "friction"]


class Mimic(Element):
    allowed_attributes = ["joint", "multiplier", "offset"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "joint" not in kwargs:
            raise Exception('Mimic must have "joint" attribute')
        super(Mimic, self).__init__(*args, **kwargs)


class Inertial(Element):
    allowed_elements = ["Origin", "Mass", "Inertia"]


class Gazebo(Element):
    allowed_elements = [
        "Material",
        "Gravity",
        "Dampingfactor",
        "Maxvel",
        "Mindepth",
        "Mu1",
        "Mu2",
        "Fdir1",
        "Kp",
        "Kd",
        "Selfcollide",
        "Maxcontacts",
        "Laserretro",
        "Plugin",
    ]
    allowed_attributes = ["reference", "xmltext"]


class Plugin(Element):
    allowed_elements = ["Robotnamespace"]
    allowed_attributes = ["name", "filename"]


class Robotnamespace(Element):
    pass


class Gravity(Element):
    pass


class Laserretro(Element):
    pass


class Maxcontacts(Element):
    pass


class Selfcollide(Element):
    pass


class Kd(Element):
    pass


class Kp(Element):
    pass


class Fdir1(Element):
    pass


class Mu2(Element):
    pass


class Dampingfactor(Element):
    pass


class Maxvel(Element):
    pass


class Mindepth(Element):
    pass


class Mu1(Element):
    pass


class Contact(Element):
    """Bullet3 element."""

    allowed_elements = ["Stiffness", "Damping", "Lateral_Friction"]


class Stiffness(Element):
    """Bullet3 element."""

    allowed_attributes = ["value"]


class Damping(Element):
    """Bullet3 element."""

    allowed_attributes = ["value"]


class Lateral_Friction(Element):
    """Bullet3 element."""

    allowed_attributes = ["value"]


def create_urdf_tree(joints_data):
    tree = defaultdict(list)
    child_links = set()
    all_links = set()

    # First pass: build the tree structure and collect information
    for joint_info in joints_data:
        joint_type, joint_name, parent_link, child_link = joint_info[:4]
        semantic_joint = joint_info[4] if len(joint_info) > 4 else None

        tree[parent_link].append(
            (child_link, joint_type, joint_name, semantic_joint))

        child_links.add(child_link)
        all_links.add(parent_link)
        all_links.add(child_link)

    # Determine the root
    root_candidates = all_links - child_links
    if len(root_candidates) == 1:
        root = root_candidates.pop()
    elif len(root_candidates) == 0:
        # If no clear root, choose the link that appears first as a parent
        root = joints_data[0][2]  # parent_link of the first joint
    else:
        # Multiple root candidates, choose 'base' if it's one of them, otherwise take the first
        root = "base" if "base" in root_candidates else root_candidates.pop()

    # Ensure all links are in the tree, even if they have no children
    for link in all_links:
        if link not in tree:
            tree[link] = []

    return root, dict(tree)


def print_urdf_tree(root, tree, indent=""):
    print(f"{indent}{root}")
    for child, joint_type, joint_name, semantic_joint in tree[root]:
        joint_info = f"({joint_type}, {joint_name}"
        if semantic_joint:
            joint_info += f", {semantic_joint}"
        joint_info += ")"
        print(f"{indent}  ├─{joint_info}→ ", end="")
        print_urdf_tree(child, tree, indent + "  │ ")


def get_semantic_name_of_link(link_name, obj_id, 
                              dataset_dir="datasets/partnet-mobility-v0/dataset/"):
    input_dir = join_path(dataset_dir, obj_id)
    semantic_file = join_path(input_dir, "semantics.txt")
    link_semantics = load_semantic(semantic_file)
    return link_semantics.get(link_name, link_name)


def extract_joint_data_and_stats(
    obj_ids=None, dataset_dir="datasets/partnet-mobility-v0/dataset"
):
    if obj_ids is None:
        obj_ids = get_obj_ids(dataset_dir)
    joint_data = {}  # Dictionary to store counts and joint IDs per type
    
    for obj_id in obj_ids:
        file_path = join_path(dataset_dir, obj_id, "mobility.urdf")
        joints_data = extract_joint_data(file_path)
        obj_joint_data = {}
        joint_type_json = {}
        
        # First pass: Just store the basic joint information
        for joint_type, joint_id, p_link, c_link in joints_data:
            if joint_type not in joint_data:
                joint_data[joint_type] = {"count": 0, "obj_ids": []}
            joint_data[joint_type]["count"] += 1
            
            # Generate semantic joint ID directly without recursive calls
            p_semantic = get_semantic_name_of_link(p_link, obj_id, dataset_dir=dataset_dir)
            c_semantic = get_semantic_name_of_link(c_link, obj_id, dataset_dir=dataset_dir)
            semantic_joint_id = f"{p_semantic}_to_{c_semantic}"
            
            joint_data[joint_type]["obj_ids"].append(
                (obj_id, joint_id, semantic_joint_id)
            )
            obj_joint_data[joint_id] = semantic_joint_id
            joint_type_json[joint_id] = joint_type

        # Save the generated data
        save_json(
            obj_joint_data, 
            join_path(dataset_dir, obj_id, "joint_semantics.json")
        )
        save_json(
            joint_type_json, 
            join_path(dataset_dir, obj_id, "joint_type.json")
        )
    
    save_pickle(joint_data, "joint_data.pkl")
    return joint_data

def get_joint_semantic(obj_dir):
    """Get joint semantics, generating them if necessary."""
    semantics_path = join_path(obj_dir, "joint_semantics.json")
    if not os.path.exists(semantics_path):
        # Extract without using semantic IDs to avoid recursion
        obj_id = obj_dir.split("/")[-1]
        file_path = join_path(obj_dir, "mobility.urdf")
        joints_data = extract_joint_data(file_path)
        
        # Generate semantic IDs directly
        obj_joint_data = {}
        for joint_type, joint_id, p_link, c_link in joints_data:
            p_semantic = get_semantic_name_of_link(p_link, obj_id, dataset_dir=os.path.dirname(obj_dir))
            c_semantic = get_semantic_name_of_link(c_link, obj_id, dataset_dir=os.path.dirname(obj_dir))
            obj_joint_data[joint_id] = f"{p_semantic}_to_{c_semantic}"
        
        save_json(obj_joint_data, semantics_path)
        return obj_joint_data
    
    return load_json(semantics_path)



def extract_joint_data(file_path, include_semantic_names=False):
    """Extract joint types and IDs from a URDF file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        joints_data = []
        for joint in root.findall("joint"):
            joint_type = joint.attrib["type"]
            joint_name = joint.attrib["name"]
            parent_link = joint.find("parent").attrib["link"]
            child_link = joint.find("child").attrib["link"]

            if include_semantic_names:
                obj_id = os.path.dirname(file_path).split("/")[-1]
                p_link = get_semantic_name_of_link(parent_link, obj_id)
                c_link = get_semantic_name_of_link(child_link, obj_id)
                semantic_joint = f"{p_link}_to_{c_link}"
                joints_data.append(
                    (joint_type, joint_name, parent_link, child_link, semantic_joint)
                )
            else:
                joints_data.append(
                    (joint_type, joint_name, parent_link, child_link))
        return joints_data

    except FileNotFoundError:
        print(f"Warning: URDF file not found: {file_path}")
        return []  # Return empty list if file not found
    except ET.ParseError:
        print(f"Warning: Error parsing URDF file: {file_path}")
        return []


def find_non_helper_parent(tree_dict, link):
    if "helper" not in link.lower():
        return link
    for parent, children in tree_dict.items():
        for child, _, _, _ in children:
            if child == link:
                return find_non_helper_parent(tree_dict, parent)
    return link  # Return original link if no non-helper parent found


def find_non_helper_child(tree_dict, link):
    if "helper" not in link.lower():
        return link
    for child, _, _, _ in tree_dict.get(link, []):
        result = find_non_helper_child(tree_dict, child)
        if result:
            return result
    return None


def get_semantic_joint_id(
    obj_id,
    joint_id,
    file_path=None,
    dataset_dir="datasets/partnet-mobility-v0/dataset",
    ignore_helper=True,
    cached=True,
):
    if file_path is None:
        file_path = join_path(dataset_dir, obj_id, "mobility.urdf")

    if cached and os.path.exists(join_path(dataset_dir, obj_id, "joint_semantics.json")):
        joint_semantics = load_json(
            join_path(dataset_dir, obj_id, "joint_semantics.json")
        )
        return joint_semantics.get(joint_id, None)

    joints_data = extract_joint_data(file_path)
    return get_semantic_joint_id_core(
        obj_id, joint_id, joints_data, ignore_helper=ignore_helper,
        dataset_dir=dataset_dir,
    )


def get_joint_id(
    obj_id,
    semantic_joint,
    file_path=None,
    input_dir="datasets/partnet-mobility-v0/dataset",
    cached=True,
):
    if file_path is None:
        file_path = join_path(input_dir, obj_id, "mobility.urdf")

    # Try to use cached data if available
    if cached:
        cache_file = join_path(input_dir, obj_id, "joint_semantics.json")
        if os.path.exists(cache_file):
            joint_semantics = load_json(cache_file)
            for joint_id, sem_joint in joint_semantics.items():
                if sem_joint == semantic_joint:
                    return joint_id

    # If not found in cache or cache not used, parse the URDF file
    joints_data = extract_joint_data(file_path)
    # joints_data = get_joint_semantic(join_path(input_dir, obj_id))

    # Reconstruct semantic joint names and find a match
    for joint_type, j_id, p_link, c_link in joints_data:
        reconstructed_semantic = f"{p_link}_to_{c_link}"
        if reconstructed_semantic == semantic_joint:
            return joint_id

    # If no matching joint is found
    return None


def get_semantic_joint_from_child_name(
    obj_id,
    child_part_name,
    file_path=None,
    dataset_dir="datasets/partnet-mobility-v0/dataset",
    ignore_helper=True,
    cached=True,
):
    if file_path is None:
        file_path = join_path(dataset_dir, obj_id, "mobility.urdf")

    if cached and os.path.exists(join_path(dataset_dir, obj_id, "joint_semantics.json")):
        joint_semantics = load_json(
            join_path(dataset_dir, obj_id, "joint_semantics.json")
        )
        for joint_id, semantic_joint in joint_semantics.items():
            if semantic_joint.endswith(f"to_{child_part_name}"):
                return semantic_joint

    joints_data = get_joint_semantic(join_path(dataset_dir, obj_id))
    for joint_id, semantic_joint in joints_data.items():
        if semantic_joint.endswith(f"to_{child_part_name}"):
            return semantic_joint

    return None  # Return None if no matching joint is found


def get_semantic_joint_id_core(obj_id, joint_id, joints_data, ignore_helper=False,
                               dataset_dir="datasets/partnet-mobility-v0/dataset"):
    if ignore_helper:
        _, tree_dict = create_urdf_tree(joints_data)

    for joint_type, j_id, p_link, c_link in joints_data:
        if j_id == joint_id:
            if ignore_helper:
                p_link = find_non_helper_parent(tree_dict, p_link)
                c_link = find_non_helper_child(tree_dict, c_link)

            if p_link and c_link:
                p_link = get_semantic_name_of_link(p_link, obj_id, dataset_dir=dataset_dir)
                c_link = get_semantic_name_of_link(c_link, obj_id, dataset_dir=dataset_dir)
                return f"{p_link}_to_{c_link}"

    return None


def process_urdf(
    input_file,
    format_rviz=False,
    ignore_helper=True,
    dataset_dir="datasets/partnet-mobility-v0/dataset",
    use_semantic_name=False,
):
    """
    Adds material tags to all visual elements in a URDF file and processes joints with limits.

    Args:
        input_file (str): Path to the original URDF file.
    """
    input_dir = os.path.dirname(input_file)
    tree = ET.parse(input_file)
    root = tree.getroot()
    processed_links = []
    processed_joints = []

    semantic_file = join_path(input_dir, "semantics.txt")
    if use_semantic_name:
        link_semantics = load_semantic(semantic_file)

    joint_semantics = {}

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        visuals = []
        collisions = []

        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            mesh_filename = mesh.attrib["filename"]
            origin_element = visual.find("origin")
            if origin_element is None:
                origin_xyz = [0, 0, 0]
                origin_rpy = [0, 0, 0]
            else:
                origin_xyz = origin_element.attrib.get("xyz", "0 0 0")
                origin_rpy = origin_element.attrib.get("rpy", "0 0 0")
                origin_xyz = list(map(float, origin_xyz.split()))
                origin_rpy = list(map(float, origin_rpy.split()))

            if format_rviz:
                my_pkg_name = "my_robot_pkg"
                prefix = join_path(
                    "package://", my_pkg_name, "urdf", input_dir)
                if not mesh_filename.startswith(prefix):
                    mesh_filename = join_path(prefix, mesh_filename)

            visuals.append(
                Visual(
                    Geometry(Mesh(filename=mesh_filename)),
                    Origin(xyz=origin_xyz, rpy=origin_rpy),
                )
            )

        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh")
            mesh_filename = mesh.attrib["filename"]

            origin_element = collision.find("origin")
            if origin_element is None:
                origin_xyz = [0, 0, 0]
                origin_rpy = [0, 0, 0]
            else:
                origin_xyz = origin_element.attrib.get("xyz", "0 0 0")
                origin_rpy = origin_element.attrib.get("rpy", "0 0 0")
                origin_xyz = list(map(float, origin_xyz.split()))
                origin_rpy = list(map(float, origin_rpy.split()))

            collisions.append(
                Collision(
                    Geometry(Mesh(filename=mesh_filename)),
                    Origin(xyz=origin_xyz, rpy=origin_rpy),
                )
            )

        if use_semantic_name:
            semantic_link_name = f"{link_semantics.get(link_name, link_name)}"
        else:
            semantic_link_name = link_name  # No semantic change, keep original name
        processed_links.append(
            Link(name=semantic_link_name, *visuals, *collisions))

    for joint in root.findall("joint"):
        joint_name = joint.attrib["name"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]

        obj_id = os.path.dirname(input_file).split("/")[-1]

        if use_semantic_name:
            semantic_joint = get_semantic_joint_id(
                obj_id, joint_name, ignore_helper=ignore_helper, dataset_dir=dataset_dir
            )
            joint_semantics[joint_name] = semantic_joint
        joint_type = joint.attrib["type"]

        origin_element = joint.find("origin")
        if origin_element is None:
            origin_xyz = [0, 0, 0]
            origin_rpy = [0, 0, 0]
        else:
            origin_xyz = origin_element.attrib.get("xyz", "0 0 0")
            origin_rpy = origin_element.attrib.get("rpy", "0 0 0")
            origin_xyz = list(map(float, origin_xyz.split()))
            origin_rpy = list(map(float, origin_rpy.split()))

        joint_axis = joint.find("axis")
        axis = [1, 0, 0]  # Default axis
        if joint_axis is not None:
            j_axis = joint_axis.attrib.get("xyz", "1 0 0")
            j_axis = j_axis.replace("None", "0")
            if j_axis is not None:
                axis = list(map(float, j_axis.split()))

        limit_element = joint.find("limit")
        if limit_element is not None:
            lower = float(limit_element.attrib.get("lower", "0"))
            upper = float(limit_element.attrib.get("upper", "0"))
            effort = float(limit_element.attrib.get("effort", "0"))
            velocity = float(limit_element.attrib.get("velocity", "0"))
            limit = Limit(lower=lower, upper=upper,
                          effort=effort, velocity=velocity)
        else:
            limit = None

        if use_semantic_name:
            joint_name = semantic_joint
        if limit is not None:
            processed_joints.append(
                Joint(
                    joint_name,
                    Parent(parent),
                    Child(child),
                    Origin(xyz=origin_xyz, rpy=origin_rpy),
                    Axis(axis),
                    Limit(lower=lower, upper=upper,
                          effort=effort, velocity=velocity),
                    type=joint_type,
                )
            )
        else:
            processed_joints.append(
                Joint(
                    joint_name,
                    Parent(parent),
                    Child(child),
                    Origin(xyz=origin_xyz, rpy=origin_rpy),
                    Axis(axis),
                    type=joint_type,
                )
            )

    robot = Robot(input_dir, *processed_links, *processed_joints)
    return robot


def save_link_states(
    urdf_file_path: str,
    out_file_path: Optional[str] = None,
    dataset_dir="datasets/partnet-mobility-v0/dataset/",
) -> None:
    """
    Extracts pose (position and orientation) data from a URDF file and saves it as JSON.

    Args:
        urdf_file_path (str): Path to the input URDF file.
        out_file_path (str): Path to the output JSON file.
    """
    if out_file_path is None:
        out_file_path = join_path(os.path.dirname(
            urdf_file_path), "link_states.json")

    robot = process_urdf(urdf_file_path, dataset_dir=dataset_dir)
    link_data = {}

    poses = robot.get_pose()

    for link_name, pose in poses.items():
        if link_name == "base":  # Exclude the base link if it exists
            continue

        position = pose["position"]
        rotation_matrix = pose["orientation"]
        bbox_position = pose["bbox_position"]

        # Convert the rotation matrix to a quaternion
        quaternion = R.from_matrix(
            rotation_matrix
        ).as_quat()  # Quaternion in the format [x, y, z, w]

        link_data[link_name] = {
            "position": position.tolist(),  # Convert NumPy array to list for JSON serialization
            # Store quaternion [x, y, z, w]
            "orientation": quaternion.tolist(),
            "bbox_position": bbox_position,
        }

    save_json(link_data, out_file_path)
    return load_json(out_file_path)


def link_data_to_np(link_data):
    """Loads link positions and orientations from a JSON file."""
    # Convert lists back to NumPy arrays
    for link_name, data in link_data.items():
        link_data[link_name]["position"] = np.array(data["position"])
        link_data[link_name]["orientation"] = np.array(data["orientation"])
        link_data[link_name]["bbox_position"] = np.array(data["bbox_position"])
    return link_data


def load_and_rotate_mesh(file_path, rpy):
    loaded_obj = trimesh.load(file_path)
    rotation = R.from_euler("xyz", rpy)
    rotation_matrix = rotation.as_matrix()

    if isinstance(loaded_obj, trimesh.Scene):
        # If it's a scene, we need to rotate each mesh in the scene
        for geometry in loaded_obj.geometry.values():
            if hasattr(geometry, "vertices"):
                geometry.vertices = np.dot(
                    geometry.vertices, rotation_matrix.T)
        # Return the first mesh in the scene, or combine all meshes
        # Depending on your needs, you might want to adjust this
        return list(loaded_obj.geometry.values())[0]
    elif isinstance(loaded_obj, trimesh.Trimesh):
        # If it's a single mesh, rotate it directly
        loaded_obj.vertices = np.dot(loaded_obj.vertices, rotation_matrix.T)
        return loaded_obj
    else:
        raise ValueError(f"Unsupported object type: {type(loaded_obj)}")


def compute_obb_and_largest_face(mesh):
    obb = mesh.bounding_box_oriented
    obb_vertices = obb.vertices
    obb_faces_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    face_areas = [
        quad_area(
            obb_vertices[face[0]],
            obb_vertices[face[1]],
            obb_vertices[face[2]],
            obb_vertices[face[3]],
        )
        for face in obb_faces_indices
    ]
    largest_face_index = np.argmax(face_areas)
    return obb, obb_faces_indices[largest_face_index]


def quad_area(v0, v1, v2, v3):
    triangle1 = [v0, v1, v2]
    triangle2 = [v0, v2, v3]
    area1 = trimesh.triangles.area(np.array([triangle1]))
    area2 = trimesh.triangles.area(np.array([triangle2]))
    return area1 + area2


def compute_normal_vector(vertices, face):
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    normal_vector = np.cross(v1 - v0, v2 - v0)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    normal_vector = -np.abs(normal_vector)  # all entries negative

    # Renormalize the vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector


def compile_python_to_urdf(file_path, out_path=None, function_name=None, **kwargs):
    if out_path is None:
        out_path = join_path(os.path.dirname(file_path), "mobility.urdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    func = get_func_from_file(file_path, function_name=function_name)
    robot = func(**kwargs)
    string_to_file(str(robot), out_path)
    return out_path


def get_func_from_file(function_file, function_name=None):
    """
    Get a function from a Python file. When `function_name` is not provided, 
    the first function defined in the file will be returned.
    """
    module_name = f"dynamic_module_{hash(function_file)}"
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, function_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Add module to system modules
    spec.loader.exec_module(module)

    # Get all functions defined in the module
    functions = [obj for name, obj in inspect.getmembers(module, inspect.isfunction)
                 if obj.__module__ == module_name]

    if function_name is None:
        if not functions:
            raise ValueError(f"No functions found in file: {function_file}")
        # Return the first function defined in the file
        return functions[0]
    else:
        for func in functions:
            if func.__name__ == function_name:
                return func
        raise AttributeError(
            f"Function '{function_name}' not found in file: {function_file}")


def save_joint_states(
        urdf_file_path: str, out_file_path=None, dataset_dir="datasets/partnet-mobility-v0/dataset/"
):
    if out_file_path is None:
        out_file_path = join_path(os.path.dirname(
            urdf_file_path), "robot_joints.json")

    robot = process_urdf(urdf_file_path, dataset_dir=dataset_dir)
    states = robot.get_joint_states()
    save_json(states, out_file_path)
    return states


def load_urdf(file_path):
    """Loads the URDF file and returns the root element."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except FileNotFoundError:
        raise ValueError("URDF file not found at: {}".format(file_path))
    except ET.ParseError as e:
        raise ValueError("Error parsing URDF file: {}".format(e))


def load_semantic(
    semantic_file, dedup=True, reversed=False, lazy=False, anonymize_joint=True
):
    """
    Loads link semantics from a file, optionally deduplicating semantic labels and anonymizing joint types.
    Args:
        semantic_file (str): Path to the file containing link semantics.
        dedup (bool, optional): Whether to deduplicate semantic labels (default is True).
        reversed (bool, optional): Whether to reverse the key-value pairs in the output dictionary (default is False).
        lazy (bool, optional): Whether to load from a cached JSON file if available (default is False).
        anonymize_joint (bool, optional): Whether to remove joint type information from semantic labels (default is False).
    Returns:
        dict: A dictionary mapping link names to semantic labels.
    """
    link_sem_path = join_path(os.path.dirname(
        semantic_file), "link_semantics.json")
    if lazy and os.path.exists(link_sem_path):
        return load_json(link_sem_path)

    link_semantics = {}
    semantic_counts = {}  # Track the number of occurrences for each semantic label

    joint_types = ["translation", "rotation", "continuous"]

    with open(semantic_file, "r") as f:
        for line in f:
            link_name, joint_type, semantic_label = line.strip().split()

            if anonymize_joint:
                # Remove joint type information more comprehensively
                for jt in joint_types:
                    semantic_label = re.sub(
                        f"{jt}[_]?", "", semantic_label, flags=re.IGNORECASE
                    )
                semantic_label = semantic_label.strip(
                    "_"
                )  # Remove any trailing underscores

            if dedup:
                # Deduplicate the semantic label
                semantic_counts[semantic_label] = (
                    semantic_counts.get(semantic_label, 0) + 1
                )
                if semantic_counts[semantic_label] > 1:
                    semantic_label = (
                        f"{semantic_label}_{semantic_counts[semantic_label]}"
                    )

            if reversed:
                link_semantics[semantic_label] = link_name
            else:
                link_semantics[link_name] = semantic_label

    if not reversed:
        save_json(link_semantics, link_sem_path)
    return link_semantics


if __name__ == "__main__":
    pass
