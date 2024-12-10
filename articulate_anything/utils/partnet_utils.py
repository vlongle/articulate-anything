from articulate_anything.api.odio_urdf import (
    get_semantic_joint_id, 
    load_semantic, 
    get_joint_semantic,
    extract_joint_data_and_stats,
)
import pickle
import random
from collections import defaultdict
from articulate_anything.utils.utils import (
    load_json,
    join_path,
    save_json,
)
import os
import xml.etree.ElementTree as ET


def get_obj_ids(dataset_dir):
    obj_ids = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(join_path(dataset_dir, d))
    ]
    obj_ids.sort(key=int)  # Sort numerically
    return obj_ids


def extract_obj_type(input_dir):
    file_path = join_path(input_dir, "meta.json")
    j = load_json(file_path)
    return j["model_cat"]


def track_obj_types(dataset_dir="datasets/partnet-mobility-v0/dataset",
                    rename=False):
    """
    Return `obj_types` a dict with keys as object types and values as list of object ids
    of that type.
    """
    obj_types = defaultdict(list)
    obj_ids = get_obj_ids(dataset_dir)
    for obj_id in obj_ids:
        obj_input_dir = join_path(dataset_dir, obj_id)
        obj_type = extract_obj_type(obj_input_dir)
        obj_types[obj_type].append(obj_id)

    if rename:
        ## rename StorageFurniture to Cabinet
        obj_types["Cabinet"] = obj_types.pop("StorageFurniture")

    save_json(obj_types, "obj_types.json")
    return obj_types


def get_obj_type(obj_id):
    if not os.path.exists("obj_types.json"):
        obj_types = track_obj_types()
    obj_types = load_json("obj_types.json")
    for obj_type, ids in obj_types.items():
        if obj_id in ids:
            return obj_type
    return None


def sample_obj_id_from_types(obj_types, obj_type, num_samples=-1):
    ids = obj_types[obj_type]
    if num_samples < 0:
        return ids
    return random.sample(ids, min(num_samples, len(ids)))


def sample_obj_id(dataset_dir="datasets/partnet-mobility-v0/dataset",
                  obj_type="all", num_samples=-1):
    obj_types = track_obj_types(dataset_dir)
    if obj_type != "all" and obj_type not in obj_types:
        raise ValueError(f"Object type '{obj_type}' not found in the dataset.")

    all_samples = []
    if obj_type == "all":
        samples_per_type = num_samples // len(obj_types)
        for type in obj_types.keys():
            all_samples.extend(
                sample_obj_id_from_types(obj_types, type, samples_per_type)
            )
    else:
        all_samples = sample_obj_id_from_types(
            obj_types, obj_type, num_samples)

    return all_samples


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_semantic_name_of_link(link_name, obj_id):
    input_dir = f"datasets/partnet-mobility-v0/dataset/{obj_id}"
    semantic_file = join_path(input_dir, "semantics.txt")
    link_semantics = load_semantic(semantic_file)
    return link_semantics.get(link_name, link_name)


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


def get_obj_ids(dataset_dir):
    obj_ids = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(join_path(dataset_dir, d))
    ]
    obj_ids.sort(key=int)  # Sort numerically
    return obj_ids


def get_joint_type(obj_id, joint_idx, joint_data, semantic_name=False):
    for joint_type, data in joint_data.items():
        for obj in data["obj_ids"]:
            if semantic_name:
                if obj[0] == obj_id and obj[2] == joint_idx:
                    return joint_type
            else:
                if obj[0] == obj_id and obj[1] == joint_idx:
                    return joint_type
    return None




def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def sample_joint_task(joint_type="all", num_samples=-1):
    if not os.path.exists("joint_data.pkl"):
        extract_joint_data_and_stats()
    joint_data = load_pickle("joint_data.pkl")

    if joint_type == "all":
        all_joints = []
        for jt, data in joint_data.items():
            if jt != "fixed":  # Skip 'fixed' joint type
                all_joints.extend(data["obj_ids"])
        if num_samples == -1:
            sampled_joints = all_joints
        else:
            sampled_joints = random.sample(
                all_joints, min(num_samples, len(all_joints))
            )
    else:
        if joint_type in joint_data:
            obj_joints = joint_data[joint_type]["obj_ids"]
            if num_samples == -1:
                sampled_joints = obj_joints
            else:
                sampled_joints = random.sample(
                    obj_joints, min(num_samples, len(obj_joints))
                )
        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

    return sampled_joints
