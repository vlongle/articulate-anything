
from typing import Dict, Optional, Union, Tuple
from articulate_anything.api.odio_urdf import (
    link_data_to_np,
    load_semantic,
)
from articulate_anything.utils.utils import load_json, join_path
from articulate_anything.utils.partnet_utils import extract_obj_type, load_pickle, extract_joint_data_and_stats, get_joint_type
import numpy as np
import os
import shutil
from collections import defaultdict


def average_link_difference(
    data1: Union[str, Dict], data2: Union[str, Dict], semantic_file: Optional[Union[str, Dict]] = None, strict=False, allow_equivalent=True
):
    """Calculates average position and orientation differences between links in two files."""
    if isinstance(data1, str):
        data1 = load_json(data1)
    if isinstance(data2, str):
        data2 = load_json(data2)

    data1 = link_data_to_np(data1)
    data2 = link_data_to_np(data2)

    # pos_key = "position"
    pos_key = "bbox_position"

    # Semantic remapping if semantic file is provided
    if semantic_file is not None:
        # remap the link names in data1 to the semantic labels
        link_semantics = load_semantic(semantic_file)
        data1 = {
            link_semantics.get(link_name, link_name): data1[link_name]
            for link_name in data1.keys()
        }

    # Ensure link names match
    link_names1 = set(data1.keys())
    link_names2 = set(data2.keys())

    if strict:
        if link_names1 != link_names2:
            err = "Link names in the two files do not match. links {}, while  links {}".format(
                link_names1, link_names2
            )
            raise ValueError(err)

        link_names = link_names1
    else:
        # get the common link names
        link_names = link_names1.intersection(link_names2)

    # Function to find the best matching link
    def find_best_match(link_name, pos1):
        base_name = link_name.split("_")[0]
        candidates = [
            name for name in link_names2 if name.startswith(base_name)]
        if not candidates:
            return None, float("inf")

        min_distance = float("inf")
        best_match = None
        for candidate in candidates:
            pos2 = data2[candidate][pos_key]
            distance = np.linalg.norm(pos1 - pos2)
            if distance < min_distance:
                min_distance = distance
                best_match = candidate

        return best_match, min_distance

    # Calculate differences
    position_differences = {}
    orientation_differences = {}
    matched_links = set()

    for link_name in link_names:
        pos1 = data1[link_name][pos_key]
        or1 = data1[link_name]["orientation"]

        if allow_equivalent:
            best_match, position_diff = find_best_match(link_name, pos1)
            if best_match is None:
                continue
            matched_links.add(best_match)
            pos2 = data2[best_match][pos_key]
            or2 = data2[best_match]["orientation"]
        else:
            pos2 = data2[link_name][pos_key]
            or2 = data2[link_name]["orientation"]
            position_diff = np.linalg.norm(pos1 - pos2)

        orientation_diff = 2 * np.arccos(np.abs(np.sum(or1 * or2)))

        position_differences[link_name] = position_diff
        orientation_differences[link_name] = orientation_diff

    return {
        "position_diff": position_differences,
        "orientation_diff": orientation_differences,
    }


def to_array(string):
    return np.array([float(x) for x in string.split()])


def compute_joint_diff(
    gt_joint_state, pred_joint_state, joint_semantics, joint_name
):
    def norm(a):
        return np.linalg.norm(a)

    def get_angle(axis1, axis2):
        return np.arccos(
            np.clip(np.dot(axis1, axis2) /
                    (norm(axis1) * norm(axis2)), -1.0, 1.0)
        )

    def compute_joint_limit_diff(a1, l1, u1, a2, l2, u2):
        m1 = a1 * (u1 - l1)
        m2 = a2 * (u2 - l2)

        dir1 = m1 / norm(m1) if norm(m1) != 0 else a1
        dir2 = m2 / norm(m2) if norm(m2) != 0 else a2

        # Unified motion difference
        motion_range_diff = norm(m2 - m1)
        motion_direction_diff = 1 - np.dot(dir1, dir2)  # between 0 and 2
        # 0: identical, 1: perpendicular, 2: opposite
        # opossite is like when the door opening inward instead of outward

        return {
            "direction_diff": motion_direction_diff,
            "range_diff": motion_range_diff,
        }

    joint = gt_joint_state[joint_name]
    semantic_joint_name = joint_semantics.get(joint_name)
    if semantic_joint_name not in pred_joint_state:
        return {}
    semantic_joint = pred_joint_state[semantic_joint_name]

    pos1 = to_array(joint["joint_origin"]["xyz"])
    pos2 = to_array(semantic_joint["joint_origin"]["xyz"])
    or1 = to_array(joint["joint_origin"]["orientation"])
    or2 = to_array(semantic_joint["joint_origin"]["orientation"])
    a1 = to_array(joint["joint_axis"])
    a2 = to_array(semantic_joint["joint_axis"])

    # Compare joint axis
    axis_diff = get_angle(a1, a2)
    axis_diff = min(axis_diff, get_angle(-a2, a1))  # Check both directions

    l1, u1 = float(joint["joint_limit"]["lower"]), float(
        joint["joint_limit"]["upper"])
    l2, u2 = float(semantic_joint["joint_limit"]["lower"]), float(
        semantic_joint["joint_limit"]["upper"]
    )

    original_diff = compute_joint_limit_diff(a1, l1, u1, a2, l2, u2)
    flipped_diff = compute_joint_limit_diff(a1, l1, u1, -a2, -u2, -l2)
    if original_diff["direction_diff"] < flipped_diff["direction_diff"]:
        motion_diff = original_diff
    else:
        motion_diff = flipped_diff

    # Compare joint origin
    if joint["joint_type"] == "revolute":
        # For revolute joints, use cross product method to find shortest distance between axes
        w = np.cross(a2, a1)
        if np.allclose(w, 0):
            # The axes are parallel or collinear
            w = (
                np.cross(a2, [1, 0, 0])
                if not np.allclose(a2, [1, 0, 0])
                else np.cross(a1, [0, 1, 0])
            )
        p = pos1 - pos2
        origin_diff = np.abs(np.dot(p, w)) / norm(w)
    else:
        # For other joint types, use Euclidean distance
        origin_diff = norm(pos1 - pos2)

    # Compare orientation (using the dot product method)
    orientation_diff = 2 * np.arccos(np.abs(np.sum(or1 * or2)))

    return {
        "joint_type": joint["joint_type"] == semantic_joint["joint_type"],
        "joint_origin": {
            "xyz": origin_diff,
            "orientation": orientation_diff,
        },
        "joint_axis": axis_diff,  # zero is perfectly aligned, pi/2 is orthogonal, pi is opposite
        "joint_limit": motion_diff,
    }



def copy_meta(result_dir="results", input_dir="datasets/partnet-mobility-v0/dataset"):

    for obj_id in os.listdir(result_dir):
        # Paths to source and destination files
        source_meta_file = os.path.join(input_dir, obj_id, "meta.json")
        destination_meta_file = os.path.join(result_dir, obj_id, "meta.json")

        # Check if source file exists
        if os.path.isfile(source_meta_file):
            # Copy the file
            shutil.copy(source_meta_file, destination_meta_file)
        else:
            continue



def pick_best_link_gt(obj_dir, iter_num=None, get_max=True):
    """
    Pick the best among different seeds and iter_nums
    """
    link_placement_dir = os.path.join(obj_dir, "link_placement")
    best_result = None
    best_diff = float("inf")
    obj_type = extract_obj_type(obj_dir)
    if iter_num is None:
        iter_nums = [
            int(iter_num.split("_")[1])
            for iter_num in os.listdir(link_placement_dir)
            if iter_num.startswith("iter_")
            and os.path.isdir(os.path.join(link_placement_dir, iter_num))
        ]
    else:
        iter_nums = list(range(int(iter_num) + 1)
                         ) if get_max else [int(iter_num)]

    for iter_num in iter_nums:
        iter_dir = os.path.join(link_placement_dir, f"iter_{iter_num}")
        if not os.path.isdir(iter_dir):
            continue
        for seed in os.listdir(iter_dir):
            seed_dir = os.path.join(iter_dir, seed)
            if not os.path.isdir(seed_dir):
                continue
            link_diff_path = os.path.join(seed_dir, "link_diff.json")
            if not os.path.exists(link_diff_path):
                continue

            link_diff = load_json(link_diff_path)
            reduced_diff = reduce_link_diff(link_diff)
            if reduced_diff["average_diff"] < best_diff:
                best_diff = reduced_diff["average_diff"]
                best_result = {
                    "iter_num": iter_num,
                    "seed": seed,
                    "link_diff": reduced_diff,
                    "obj_type": obj_type,
                }

    return best_result


def pick_best_by_critic(obj_dir, task_name="link_critic", iter_num=None, get_max=True):
    """
    Pick the best among different seeds and iter_nums according to the critic
    """
    link_critic_dir = os.path.join(obj_dir, task_name)
    best_result = None
    best_score = float("-inf")
    obj_type = extract_obj_type(obj_dir)
    if iter_num is None:
        iter_nums = [
            int(iter_num.split("_")[1])
            for iter_num in os.listdir(link_critic_dir)
            if iter_num.startswith("iter_")
            and os.path.isdir(os.path.join(link_critic_dir, iter_num))
        ]
    else:
        iter_nums = list(range(int(iter_num) + 1)
                         ) if get_max else [int(iter_num)]

    for iter_num in iter_nums:
        iter_dir = os.path.join(link_critic_dir, f"iter_{iter_num}")
        if not os.path.isdir(iter_dir):
            continue

        for seed in os.listdir(iter_dir):
            seed_dir = os.path.join(iter_dir, seed)
            if not os.path.isdir(seed_dir):
                continue
            link_diff_path = os.path.join(seed_dir, f"{task_name}.json")
            if not os.path.exists(link_diff_path):
                continue

            link_diff = load_json(link_diff_path)
            link_diff["realism_rating"] = int(link_diff["realism_rating"])
            score = link_diff["realism_rating"]
            if score > best_score:
                best_score = score
                best_result = {
                    "iter_num": iter_num,
                    "seed": seed,
                    "link_diff": link_diff,
                    "obj_type": obj_type,
                }

    return best_result



def compute_link_placement_result(
    result_dir, strategy="gt", task_name="link_critic", iter_num=None, get_max=True,
    dataset_dir="datasets/partnet-mobility-v0/dataset"
):
    copy_meta(result_dir, dataset_dir)
    link_diffs = {}
    obj_ids = os.listdir(result_dir)
    for obj_id in obj_ids:
        obj_dir = os.path.join(result_dir, obj_id)
        if not os.path.isdir(obj_dir):
            continue
        if strategy == "gt":
            result = pick_best_link_gt(
                obj_dir, iter_num=iter_num, get_max=get_max)
            # Assume worst case for gt strategy
            if result is None:
                result = {
                    "iter_num": -1,
                    "seed": "none",
                    "link_diff": {
                        "position_diff": float('inf'),
                        "orientation_diff": float('inf'),
                        "average_diff": float('inf')
                    },
                    "obj_type": extract_obj_type(obj_dir)
                }

        else:
            result = pick_best_by_critic(
                obj_dir, task_name=task_name, iter_num=iter_num, get_max=get_max
            )
            if result is None:
                # Assume worst case for critic strategy
                result = {
                    "iter_num": -1,
                    "seed": "none",
                    "link_diff": {
                        "realism_rating": 0  # Assuming 0 is the worst rating
                    },
                    "obj_type": extract_obj_type(obj_dir)
                }

        link_diffs[obj_id] = result

    return link_diffs



def compute_avg_diff_by_type(link_diffs, metric="average_diff", include_avg=False):
    """
    Computes the average diff for each object type and optionally for the 'Average' type.

    Now stores the object ID along with the metric value in a nested dictionary.
    """
    type_diffs = defaultdict(dict)

    for obj_id, diff in link_diffs.items():
        obj_type = diff["obj_type"]
        avg_diff = diff["link_diff"][metric]
        # Store obj_id and avg_diff together
        type_diffs[obj_type][obj_id] = avg_diff

    if include_avg:
        all_diffs = {
            obj_id: diff
            for obj_type in type_diffs.values()
            for obj_id, diff in obj_type.items()
        }
        type_diffs["Average"] = (
            all_diffs  # Add all diffs as a single dict under 'Average'
        )

    return type_diffs


def is_success(diff, cutoff, metric):
    if metric == "average_diff":
        return diff < cutoff
    elif metric == "realism_rating":
        return diff > cutoff


def reduce_link_diff(link_diff):
    """Reduces a link difference dictionary to average position and orientation differences.

    Args:
        link_diff: A dictionary containing:
            - "position_diff": A dictionary mapping link names to position differences.
            - "orientation_diff": A dictionary mapping link names to orientation differences.

    Returns:
        A dictionary containing:
            - "position_diff": The average position difference across all links.
            - "orientation_diff": The average orientation difference across all links.
    """

    position_diffs = list(link_diff.get("position_diff", {}).values())
    orientation_diffs = list(link_diff.get("orientation_diff", {}).values())

    # Handle cases with empty link lists gracefully
    if position_diffs:
        average_position_diff = sum(position_diffs) / len(position_diffs)
    else:
        average_position_diff = 0.0  # Or another default value

    if orientation_diffs:
        average_orientation_diff = sum(
            orientation_diffs) / len(orientation_diffs)
    else:
        average_orientation_diff = 0.0

    return {
        "position_diff": average_position_diff,
        "orientation_diff": average_orientation_diff,
        "average_diff": (average_position_diff + average_orientation_diff) / 2,
    }

### JOINT


def compute_joint_diff_score(
    j,
    angular_cutoff=0.25,
    euclidean_cutoff=0.05,
    angular_limit_cutoff=1.6057,  # 92 degree difference
    # Allow up to perpendicular (mostly for detecting opposite limit)
    direction_cutoff=1.0,
):  # 30 degree
    """
    Convention for gt: Smaller score is better
    """

    rating = {
        "success": 10,
        "joint_type": 0,
        "joint_axis": 1,
        "joint_origin": 2,
        "joint_limit": 3,
    }

    def compute_diff(score):
        return (10 - score) / 10  # 0 to 1
    if j["joint_type"] != 1:
        # joint type is not predicted correctly
        failure_reason = "joint_type"
        return compute_diff(rating[failure_reason]), failure_reason
    if j["joint_axis"] > angular_cutoff:  # NOTE: a bit hacky
        failure_reason = "joint_axis"
        return compute_diff(rating[failure_reason]), failure_reason

    origin_er = j["joint_origin"]
    origin_er = (origin_er["xyz"] + origin_er["orientation"]) / 2
    if origin_er > euclidean_cutoff:
        failure_reason = "joint_origin"
        return compute_diff(rating[failure_reason]), failure_reason

    if "joint_limit" in j:
        if isinstance(j["joint_limit"], dict) and "direction_diff" in j["joint_limit"]:
            direction_diff = j["joint_limit"]["direction_diff"]
            range_diff = j["joint_limit"]["range_diff"]

            if direction_diff > direction_cutoff:
                failure_reason = "joint_limit"
                return compute_diff(rating[failure_reason]), failure_reason

            if range_diff > angular_limit_cutoff:
                failure_reason = "joint_limit"
                return compute_diff(rating[failure_reason]), failure_reason
        else:
            range_diff = j["joint_limit"]
            if range_diff > angular_cutoff:
                failure_reason = "joint_limit"
                return compute_diff(rating[failure_reason]), failure_reason

    failure_reason = "success"
    return 0, failure_reason


def pick_best_joint_gt(joint_dir, iter_num=None, get_max=True, reference_result=None):
    if reference_result:
        iter_num = reference_result["iter_num"]
        seed = reference_result["seed"]
        iter_dir = os.path.join(joint_dir, f"iter_{iter_num}")
        seed_dir = os.path.join(iter_dir, seed)
        joint_diff_path = os.path.join(seed_dir, "joint_diff.json")

        if os.path.exists(joint_diff_path):
            joint_diff = load_json(joint_diff_path)
            if not joint_diff:
                return None
            score, failure_reason = compute_joint_diff_score(joint_diff)
            return {
                "iter_num": iter_num,
                "seed": seed,
                "joint_diff": joint_diff,
                "failure_reason": failure_reason,
                "score": score,
            }
        else:
            return None

    results = []
    if not os.path.isdir(joint_dir):
        return None
    if iter_num is None:
        iter_nums = [
            int(iter_num.split("_")[1])
            for iter_num in os.listdir(joint_dir)
            if iter_num.startswith("iter_")
            and os.path.isdir(os.path.join(joint_dir, iter_num))
        ]
    else:
        iter_nums = list(range(int(iter_num) + 1)
                         ) if get_max else [int(iter_num)]

    for iter_num in sorted(iter_nums):
        iter_dir = os.path.join(joint_dir, f"iter_{iter_num}")
        if not os.path.isdir(iter_dir):
            continue

        for seed in sorted(os.listdir(iter_dir)):
            seed_dir = os.path.join(iter_dir, seed)
            if not os.path.isdir(seed_dir):
                continue
            joint_diff_path = os.path.join(seed_dir, f"joint_diff.json")
            if not os.path.exists(joint_diff_path):
                continue

            joint_diff = load_json(joint_diff_path)
            if not joint_diff:
                continue
            score, failure_reason = compute_joint_diff_score(joint_diff)
            results.append(
                {
                    "iter_num": iter_num,
                    "seed": seed,
                    "joint_diff": joint_diff,
                    "failure_reason": failure_reason,
                    "score": score,
                }
            )

    return pick_best_gt(results)


def pick_best_joint_by_critic(
    joint_dir,
    task_name="joint_critic",
    iter_num=None,
    get_max=True,
    reference_result=None,
):
    if reference_result:
        iter_num = reference_result["iter_num"]
        seed = reference_result["seed"]
        iter_dir = os.path.join(joint_dir, f"iter_{iter_num}")
        seed_dir = os.path.join(iter_dir, seed)
        joint_diff_path = os.path.join(seed_dir, f"{task_name}.json")

        if os.path.exists(joint_diff_path):
            joint_diff = load_json(joint_diff_path)
            joint_diff["realism_rating"] = int(joint_diff["realism_rating"])
            score = joint_diff["realism_rating"]
            if "failure_reason" in joint_diff:
                failure_reason = joint_diff["failure_reason"]
            return {
                "iter_num": iter_num,
                "score": score,
                "seed": seed,
                "joint_diff": joint_diff,
                "failure_reason": failure_reason,
            }
        else:
            return None

    results = []
    if not os.path.isdir(joint_dir):
        return None
    if iter_num is None:
        iter_nums = [
            int(iter_num.split("_")[1])
            for iter_num in os.listdir(joint_dir)
            if iter_num.startswith("iter_")
            and os.path.isdir(os.path.join(joint_dir, iter_num))
        ]
    else:
        iter_nums = list(range(int(iter_num) + 1)
                         ) if get_max else [int(iter_num)]

    for iter_num in sorted(iter_nums):
        iter_dir = os.path.join(joint_dir, f"iter_{iter_num}")
        if not os.path.isdir(iter_dir):
            continue

        for seed in sorted(os.listdir(iter_dir)):
            seed_dir = os.path.join(iter_dir, seed)
            if not os.path.isdir(seed_dir):
                continue
            joint_diff_path = os.path.join(seed_dir, f"{task_name}.json")
            if not os.path.exists(joint_diff_path):
                continue

            joint_diff = load_json(joint_diff_path)
            joint_diff["realism_rating"] = int(joint_diff["realism_rating"])
            score = joint_diff["realism_rating"]
            if "failure_reason" in joint_diff:
                failure_reason = joint_diff["failure_reason"]
            else:
                cutoff = 5
                failure_reason = "success" if score > cutoff else "fail_legacy"

            results.append(
                {
                    "iter_num": iter_num,
                    "score": score,
                    "seed": seed,
                    "joint_diff": joint_diff,
                    "failure_reason": failure_reason,
                }
            )

    return pick_best_critic(results)


def pick_best_critic(results):
    # higher score is better
    # Sort results by score (descending), then by iteration (descending), then by seed (ascending)
    sorted_results = sorted(
        # results, key=lambda x: (-x["score"], -x["iter_num"], x["seed"])
        results,
        # Sort results by iteration (descending), then by score (descending), then by seed (ascending)
        # key=lambda x: (-x["iter_num"], -x["score"], x["seed"]),
        key=lambda x: (-x["score"], -x["iter_num"], x["seed"]),
    )
    return sorted_results[0] if sorted_results else None


def pick_best_gt(results):
    # lower "score" is better because "score" is actually the "avg_diff"
    # Sort results by score (descending), then by iteration (descending), then by seed (ascending)
    sorted_results = sorted(
        # results, key=lambda x: (-x["score"], -x["iter_num"], x["seed"])
        results,
        # Sort results by iteration (descending), then by score (descending), then by seed (ascending)
        # key=lambda x: (-x["iter_num"], -x["score"], x["seed"]),
        key=lambda x: (x["score"], -x["iter_num"], x["seed"]),
    )
    return sorted_results[0] if sorted_results else None


def compute_joint_pred_result(
    result_dir,
    strategy="gt",
    dataset_dir="datasets/partnet-mobility-v0/dataset",
    ignore_origin_prismatic=True,
    ignore_limit_prismatic=True,
    iter_num=None,
    get_max=True,
    reference_results=None,
    joint_types=["revolute", "prismatic"],
    use_result_dir=False
):
    copy_meta(result_dir, dataset_dir)
    if not os.path.exists("joint_data.pkl"):
        extract_joint_data_and_stats(dataset_dir=dataset_dir)

    joint_data = load_pickle("joint_data.pkl")
    joint_diffs = {}
    task_name = "joint_critic" if strategy == "critic" else "joint_pred"

    def process_joint(obj_id, joint_id, semantic_joint_id):
        obj_dir = os.path.join(result_dir, obj_id)
        obj_type = extract_obj_type(obj_dir)
        joint_dir = os.path.join(obj_dir, task_name, joint_id)
        joint_type = get_joint_type(obj_id, joint_id, joint_data)
        key = (obj_id, joint_id, semantic_joint_id)
        reference_result = reference_results.get(
            key) if reference_results else None

        pick_function = pick_best_joint_by_critic if strategy == "critic" else pick_best_joint_gt
        result = pick_function(joint_dir, iter_num=iter_num,
                               get_max=get_max, reference_result=reference_result)

        if result is None:
            result = create_worst_case_critic_result(
            ) if strategy == "critic" else create_worst_case_gt_result()
        elif strategy == "gt":
            result = adjust_result_for_prismatic(
                result, joint_type, ignore_origin_prismatic, ignore_limit_prismatic)

        result.update({"obj_type": obj_type, "joint_type": joint_type})
        return key, result

    if use_result_dir:
        for obj_id in os.listdir(result_dir):
            task_dir = os.path.join(result_dir, obj_id, task_name)
            if not os.path.isdir(task_dir):
                continue
            for joint_id in os.listdir(task_dir):
                semantic_joint_id = get_semantic_joint_id(
                    obj_id, joint_id, input_dir=dataset_dir)
                key, result = process_joint(
                    obj_id, joint_id, semantic_joint_id)
                joint_diffs[key] = result
    else:
        for joint_type in joint_types:
            if joint_type not in joint_data:
                continue
            for obj_id, joint_id, semantic_joint_id in joint_data[joint_type]['obj_ids']:
                key, result = process_joint(
                    obj_id, joint_id, semantic_joint_id)
                joint_diffs[key] = result

    return joint_diffs


def create_worst_case_gt_result():
    return {
        "iter_num": -1,
        "seed": "none",
        "joint_diff": {
            "joint_type": False,
            "joint_origin": {"xyz": float('inf'), "orientation": float('inf')},
            "joint_axis": float('inf'),
            "joint_limit": {"direction_diff": float('inf'), "range_diff": float('inf')},
            "average_diff": float('inf')
        },
        "failure_reason": "no_valid_result",
        "score": float('inf')
    }


def create_worst_case_critic_result():
    return {
        "iter_num": -1,
        "seed": "none",
        "failure_reason": "no_valid_result",
        "joint_diff": {
            "realism_rating": 0  # Assuming 0 is the worst rating
        },
        "score": 0
    }


def adjust_result_for_prismatic(result, joint_type, ignore_origin_prismatic, ignore_limit_prismatic):
    if joint_type == "prismatic":
        if ignore_origin_prismatic and result["failure_reason"] == "joint_origin":
            result["failure_reason"] = "success"
            result["score"] = 0
        if ignore_limit_prismatic and result["failure_reason"] == "joint_limit":
            result["failure_reason"] = "success"
            result["score"] = 0
    return result





def analyze_link_placement_by_type(
    link_diffs, cutoff=0.05, metric="average_diff", include_avg=True
):
    type_diffs = compute_avg_diff_by_type(
        link_diffs, metric=metric, include_avg=include_avg
    )
    stats = defaultdict(dict)
    for (
        obj_type,
        diffs,
    ) in type_diffs.items():  # diffs is now a dictionary of obj_id: diff
        stats[obj_type]["avg_diff"] = np.mean(list(diffs.values()))
        stats[obj_type]["count"] = len(diffs)

        # Store list of dictionaries for object ID and success/failure status
        success_dicts = {
            obj_id: is_success(diff, cutoff, metric) for obj_id, diff in diffs.items()
        }
        stats[obj_type]["success_dicts"] = success_dicts
        success_count = sum(success_dicts.values())
        stats[obj_type]["success_rate"] = success_count / len(diffs)
        stats[obj_type]["num_failures"] = len(diffs) - success_count

    return stats


def analyze_joint_pred_by_type(joint_diffs, include_avg=True):
    stats = defaultdict(lambda: defaultdict(
        lambda: {'success_dicts': {}, 'count': 0}))

    for (obj_id, joint_id, semantic_joint_id), result in joint_diffs.items():
        obj_type = result['obj_type']
        joint_type = result['joint_type']
        is_success = result['failure_reason'] == 'success'
        joint_key = (obj_id, joint_id, semantic_joint_id)

        # Update joint type stats
        stats[obj_type][joint_type]['success_dicts'][joint_key] = is_success
        stats[obj_type][joint_type]['count'] += 1

        # Update total stats for object type
        stats[obj_type]['total']['success_dicts'][joint_key] = is_success
        stats[obj_type]['total']['count'] += 1

    # Calculate success rates
    for obj_type, joint_types in stats.items():
        for joint_type, data in joint_types.items():
            data['success_rate'] = sum(
                data['success_dicts'].values()) / data['count']

    if include_avg:
        avg_stats = defaultdict(lambda: {'success_dicts': {}, 'count': 0})

        for obj_type, joint_types in stats.items():
            for joint_type, data in joint_types.items():
                avg_stats[joint_type]['success_dicts'].update(
                    data['success_dicts'])
                avg_stats[joint_type]['count'] += data['count']

        # Calculate success rates for average
        for joint_type, data in avg_stats.items():
            data['success_rate'] = sum(
                data['success_dicts'].values()) / data['count']

        avg_stats["success_rate"] = avg_stats["total"]["success_rate"]
        avg_stats["count"] = avg_stats["total"]["count"]
        stats['Average'] = avg_stats

    return stats