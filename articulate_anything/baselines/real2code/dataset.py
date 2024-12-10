from articulate_anything.utils.partnet_utils import track_obj_types, sample_obj_id
import random
from articulate_anything.utils.utils import save_json, load_json
from collections import Counter
from glob import glob
from torch.utils.data import Dataset as TorchDataset
import json
import os


class LLaMAInstructReal2CodeTextDataset(TorchDataset):
    def __init__(self, base_dir, obj_ids, strat="obb_rel"):
        self.obj_ids = obj_ids
        self.base_dir = base_dir
        self.strat = strat
        
        self.data_files = self._load_data_files()
        self.data = self._load_all_data()
        
    def _load_data_files(self):
        data_files = []
        for obj_id in os.listdir(self.base_dir):
            if obj_id not in self.obj_ids:
                continue
            obj_path = os.path.join(self.base_dir, obj_id, self.strat)
            if os.path.isdir(obj_path):
                data_files.extend(glob(os.path.join(obj_path, f'data_loop_*.json')))
        return sorted(data_files)
    
    def _load_all_data(self):
        all_data = []
        for file in self.data_files:
            with open(file, 'r') as f:
                data = json.load(f)
                for aug_obb, aug_label in zip(data['aug_obbs'], data['aug_labels']):
                    all_data.append((aug_obb, aug_label))
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        aug_obb, aug_label = self.data[idx]
        
        prompt = f"""[INST] You are an AI assistant trained to understand 3D scenes and object relationships. Given the following Oriented Bounding Box (OBB) information, your task is to generate a list of child joints that describes the articulations between object parts.

OBB Information:
{aug_obb}

Generate a list of child joints. Each joint should be described by a dictionary with the following keys:
- box: The ID of the child bounding box
- type: The joint type ('hinge' for revolute joints, 'slide' for prismatic joints)
- idx: The rotation axis index (0 for x-axis, 1 for y-axis, 2 for z-axis)
- edge: Edge coordinates on the OBB, for example [1, -1]
- sign: Direction of the joint (+1 or -1)

IMPORTANT: Your response must contain ONLY the child_joints list, exactly as shown below, with no additional text before or after:
child_joints = [
    dict(box=[child OBB ID], type=[joint type], idx=[rotation axis index], edge=[edge coordinates], sign=[direction]),
    # Additional joints as needed
]


Generate the child_joints list: [/INST]"""

        completion = self._extract_child_joints(aug_label)
        
        # prompt template: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/
        return {
            "prompt": prompt,
            "completion": completion,
            "text": f"<s>{prompt}\n{completion}</s>"
        }
    


    def _extract_child_joints(self, label_code):
        # Extract only the child_joints part from the label_code
        start = label_code.index("child_joints = [")
        end = label_code.rindex("]") + 1
        return label_code[start:end]
    
def sample_proportional_to_category_size(real2code_categories, num_samples):
    obj_types = track_obj_types()
    total_objects = sum(len(obj_types[cat]) for cat in real2code_categories if cat in obj_types)
    
    sampled_objects = []
    for category in real2code_categories:
        if category in obj_types:
            proportion = len(obj_types[category]) / total_objects
            num_category_samples = int(num_samples * proportion)
            category_samples = sample_obj_id(obj_type=category, num_samples=num_category_samples)
            sampled_objects.extend(category_samples)
    
    # Add any remaining samples due to rounding
    while len(sampled_objects) < num_samples:
        category = random.choice(real2code_categories)
        extra_sample = sample_obj_id(obj_type=category, num_samples=1)
        sampled_objects.extend(extra_sample)
    
    return sampled_objects

def train_val_split(real2code_categories, num_trains, num_vals):
    total_samples = num_trains + num_vals
    
    # Sample all objects at once
    all_samples = sample_proportional_to_category_size(real2code_categories, total_samples)
    
    # Shuffle the samples
    random.shuffle(all_samples)
    
    # Split into train and val sets
    train_samples = all_samples[:num_trains]
    val_samples = all_samples[num_trains:]
    
    return train_samples, val_samples






def count_objects_per_category(obj_ids, real2code_categories):
    obj_types = track_obj_types()
    category_counts = Counter()
    for obj_id in obj_ids:
        for category in real2code_categories:
            if obj_id in obj_types.get(category, []):
                category_counts[category] += 1
                break
        else:
            category_counts['Other'] += 1
    return category_counts

def print_category_distribution(obj_ids, real2code_categories, set_name):
    category_counts = count_objects_per_category(obj_ids, real2code_categories)
    total_objects = len(obj_ids)
    print(f"Distribution of objects per category ({set_name} set):")
    print("-------------------------------------")
    for category in real2code_categories:
        count = category_counts[category]
        percentage = (count / total_objects) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")
    if category_counts['Other'] > 0:
        other_count = category_counts['Other']
        other_percentage = (other_count / total_objects) * 100
        print(f"Other: {other_count} ({other_percentage:.2f}%)")
    print(f"Total: {total_objects}")



def make_real2code_dataset(base_dir, strat="obb_rel"):
    val_ids = load_json("val_obj_ids.json")
    val_dataset = LLaMAInstructReal2CodeTextDataset(
        base_dir=base_dir,
        obj_ids=val_ids,
        strat=strat,
    )    
    train_ids = load_json("train_obj_ids.json")
    train_dataset = LLaMAInstructReal2CodeTextDataset(
        base_dir=base_dir,
        obj_ids=train_ids,
        strat=strat,
    )
    return train_dataset, val_dataset



if __name__ == "__main__":
    # get train and val object
    real2code_categories = ["StorageFurniture", "Laptop", "Box", "Table", "Refrigerator"]
    num_trains = 467
    num_vals = 35

    # Perform train-val split
    train_obj_ids, val_obj_ids = train_val_split(real2code_categories, num_trains, num_vals)

    # Save the train and val object IDs
    save_json(train_obj_ids, "train_obj_ids.json")
    save_json(val_obj_ids, "val_obj_ids.json")

    # Print the distribution of objects per category for both sets
    print_category_distribution(train_obj_ids, real2code_categories, "Training")
    print("\n")
    print_category_distribution(val_obj_ids, real2code_categories, "Validation")