from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import pandas as pd
import itertools
import os
import glob
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
import logging
from articulate_anything.utils.utils import load_json
from articulate_anything.utils.clip_utils import ClipModel

logging.basicConfig(level=logging.INFO)

"""
Create CLIP embedding for each part mesh in the PartNet Mobility dataset
using `object_annotation_gemini.txt` file.
"""


def create_embedding_entries(obj_dir):
    annotation_path = join_path(obj_dir, "object_annotation_gemini.json")
    logging.info(f"👾 Processing {annotation_path}")

    if not os.path.exists(annotation_path):
        logging.error(f"  ❌ Annotation file not found: {annotation_path}")
        return []

    entries = []

    semantics = load_json(annotation_path)['annotation']

    links = [k for k in semantics.keys() if k.startswith("link")]
    semantic_labels = []
    for link_name in links:
        entry = {}
        entry['path'] = obj_dir
        entry['link_name'] = link_name
        semantic_label = semantics[link_name]
        entry['semantic_label'] = semantic_label
        semantic_labels.append(semantic_label)
        entries.append(entry)

    clip_model = ClipModel()
    embeddings = clip_model.get_embedding(semantic_labels)
    for i, entry in enumerate(entries):
        entry['embedding'] = embeddings[i].tolist()

    logging.info(f"  ✅ Processed {annotation_path}")

    return entries


def embedding_entries_to_csv(embedding_entries, csv_path="partnet_mobility_embeddings.csv"):
    # use pandas
    # entries is [list of dictionaries]
    # each dictionary has keys: link_name, embedding
    embedding_entries = list(itertools.chain(*embedding_entries))
    df = pd.DataFrame(embedding_entries)
    if os.path.exists(csv_path):
        # File exists, append without header
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # File doesn't exist, create new file with header
        df.to_csv(csv_path, index=False)
    logging.info(f"📝 Saved to {csv_path}")


if __name__ == '__main__':
    DATASET_DIR = "datasets/partnet-mobility-v0/dataset/"
    obj_dirs = glob.glob(DATASET_DIR + "*/")
    mp.set_start_method('spawn')
    logging.info("Starting")
    steps = [
        (create_embedding_entries, embedding_entries_to_csv),]
    for i, step in enumerate(steps):
        logging.info(f"✅ Step {i}")
        worker, aggregator = step
        # with ProcessPoolExecutor(max_workers=6) as executor:
        #     outputs = list(executor.map(worker, obj_dirs))
        outputs = list(map(worker, obj_dirs))

        if aggregator is not None:
            aggregator(outputs)

    logging.info("Done")
