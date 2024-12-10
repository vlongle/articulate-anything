import pandas as pd
import numpy as np
import os
from articulate_anything.agent.agent import Agent
from articulate_anything.utils.utils import (
    save_json,
    join_path,
    string_to_file,
    create_dir
)
from typing import List, Tuple, Dict
import logging
from articulate_anything.utils.clip_utils import (
    ClipModel, cosine_similarity
)
from articulate_anything.api.odio_urdf import (
    Geometry,
    Mesh,
    Visual,
    Collision,
    Link,
    Box,
)
from articulate_anything.utils.mesh_utils import add_mesh


class PartnetMeshRetrieval(Agent):
    OUT_RESULT_PATH = "semantic_search_result.json"
    UNFILTERED_OUT_RESULT_PATH = "semantic_search_all_result.json"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.df = pd.read_csv(cfg.partnet_mesh_retrieval.embedding_path)
        self.df['embedding'] = self.df['embedding'].apply(
            lambda x: np.array(eval(x)))

        self.clip_model = ClipModel("ViT-B/32")
        self.bigger_clip_model = ClipModel("ViT-L/14")

        self.clip_threshold = cfg.partnet_mesh_retrieval.threshold or 0.5

    def _make_system_instruction(self):
        return "COMPOSITE SYSTEM"

    def select_mesh(self, description: str, topk: int = 1) -> List[Tuple[str, float, str]]:
        embedding = np.array(
            self.clip_model.get_embedding(description)).flatten()
        cosine_similarity_lamb = self.df['embedding'].apply(
            lambda x: cosine_similarity(x, embedding)
        ).values
        topk_idx = np.argsort(cosine_similarity_lamb)[::-1][:topk]

        results = []
        for idx in topk_idx:
            path, link_name = self.df.iloc[idx][['path', 'link_name']]
            mesh_file = join_path(path, f"{link_name}_combined_mesh.obj")
            cosine_value = cosine_similarity_lamb[idx]
            mesh_description = self.df.iloc[idx]['semantic_label']
            results.append((mesh_file, cosine_value, mesh_description))

        return results

    def rerank_results(self, results: List[Tuple[str, float, str]], description: str) -> List[Tuple[str, float, str]]:
        reranked_results = []
        for mesh_file, _, mesh_description in results:
            new_cosine_value = self.bigger_clip_model.cosine_similarity_text(
                description, mesh_description)
            reranked_results.append(
                (mesh_file, float(new_cosine_value), mesh_description))
        return sorted(reranked_results, key=lambda x: x[1], reverse=True)

    def generate_prediction(self, box_layout, overwrite=False, include_description=True,
                            rerank_using_bigger_clip=False, **kwargs):
        if (
            os.path.exists(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))
            and not overwrite
        ):
            logging.info(
                f"{self.__class__.__name__}: Prediction already exists. Skipping generation.")
            return

        meshes = {}
        for box in box_layout:
            name = box["name"]
            link_description = box["name"]
            if include_description:
                link_description += ". " + box["description"]

            result = self.select_mesh(link_description, topk=5)

            if rerank_using_bigger_clip:
                result = self.rerank_results(result, link_description)
            mesh_file, cosine_value, mesh_description = result[0]

            meshes[name] = {
                'mesh_file': mesh_file,
                'cosine_similarity': cosine_value,
                'mesh_description': mesh_description,
                "link_description": link_description
            }

        self.parse_response(meshes)
        return meshes

    def parse_response(self, meshes, **kwargs):
        logging.info(f">>meshes {meshes}")
        save_json(meshes, join_path(self.cfg.out_dir,
                  self.UNFILTERED_OUT_RESULT_PATH))
        filtered_meshes = {k: v for k, v in meshes.items(
        ) if v['cosine_similarity'] > self.clip_threshold}
        save_json(filtered_meshes, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))


def write_box_layout_to_link_summary(box_layout: List[Dict], output_path: str) -> None:
    """
    Write the box layout to a link summary file. This is then fed into the LinkPlacementActor

    Args:
    box_layout (List[Dict]): List of dictionaries containing box layout information.
    output_path (str): Path to save the link summary file.
    """
    create_dir(os.path.dirname(output_path))

    object_id = "111"

    link_summary = [
        f"object_id: {object_id}",
        "Robot Link Summary:\n",
        "- base"
    ]
    link_summary.extend([f"- {item['name']}" for item in box_layout])
    description = '\n'.join([
        '- ' + str({'link_name': e['name'], 'description': e['description']}) for e in box_layout])

    link_summary.extend([
        "",
        "Here's the semantic meaning of the links:",
        "",
        description,
        "",
        "Please take the semantic description of the links into account to produce accurate link placement."
    ])

    link_summary = "\n".join(link_summary)

    string_to_file(link_summary, output_path)


def make_links_from_json(json_data: List[Dict]) -> Dict:
    """
    Create links from JSON data.

    Args:
    json_data (List[Dict]): List of dictionaries containing link information.

    Returns:
    Dict: Dictionary of created links.
    """
    links = {}
    for link in json_data:
        if "mesh_file" in link:
            geometry = Geometry(Mesh(link['mesh_file']))
        else:
            geometry = Geometry(Box(link['dimensions']))
        visual = Visual(geometry)
        collision = Collision(geometry)
        links[link['name']] = Link(link['name'], visual, collision)
    links["base"] = Link("base")
    return links


def save_mesh_to_obj(vertices: np.ndarray, faces: np.ndarray, filename: str = 'mesh.obj') -> None:
    """
    Save mesh data to an OBJ file.

    Args:
    vertices (np.ndarray): Array of vertex coordinates.
    faces (np.ndarray): Array of face indices.
    filename (str): Path to save the OBJ file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    faces = faces + 1  # OBJ indices start at 1

    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    logging.info(f"Mesh saved to {filename}")


def add_mesh_to_out_folder(task: str, box_layout: List[Dict], meshes: Dict, task_out: str, overwrite: bool = False) -> List[Dict]:
    """
    Add mesh to output folder and update box layout.

    Args:
    task (str): Task name.
    box_layout (List[Dict]): List of dictionaries containing box layout information.
    meshes (Dict): Dictionary of mesh files.
    task_out (str): Output directory for the task.
    overwrite (bool): Whether to overwrite existing files.

    Returns:
    List[Dict]: Updated box layout.
    """
    new_box_layout = []
    for box_spec in box_layout:
        link_name = box_spec["name"]
        if link_name not in meshes:
            continue

        mesh_file = meshes[link_name]
        mesh_path = join_path(task_out, task, f"{link_name}.obj")

        vertices, faces = add_mesh(box_spec, mesh_file)
        if not os.path.exists(mesh_path) or overwrite:
            save_mesh_to_obj(vertices, faces, mesh_path)

        logging.info(
            f">>> Saved mesh to {mesh_path} from partnet: {mesh_file}")

        box_spec["mesh_file"] = os.path.basename(mesh_path)
        new_box_layout.append(box_spec)

    return new_box_layout
