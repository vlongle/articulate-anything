from typing import List, Tuple, Dict, Any
import json
from articulate_anything.utils.utils import (
    save_json,
    load_json,
    join_path,
)
from articulate_anything.utils.utils import create_task_config
from PIL import Image
import logging
import os
from articulate_anything.utils.partnet_utils import track_obj_types
from articulate_anything.agent.agent import Agent
from articulate_anything.utils.viz import get_frames_from_video, display_frames

OBJECT_SELECTION_INSTRUCTION = """
You will be presented with a target image and a set of candidate images. Your task is to select the candidate image that is most similar to the target image in terms of the object depicted.

Please consider the following aspects when making your selection:

1. Object type and category.
2. Shape and structure.
3. Key features and characteristics.

Do not consider colors as we can always recolor the assets later. Pay attention to parts of the objects. The selected image should have the same parts as the target image.

Important: The images are numbered starting from 0 (0-indexing).

Provide your response in the following format:
json
{
    "image_description": "Describe the groundtruth image in details including salient features and parts",
    "selected_image": "Image X",
    "reasoning": "Explanation of why this image was selected. Why other images were not selected?"
}

where X is the number of the selected image (0, 1, 2, 3, etc.). 
"""

class ObjectSelector(Agent):
    OUT_RESULT_PATH = "object_selector_result.json"

    def _make_system_instruction(self):
        return OBJECT_SELECTION_INSTRUCTION

    def _make_prompt_parts(self, target_image, candidate_images,
                           **kwargs):
        prompt_parts = ["Here is the target image:\n", target_image,
                        "\n\nHere are the candidate images (numbered starting from 0):\n"]
        for i, img in enumerate(candidate_images):
            prompt_parts.extend([f"\nImage {i}:\n", img])
        return prompt_parts


    def parse_response(self, response, **kwargs):
        json_str = response.text.strip().strip('```json').strip()
        print("json_str:", json_str)
        parsed_response = json.loads(json_str, strict=False)
        logging.info(f"Object selector response: {parsed_response}")
        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))


    def load_prediction(self):
        return load_json(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))


    def load_prediction(self):
        return load_json(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))

    def load_predicted_rendering(self):
        return Image.open(join_path(self.cfg.out_dir, "chosen_object.png"))

class HierarchicalObjectSelector(Agent):
    OUT_RESULT_PATH = "object_selector_result.json"

    def _make_system_instruction(self):
        return "COMPOSITE SYSTEM."


    def hierarchical_selection(self, target_image: Image.Image, candidate_images: List[Image.Image], candidate_obj_ids: List[Any], overwrite: bool = False,
                               max_images=10) -> None:
        """
        Perform hierarchical selection on candidate images while tracking batch and original indices.

        Args:
            target_image (Image.Image): The target image to compare against.
            candidate_images (List[Image.Image]): List of candidate images.
            candidate_obj_ids (List[Any]): List of candidate object IDs.
            overwrite (bool): Whether to overwrite existing results.

        Returns:
            None
        """
        current_candidates = [(i, i, img, obj_id) for i, (img, obj_id) in enumerate(zip(candidate_images, candidate_obj_ids))]
        round_num = 0

        previous_obj_ids = []  # To store obj_ids from the previous round

        while True:
            # Save obj_ids of current candidates at the beginning of each round
            last_round_obj_ids = [obj_id for _, _, _, obj_id in current_candidates]

            # Save obj_ids to the round folder
            round_dir = join_path(self.cfg.out_dir, f"round_{round_num}")
            os.makedirs(round_dir, exist_ok=True)
            save_json(last_round_obj_ids, join_path(round_dir, "obj_ids.json"))

            # If we have only one candidate, break out of the loop
            if len(current_candidates) <= 1:
                break

            # Store the obj_ids from the current round before processing the batch
            previous_obj_ids = last_round_obj_ids

            next_candidates = []
            for i in range(0, len(current_candidates), max_images):
                batch = current_candidates[i:i+max_images]
                best_candidate = self._process_batch(
                    target_image, batch, round_num, i, overwrite)
                next_candidates.append(best_candidate)

            # Update current_candidates
            current_candidates = next_candidates
            round_num += 1

        # After the loop ends, current_candidates contains the final candidate(s)
        # previous_obj_ids contains the obj_ids from the last round with more than one candidate
        best = current_candidates[0]
        prediction = {
            "selected_image": "Image " + str(best[1]),
            "obj_ids": previous_obj_ids,  # obj_ids of candidates in the next-to-last round
        }
        self.save_prediction(prediction)




    def _process_batch(self, target_image: Image.Image, batch: List[Tuple[int, int, Image.Image, Any]], round_num: int, batch_num: int, overwrite: bool) -> Tuple[int, int, Image.Image, Any]:
        """Process a single batch of images within a round, maintaining batch and original indices."""
        batch_images = [img for _, _, img, _ in batch]
        batch_obj_ids = [obj_id for _, _, _, obj_id in batch]

        round_selector = ObjectSelector(create_task_config(
            self.cfg, f"round_{round_num}_{batch_num}"))
        round_selector.generate_prediction(
            target_image, batch_images,
            overwrite=overwrite,)

        # Save obj_ids for this batch
        obj_ids_path = join_path(round_selector.cfg.out_dir, "obj_ids.json")
        save_json(batch_obj_ids, obj_ids_path)

        round_response = round_selector.load_prediction()
        selected_index = int(round_response['selected_image'].split()[-1])

        best_batch_index, best_original_index, best_image, best_obj_id = batch[selected_index]
        # save best image
        best_image.save(join_path(
            round_selector.cfg.out_dir, f"best_image_{best_original_index}.png"))

        return (batch_num, best_original_index, best_image, best_obj_id)

    def generate_prediction(self, target_image: Image.Image, candidate_images: List[Image.Image], candidate_obj_ids: List[Any], overwrite: bool = False,
                            max_images=10, **kwargs) -> Dict:
        """Generate the final prediction by running the hierarchical selection process."""
        return self.hierarchical_selection(
            target_image, candidate_images, candidate_obj_ids, overwrite,
            max_images=max_images)

    def save_prediction(self, prediction: Dict):
        """Save the final prediction to a JSON file."""
        save_json(prediction, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))

    def load_prediction(self) -> Dict:
        """Load the saved prediction from the JSON file."""
        return load_json(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))

    def load_predicted_rendering(self):
        return Image.open(join_path(self.cfg.out_dir, "chosen_object.png"))

def make_obj_selector(cfg):
    ObjSelectorCls = {
        "yolo": ObjectSelector,
        "hierarchical": HierarchicalObjectSelector,
    }
    return ObjSelectorCls[cfg.obj_selector.name](create_task_config(cfg, "obj_selector"))

def get_candidate_objs(most_similar_object,
                       cam_view="frontview",
                       input_dir="datasets/partnet-mobility-v0/dataset",
                       rename=True):
    obj_type_ids = track_obj_types(input_dir, rename=rename)
    obj_ids = obj_type_ids[most_similar_object]
    # Prepare candidate images
    candidate_images = []
    valid_obj_ids = []  # Keep track of valid object IDs
    for obj_id in obj_ids:
        img_path = join_path(input_dir, str(obj_id),
                             f"robot_{cam_view}.png")
        if os.path.exists(img_path):
            candidate_images.append(Image.open(img_path))
            valid_obj_ids.append(obj_id)
        else:
            logging.warning(f"Image not found for object ID {obj_id}")
    return candidate_images, valid_obj_ids

def get_candidate_objs_from_categories(obj_categories, cam_view, input_dir="datasets/partnet-mobility-v0/dataset"):
    all_candidate_images = []
    all_candidate_obj_ids = []

    for obj_category in obj_categories:
        candidate_images, candidate_obj_ids = get_candidate_objs(
            obj_category, cam_view=cam_view, input_dir=input_dir
        )
        all_candidate_images.extend(candidate_images)
        all_candidate_obj_ids.extend(candidate_obj_ids)

    return all_candidate_images, all_candidate_obj_ids

def get_obj_image(path, frame_index=0):
    if ".mp4" in path:
        return get_frames_from_video(path, num_frames=5)[frame_index]
    else:
        # Assume it's an image
        return Image.open(path)

def save_obj_selector_viz(selected_obj, candidate_images, gt_image,
                          candidate_obj_ids,
                          out_dir):
    selected_index = int(selected_obj['selected_image'].split()[-1])
    selected_obj = candidate_obj_ids[selected_index]
    display_frames(candidate_images + [gt_image], cols=10,
                   save_file=join_path(out_dir, "candidates.png"),
                   titles=candidate_obj_ids + ["gt"])
    display_frames([candidate_images[selected_index], gt_image],
                   save_file=join_path(out_dir, "chosen_object.png"))

    closest_obj_id = candidate_obj_ids[selected_index]
    result = load_json(join_path(out_dir, "object_selector_result.json"))
    result["obj_id"] = closest_obj_id
    save_json(result, join_path(out_dir, "object_selector_result.json"))

