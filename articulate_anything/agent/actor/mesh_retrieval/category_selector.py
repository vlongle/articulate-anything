from articulate_anything.agent.agent import (
    Agent,
    AgentConfig,
)
import json
from articulate_anything.utils.utils import (
    save_json,
    load_json,
    join_path,
)
from articulate_anything.utils.viz import get_frames_from_video
from articulate_anything.utils.clip_utils import ClipModel
import os
import logging
from articulate_anything.utils.utils import create_task_config
from articulate_anything.utils.partnet_utils import track_obj_types
import numpy as np

OBJECT_DETECTOR_INSTRUCTION = \
    """
An affordance of an object is a property or a feature that allows for specific actions or interactions. For example, a drawer might be opened and close.
You'll be given some videos. For each video, please identify
    1. the object
    2. the affordance being explored in the video
Remember that while an object might have multiple affordances, we are strictly interested in the one being explored in the video.
The response should be in the format:

```json
{
    "object": {description of the object},
    "affordance": {description of the affordance}
}
```

Some helpful tips:

- strings must be enclosed in double quotes so that we can parse the response correctly
using `json.loads`
"""


class ObjectDetector(Agent):
    OUT_RESULT_PATH = "object_detector.json"

    def _make_system_instruction(self):
        system_instruction = OBJECT_DETECTOR_INSTRUCTION
        return system_instruction

    def _make_prompt_parts(self, video,
                           num_frames=5,
                           video_encoding_strategy="individual"):
        pil_frames = get_frames_from_video(video,
                                           num_frames=num_frames,
                                           video_encoding_strategy=video_encoding_strategy)
        prompt_parts = ["The video is\n"] + pil_frames
        return prompt_parts

    def parse_response(self, response, **kwargs):
        json_str = response.text.strip().strip('```json').strip()

        parsed_response = json.loads(json_str, strict=False)

        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))


class CategorySelector(Agent):
    OUT_RESULT_PATH = "category_selector.json"

    def __init__(self, cfg: AgentConfig):
        self.object_detector = ObjectDetector(
            create_task_config(cfg, "object_detector"))
        super().__init__(cfg)

    def _make_system_instruction(self):
        return "COMPOSITE SYSTEM."

    def generate_prediction(self, prompt, additional_prompt=None, gen_config=None,
                            clip_config={},
                            overwrite=False, **kwargs):
        if (
            os.path.exists(join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))
            and not overwrite
        ):
            logging.info(
                f"{self.__class__.__name__}: Prediction already exists. Skipping generation."
            )
            return

        if additional_prompt is not None:
            target_object = additional_prompt
            logging.info(
                f"Using additional prompt. Target object: {target_object}")
        else:
            self.object_detector.generate_prediction(
                video=prompt, gen_config=gen_config, overwrite=overwrite, **kwargs
            )
            affordance = self.object_detector.load_prediction()
            target_object = affordance['object']

        if not os.path.exists("partnet_obj_types.json"):
            obj_types = track_obj_types(rename=True)
            save_json(obj_types, "partnet_obj_types.json")

        candidate_objs = list(load_json("partnet_obj_types.json").keys())
        clip_model = ClipModel(**clip_config)

        # Calculate similarities
        similarities = [
            clip_model.cosine_similarity_text(
                target_object, candidate)
            for candidate in candidate_objs
        ]
        # Sort indices by similarity scores in descending order
        sorted_indices = np.argsort(similarities)[::-1]

        # Get top-k indices
        topk = self.cfg.category_selector.topk
        top_k_indices = sorted_indices[:topk]

        # Get top-k objects and scores
        most_similar_objects = [candidate_objs[i] for i in top_k_indices]
        similarity_scores = [float(similarities[i]) for i in top_k_indices]

        # Prepare the result
        result = {
            "target_object": target_object,
            "most_similar_objects": most_similar_objects,
            "similarity_scores": similarity_scores,
        }

        # Save the result
        save_json(result, join_path(self.cfg.out_dir, self.OUT_RESULT_PATH))

        return result
