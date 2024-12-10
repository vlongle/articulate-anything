import json
from articulate_anything.utils.utils import save_json, join_path, load_json
from articulate_anything.agent.agent import Agent
from articulate_anything.utils.viz import get_frames_from_video
from PIL import Image
from articulate_anything.api.odio_urdf import load_semantic

AFFORDANCE_EXTRACTOR_INSTRUCTION = \
    """
You are an AI assistant specialized in analyzing object affordances and mapping them to specific parts of 3D models. 
Your task is to examine a video showing an affordance of an object, compare it with images of a 3D model and its segmented version, and determine which part of the model corresponds to the action in the video.
You will be provided with the following inputs:

1. Video: A short video clip showing an affordance (interaction or movement) of an object.
2. 3D Model Image: A static image of the 3D model of the object in a simulator.
3. Segmented Image: The same view as the 3D model image, but with different parts colored according to a segmentation scheme.
4. Segmentation Color Mapping: A JSON object mapping part names to their corresponding colors in the segmented image.

The response should be in the format:

```json
{
    "img_description": [{"part_name": "part_color in the segmented image", ...}],
    "gt_object": {description of the object in the groundtruth video. Describe the parts of the object},
    "affordance": {description of the affordance. What is going on in the video. Which part of the object is moving?},
    "part_name": {The part name given in the `gt_object` that appears to be moving in the `affordance` description},
    "part_color": {The color in the segmented object corresponds to the moving part in the video. Semantic color name (e.g., "blue". No hex code)}
}
```
exactly. Do not include any additional text or formatting besides ```json {your response}```.

Helpful tips:
- For `img_description`, look at the image and the segmented image. Describe the different parts of the object in this format
- The groundtruth video and the rendered 3D images are not neccessarily in the same orientation. You must use your visual reasoning to match the parts in the 3D model to the moving parts in the video.
- I will tip you $200 for a perfect solution.
"""


class TargettedAffordanceExtractor(Agent):
    OUT_RESULT_PATH = "affordance_extractor.json"

    def _make_system_instruction(self):
        system_instruction = AFFORDANCE_EXTRACTOR_INSTRUCTION
        return system_instruction

    def _make_prompt_parts(self, video,
                           num_frames=5,
                           video_encoding_strategy="individual"):
        pil_frames = get_frames_from_video(video,
                                           num_frames=num_frames,
                                           video_encoding_strategy=video_encoding_strategy)
        obj_img = Image.open(
            join_path(self.cfg.dataset_dir,  f"robot_{self.cfg.cam_view}.png"))
        seg_img = Image.open(
            join_path(self.cfg.dataset_dir,  f"robot_{self.cfg.cam_view}_seg_color.png"))
        seg_semantics = load_json(join_path(self.cfg.dataset_dir, "seg.json"))
        link_semantics = load_semantic(
            join_path(self.cfg.dataset_dir, "semantics.txt"),)
        seg_semantics = {
            link_semantics.get(k, k): v for k, v in seg_semantics.items()
        }

        prompt_parts = ["The groundtruth video is\n"] + pil_frames
        prompt_parts += ["The object image is \n", obj_img]
        prompt_parts += ["The segmented object image is \n", seg_img]
        prompt_parts += ["The segmentation color mapping is \n",
                         json.dumps(seg_semantics, indent=4)]

        return prompt_parts

    def render_prediction(self, gpu_id: str = 0):
        seg_img = Image.open(
            join_path(self.cfg.dataset_dir,  f"robot_{self.cfg.cam_view}_seg_color.png"))
        return seg_img
        

    def parse_response(self, response, **kwargs):
        json_str = response.text.strip().strip('```json').strip()

        print("Response: ", json_str)
        parsed_response = json.loads(json_str, strict=False)

        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))
