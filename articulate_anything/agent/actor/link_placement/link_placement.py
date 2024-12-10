from articulate_anything.api.odio_urdf import (
    save_link_states,
    compile_python_to_urdf,
)
from articulate_anything.utils.metric import average_link_difference
from articulate_anything.preprocess.preprocess_utils import mask_urdf
from articulate_anything.preprocess.preprocess_partnet import (
    get_urdf_file,
    render_object,
)
from PIL import Image
from articulate_anything.agent.agent import Agent
from articulate_anything.utils.utils import (
    file_to_string,
    string_to_file,
    load_json,
    join_path,
    save_json,
)
from articulate_anything.utils.prompt_utils import (
    extract_code_from_string,
    get_n_examples_from_python_code,
)
import json
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_retrieval import make_links_from_json

LINK_PLACEMENT_INSTRUCTION = \
    """
## General Instructions

We'd like to build a simple model of an object consisting of object parts (links). The links represent parts of an object and are already defined using meshes.
We will you the link summary with this format
```text
object_id: {int}
    Robot Link Summary:
    - base
    {additional links}
```

{optional_image_inst}

Then, write a function named `partnet_{object_id}` that will make API calls to place the links of the object in the correct position. This function has the signature `def partnet_{object_id}(input_dir, links) -> Robot:`.
The function generally has this structure

```python

def partnet_{object_id}(input_dir, links) -> Robot:
    pred_robot = Robot(input_dir=input_dir)
    pred_robot.add_link(links["base"])
    pred_robot.add_link(links[{some_link_name}])
    pred_robot.add_joint(
    Joint(
        "base_to_{some_link_name}",
        Parent("base"),
        Child("{some_link_name}"),
        type="fixed",
    ))

    # some more placement
    pred_robot.add_link(links[{some_other_link_name}])
    pred_robot.place_relative_to(
        "{some_other_link_name}", "{some_link_name}", placement={placement}, 
        ...
    )
    # ...
    return pred_robot
```
"""


PLACE_RELATIVE_TO_SIGNATURE = \
    """
## Placement API

The signature for `place_relative_to` is
```
    def place_relative_to(
        self,
        child_link_name: str,
        parent_link_name: str,
        placement: str = "above",
        clearance: float = 0.0,
        snap_to_place: bool = True,
        overwrite: bool = True,
    ) -> None:
        # Places the child link relative to the parent link's bounding box with a specified placement and clearance,
        # accounting for the relative orientation of the links.

        # Args:
        #     child_link_name (str): Name of the child link.
        #     parent_link_name (str): Name of the parent link.
        #     placement (str): Placement direction. Options are:
        #         "above", "above_inside", "below", "below_inside",
        #         "left", "left_inside", "right", "right_inside",
        #         "front", "front_inside", "back", "back_inside",
        #         "inside"
        #     clearance (float): Distance to maintain between the links.
        #     snap_to_place (bool): Whether to snap the child to place after initial positioning.
        #     use_bounding_box_center (bool): Whether to use bounding box center for positioning.
        #     overwrite (bool): Whether to overwrite the existing joint transformation or add to it.

```
"""
# Use `snap_to_place` judiciously. For example,
# for switch of faucet, you might want to set `snap_to_place=True` for the switch to get attached tightly to the base. However, for something like reading glasses
# with two legs placed left and right, you might want to set `snap_to_place=False` to fix the legs in the correct position.

LINK_PLACEMENT_ENDING = \
    """

## Ending Instructions

Your response must follow this format
```text
   object: {description of the object}
```
```python
from articulate_anything.api.odio_urdf import *
# code of the partnet_{object_id} function
```
Some helpful tips:

- Avoid doing `place_relative_to({some_link}, "base", ...)` as "base" is a special link that does not have a bounding box. In general, stick to
`add_joint(Joint({name}, Parent("base"), Child({some_link}), type="fixed"))` to attach links to the base.
- Generally, there might be multiple parts with the same name. The first part is always labeled simply as the `part_name`. Subsequent
parts are labeled as `part_name_2`, `part_name_3` ect.
- Make sure that you examine the names of links that are given to you to avoid running into syntax error of calling
non-existent links i.e., the link names in the Python code must match exact to the names under `Robot Link Summary`. Generally, each word in the
link name is separated by an underscore `_`.
"""

HELPER_CODE = \
    """
## Helper code

We have some helper functions that might be useful for you.
``` python
    def get_bounding_boxes(self, link_names: List[str] = None, include_dim: bool = False) -> Dict[str, Any]:
        # Gets the AABB bounding boxes for specified links on a robot in PyBullet.
        #
        # Args:
        #     link_names (list): A list of strings representing the names of the links to get bounding boxes for.
        #     include_dim (bool): If True, include the dimensions of the bounding box. The dimensions are length,
        #     width, and height.
        #
        # Returns:
        #     dict: A dictionary mapping link names to tuples of (aabb_min, aabb_max) coordinates.
```
"""


class LinkPlacementActor(Agent):
    OUT_RESULT_PATH = "link_placement.py"

    def _get_code_example(self):
        if self.cfg.modality == "text":
            # Need to have finer placement because the mesh are all re-centered to the origin
            example = 'articulate_anything/examples/link_placement_desc_examples.py'
        else:
            # Use the off-centered meshes from PartNet-Mobility so the placement needs not be so precise
            example = 'articulate_anything/examples/link_placement_examples.py'

        example = file_to_string(example)

        # randomly pick a subset of examples
        example = get_n_examples_from_python_code(
            example, self.cfg.in_context.num_examples)

        return '```python\n' + example + '\n```'

    def _make_system_instruction(self):
        """
        ## General Instructions
        {...}

        ## Examples
        {...}

        ## Helper code
        {...}

        ## Placement API
        {...}

        ## Ending Instructions
        {...}
        """
        if self.cfg.link_actor.mode == "image":
            optional_image_inst = "We will also provide you a groundtruth image of the object with the correct placement of the parts. The image is a PIL image object. Please study the image carefully to understand the correct placement of the links"
        elif self.cfg.link_actor.mode == "text":
            optional_image_inst = ""
        else:
            raise ValueError(
                f"Invalid mode {self.cfg.link_actor.mode}. Must be either 'image' or 'text'")

        system_instruction = LINK_PLACEMENT_INSTRUCTION.replace(
            "{optional_image_inst}", optional_image_inst
        )

        example = self._get_code_example()
        system_instruction += '## Examples\n\n Here are some examples of creating various objects using our API\n'
        system_instruction += example

        system_instruction += HELPER_CODE
        system_instruction += PLACE_RELATIVE_TO_SIGNATURE

        system_instruction += LINK_PLACEMENT_ENDING

        return system_instruction

    def parse_response(self, response, **kwargs):
        string_to_file(response.text, join_path(
            self.cfg.out_dir, "response.txt"))
        function_definition = extract_code_from_string(response.text)
        string_to_file(function_definition, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))

    def _make_prompt_parts(self, link_summary_path=None,
                           gt_image_path=None,
                           link_pred_path=None,
                           feedback_path=None,**kwargs):
        if gt_image_path is None and self.cfg.link_actor.mode == "image":
            gt_image_path = join_path(
                self.cfg.dataset_dir, f"robot_{self.cfg.cam_view}.png")
        if feedback_path is not None and link_pred_path is not None:
            return self._make_prompt_parts_retry(gt_image_path, link_pred_path, feedback_path)
        else:
            return self._make_prompts_parts_init(gt_image_path, link_summary_path)

    def _make_prompts_parts_init(self,  gt_image_path, link_summary_path=None):
        if link_summary_path is None:
            link_summary_path = join_path(
                self.cfg.dataset_dir, "link_summary.txt")

        link_summary = file_to_string(link_summary_path)
        link_summary_instruction = 'Link summary is \n\n' + link_summary

        if self.cfg.link_actor.mode == "image":
            gt_img = Image.open(gt_image_path)
            prompt_parts = [
                "The groundtruth image is shown below\n",  gt_img, link_summary_instruction]
        else:
            prompt_parts = [link_summary_instruction]
        return prompt_parts

    def _make_prompt_parts_retry(self,
                                 gt_image_path,
                                 link_pred_path=None,
                                 feedback_path=None):
        assert self.cfg.link_actor.mode == "image", "Retrying is only supported for image mode"
        # for `image`` mode only
        gt_img = Image.open(gt_image_path)
        prompt_parts = ["The groundtruth image is shown below\n", gt_img]

        candidate_function = file_to_string(link_pred_path)
        feedback = '```json\n' + \
            json.dumps(load_json(feedback_path), indent=4) + '\n```'

        prompt_parts += ["Previously, you wrote the following function\n" +
                         '```python\n' + candidate_function + '\n```']
        prompt_parts += ["\nHere's the feedback\n" + feedback]
        prompt_parts += ["\nPlease take the feedback into account and rewrite the function"]
        return prompt_parts

    def get_links(self):
        if self.cfg.modality == "text":
            links = make_links_from_json(self.cfg.link_actor.new_box_layout)
        else:
            # masked gt robot urdf then reconstruct with our VLM
            robot = mask_urdf(get_urdf_file(self.cfg.dataset_dir))
            links = robot.get_links()
        return links

    def render_prediction(self, gpu_id: str):
        pred_python_file = join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH)

        links = self.get_links()
        if self.cfg.modality == "text":
            input_dir = self.cfg.out_dir
        else:
            input_dir = self.cfg.dataset_dir
        urdf_file = compile_python_to_urdf(pred_python_file, input_dir=input_dir,
                                           links=links)
        # render an image of the prediction
        render_object(urdf_file, gpu_id, self.cfg.simulator,
                      "stationary")

    def compute_gt_diff(self):
        gt = save_link_states(get_urdf_file(self.cfg.dataset_dir))
        pred = save_link_states(get_urdf_file(self.cfg.out_dir))
        link_diff = average_link_difference(
            gt, pred,  semantic_file=join_path(self.cfg.dataset_dir, "semantics.txt"))
        save_json(link_diff, join_path(self.cfg.out_dir, "link_diff.json"))
        return link_diff

    def load_predicted_rendering(self):
        return join_path(self.cfg.out_dir, f"robot_{self.cfg.cam_view}.png")
