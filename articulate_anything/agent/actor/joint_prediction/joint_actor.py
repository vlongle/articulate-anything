from articulate_anything.utils.cotracker_utils import make_cotracker
import glob
import os
from articulate_anything.preprocess.preprocess_utils import mask_urdf
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_retrieval import make_links_from_json
from articulate_anything.api.odio_urdf import (
    compile_python_to_urdf,
    save_joint_states,
    process_urdf,
)
from articulate_anything.preprocess.preprocess_partnet import (
    get_urdf_file,
    render_object,
)
from articulate_anything.agent.agent import Agent
from articulate_anything.utils.utils import (
    file_to_string,
    string_to_file,
    join_path,
    create_task_config,
    save_json,
)
from PIL import Image
from articulate_anything.utils.prompt_utils import (
    remove_lines_containing,
    extract_code_from_string,
    get_n_examples_from_python_code,
)
from articulate_anything.utils.viz import get_frames_from_video, convert_mp4_to_gif
from articulate_anything.agent.multimodal_incontext_agent import InContextExampleModel
from typing import Optional
from articulate_anything.utils.metric import compute_joint_diff
from articulate_anything.utils.partnet_utils import get_joint_semantic
from omegaconf import DictConfig, OmegaConf

JOINT_PREDICTION_GENERAL_SYSTEM_INSTRUCTION = """
## General Instructions

You're a robotics expert whose job is to complete the articulation of an object.

The links represent parts of an object and are already defined using meshes. Further, the placement of various object parts (links) have also already been done for you.

You're only responsible for creating the joint for the object. Particularly, we will provide you the function `partnet_{object_id}` that has object placement, you're responsible to fill in this function to create joints.

Note that your `partnet_{object_id}` function must begin with the code that we provide you for the object placement. Then you must include this comment
"# ====================JOINT PREDICTION====================" before you start writing the code for the joint prediction. Please ensure that the code
runs without any errors. DO NOT MODIFY the link placement code. Make sure to use the same link names as provided in the link placement code. Make sure to include the import statement `from articulate_anything.api.odio_urdf import *`.

We have some helper functions that might be useful for you.
```python
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

    def make_revolute_joint(self, child_link_name: str, parent_link_name: str, global_axis: List[float], lower_angle_deg: float, upper_angle_deg: float, force_overwrite: bool = True, pivot_point: Optional[List[float]] = None) -> None:
        # Creates or updates a revolute joint between the specified child and parent links.

        # Args:
        #     child_link_name (str): Name of the child link.
        #     parent_link_name (str): Name of the parent link.
        #     global_axis (list): The rotation axis of the joint in the world frame.
        #     lower_angle_deg (float): The lower joint angle limit in degrees.
        #     upper_angle_deg (float): The upper joint angle limit in degrees.
        #     force_overwrite (bool, optional): If True, overwrite existing joint.
        #     pivot_point (list, optional): The pivot point for the rotation in the world frame.

    def make_prismatic_joint(self, child_link_name: str, parent_link_name: str, global_lower_point: List[float], global_upper_point: List[float], force_overwrite: bool = True) -> None:
        # Creates or updates a prismatic joint between the specified child and parent links.

        # Args:
        #     child_link_name (str): Name of the child link.
        #     parent_link_name (str): Name of the parent link.
        #     global_lower_point (list): The global coordinates of the lower point.
        #     global_upper_point (list): The global coordinates of the upper point.
        #     force_overwrite (bool): If True, overwrite existing joint.
```

These functions are the class functions of `Robot`. For example, you can call `pred_robot.get_bounding_boxes(...)`. The following function(s) are helper functions
and are not bound to any particular class.

```python
    def compute_aabb_vertices(aabb_min: Union[List[float], np.ndarray], aabb_max: Union[List[float], np.ndarray]) -> np.ndarray:
        # Computes the 8 vertices of an axis-aligned bounding box. x-axis is front back, y-axis is left right, z-axis is up down.

        # Args:
        # aabb_min (list or numpy.ndarray): A list or array of 3 elements representing the minimum x, y, and z coordinates.
        # aabb_max (list or numpy.ndarray): A list or array of 3 elements representing the maximum x, y, and z coordinates.

        # Returns:
        # numpy.ndarray: An 8x3 array where each row represents the coordinates of a vertex.
        # The vertices are ordered as follows:
        # 0: Back-Left-Bottom (BLB)
        # 1: Back-Right-Bottom (BRB)
        # 2: Front-Left-Bottom (FLB)
        # 3: Front-Right-Bottom (FRB)
        # 4: Back-Left-Top (BLT)
        # 5: Back-Right-Top (BRT)
        # 6: Front-Left-Top (FLT)
        # 7: Front-Right-Top (FRT)

```   
"""


JOINT_PREDICTION_ENDING_SYSTEM_INSTRUCTION = """
## Ending Instructions

**Retry instruction**: We might also provide you some feedback on your previously generated output.
In that case, you **must** take the feedback into account when rewriting the previous function.

Important points:

- Make sure that for revolute joint, lower_angle_deg < upper_angle_deg.
- If you're given a video or image of the groundtruth object joint, make sure to study the provided input carefully to understand the correct behavior of the object joint.
- Do not make joints whose direct parent is `base`. For example, `make_revolute_joint("link1", "base",...)` is not a good practice. Instead,
    consider making joints between two non-base links. 
- Joint axis order is [x, y, z]:
    - x : forward -- positive x, backward -- negative x
    - y: right -- positive y, left -- negative y
    - z: up -- positive z, down -- negative z
- When applicable, pay close attention to the relationship between pivot points and joint axis as demonstrated in the provided examples.
- When feedback is provided, pay close attention to `improvement_suggestion` and modify the code accordingly.
- Only include this import statement `from articulate_anything.api.odio_urdf import *`. Remove any other import statements.
- Your return format is
```text
{Describe what is going on in the input image/video if provided}
```
```python
{your code here}
```

"""

JOINT_PREDICTION_VISUAL_INPUT_INSTRUCTION = """
## {visual_input} Input 

We will also give you a {visual_input} of the object joint in action. Please study the image carefully to understand the correct behavior of the object joint.
"""


JOINT_PREDICTION_COTRACKER_INSTRUCTION = """
## CoTracker Motion Tracing

We also use a motion tracker algorithm (CoTracker) to highlight the moving part in the videos. Pay close attention to the motion traces annotated in the videos to gain
information about the joint type, axis, origin, and limit.

Important points:

- Ignore traces in the background.
- Sometimes, cotracker might fail to capture traces of moving parts especially when the parts is moving forward and backward. Do your best to detect motion on your own.
- Traces moving in an arc indicates a revolute joint while linear traces indicate a prismatic joint.

"""


JOINT_PREDICTION_TARGETTED_AFFORDANCE_INSTRUCTION = """
## Targetted Affordance

- You must only make the kinematic joints targetted in the groundtruth input. We will provide you the targetted affordance. You must then include the comment
```
Object: {description}
Targetted affordance: {description}
```
to the code after the link summary as seen in our provided example codes.
"""


JOINT_PREDICTION_ALL_JOINTS_INSTRUCTION = """
## All Joints

- You must make **all** possible kinematic joints for the object. For example, if the object contains a lid and a bunch of buttons, you must create joints for both the lid and the buttons.
"""


class JointPredictionActor(Agent):
    OUT_RESULT_PATH = "joint_pred.py"

    def _get_code_example(self):
        if self.cfg.joint_actor.targetted_affordance:
            example = "articulate_anything/examples/joint_examples_targetted.py"
        else:
            example = "articulate_anything/examples/joint_examples_all.py"

        example = file_to_string(example)

        # randomly pick a subset of examples
        example = get_n_examples_from_python_code(
            example, self.cfg.in_context.num_examples)

        return '```python\n' + example + '\n```'

    def _make_system_instruction(self):
        """
        ## General Instructions
        {...}

        {## Examples. Only for `basic` prompting}
        {...}

        ## Visual Input or Text
        {For text, we'll also include exampes.py}

        {## CoTracker Motion Tracing}
        {If video and use_cotracker is True}

        ## All Joints or Targetted Affordance
        {...}

        ## Ending Instructions
        {...}

        """
        system_instruction = JOINT_PREDICTION_GENERAL_SYSTEM_INSTRUCTION

        if (self.cfg.joint_actor.mode == "text" or self.cfg.joint_actor.type == "basic") and not self.cfg.joint_actor.type == "llama_finetune":
            system_instruction += "## Examples \n\n We also provide you with some examples. Examine them CAREFULLY:\n\n"
            system_instruction += self._get_code_example()

        if self.cfg.joint_actor.mode != "text":
            system_instruction += JOINT_PREDICTION_VISUAL_INPUT_INSTRUCTION.replace(
                "{visual_input}", self.cfg.joint_actor.mode.capitalize())

        if self.cfg.joint_actor.mode == "video" and self.cfg.joint_actor.use_cotracker:
            system_instruction += JOINT_PREDICTION_COTRACKER_INSTRUCTION

        if self.cfg.joint_actor.targetted_affordance:
            system_instruction += JOINT_PREDICTION_TARGETTED_AFFORDANCE_INSTRUCTION
        else:
            system_instruction += JOINT_PREDICTION_ALL_JOINTS_INSTRUCTION

        system_instruction += JOINT_PREDICTION_ENDING_SYSTEM_INSTRUCTION
        return system_instruction

    def _load_func(self, candidate_function):
        return remove_lines_containing(file_to_string(candidate_function), "ffordance")

    # ======================================
    # ============== No retry ===============

    def _make_text_prompt_parts(self, link_placement_path):
        prompt_parts = [
            "The link placement code is\n"
            + "```python\n"
            + self._load_func(link_placement_path)
            + "\n```"
        ]
        return prompt_parts

    def _make_image_prompt_parts(self, image_path):
        if "mp4" in image_path:
            frames = get_frames_from_video(
                image_path,
                **self.cfg.video_encoding,)
            img = frames[0]
        else:
            img = Image.open(image_path)
        prompt_parts = [
            "The groundtruth image is\n", img]
        return prompt_parts
    # ======================================

    def _make_video_prompt_parts(self, video_path):
        gt_frames = get_frames_from_video(
            video_path,
            **self.cfg.video_encoding,
        )
        prompt_parts = [
            "The groundtruth video is\n",] + gt_frames
        return prompt_parts

    def _make_prompt_parts(self,
                           link_placement_path: os.PathLike,
                           gt_input: Optional[os.PathLike] = None,
                           feedback: Optional[str] = None,
                           targetted_affordance: Optional[str] = None,
                           candidate_function_path: Optional[os.PathLike] = None,
                           **kwargs):
        if feedback is not None:
            return self._make_video_prompt_parts_retry(gt_input, candidate_function_path,
                                                       feedback, **kwargs)

        prompt_parts = []
        if self.cfg.joint_actor.mode == "image":
            assert gt_input is not None, "GT input is required for image modality"
            prompt_parts.extend(self._make_image_prompt_parts(gt_input))
        elif self.cfg.joint_actor.mode == "video":
            assert gt_input is not None, "GT input is required for video modality"
            prompt_parts.extend(self._make_video_prompt_parts(gt_input))

        if self.cfg.joint_actor.targetted_affordance:
            assert targetted_affordance is not None, "Targetted affordance is required"
            prompt_parts.append("The targetted affordance is\n"
                                + f"`{targetted_affordance}`")

        prompt_parts += self._make_text_prompt_parts(link_placement_path)

        return prompt_parts

    def _make_video_prompt_parts_retry(self, gt_video: os.PathLike,
                                       candidate_function_path: os.PathLike,
                                       feedback: str,
                                       **kwargs):
        # only for video modality
        assert self.cfg.joint_actor.mode == "video", "Joint pred retry only for video modality"

        prompt_parts = self._make_video_prompt_parts(gt_video)
        prompt_parts += [
            "Previously, you wrote the following function\n"
            + "```python\n"
            + file_to_string(candidate_function_path)
            + "\n```"
        ]
        prompt_parts += ["\nHere's the feedback\n" + feedback]
        prompt_parts += [
            "\nPlease examine the original function provided and the feedback carefully. Then, modify that function to address the feedback."
        ]
        return prompt_parts

    def parse_response(self, response, **kwargs):
        string_to_file(response.text, join_path(
            self.cfg.out_dir, "response.txt"))
        function_definition = extract_code_from_string(response.text)
        string_to_file(function_definition, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))

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
        # render videos
        render_object(urdf_file, gpu_id, self.cfg.simulator,
                      "move")

    def load_predicted_rendering(self, use_gif=False, joint_name=None):
        if self.cfg.modality == 'partnet' and self.cfg.joint_actor.targetted_semantic_joint:
            video = join_path(self.cfg.out_dir,
                              f"video_{self.cfg.joint_actor.targetted_semantic_joint}_{self.cfg.cam_view}.mp4")
        else:
            videos = [f for f in os.listdir(
                self.cfg.out_dir) if f.endswith(".mp4") and not f.startswith("aug")]
            print("BEFORE FILTER VIDEO", videos)
            if joint_name:
                videos = [v for v in videos if joint_name in v]
                print("AFTER FILTER VIDEO", videos)
            video = videos[0]
            video = join_path(self.cfg.out_dir, video)
        # apply cotracker to the predicted video
        if self.cfg.joint_actor.use_cotracker and self.cfg.modality != "text":
            # cotracker_model = make_cotracker(self.cfg)
            cfg = OmegaConf.create(self.cfg)
            cfg.cotracker.mode = "offline"

            cotracker_model = make_cotracker(cfg)
            seg_mask_path = join_path(
                self.cfg.dataset_dir, f"robot_{self.cfg.cam_view}_seg.png")
            if not os.path.exists(seg_mask_path):
                temp_cfg = OmegaConf.create(cfg)
                temp_cfg.simulator.use_segmentation = True
                temp_cfg.simulator.ray_tracing = False
                temp_cfg.simulator.object_white = True
                render_object(get_urdf_file(self.cfg.dataset_dir), str(self.cfg.gpu_id),
                              temp_cfg.simulator,
                              "stationary")

            cotracker_model.forward(video, seg_mask_path=seg_mask_path,
                                    **self.cfg.cotracker)
            video = join_path(
                os.path.dirname(video), f"aug_{os.path.basename(video)}")

        if use_gif:
            gif = convert_mp4_to_gif(video, video.replace(".mp4", ".gif"),)
            video = video.replace(".mp4", ".gif")
        return video

    def compute_gt_diff(self):
        gt = save_joint_states(get_urdf_file(self.cfg.dataset_dir))
        pred = save_joint_states(get_urdf_file(self.cfg.out_dir))
        joint_semantics = get_joint_semantic(self.cfg.dataset_dir)
        joint_diffs = compute_joint_diff(
            gt, pred, joint_semantics=joint_semantics,
            joint_name=self.cfg.joint_actor.targetted_joint)
        save_json(joint_diffs, join_path(self.cfg.out_dir, "joint_diff.json"))
        return joint_diffs


    def get_all_joint_names(self):
        robot = process_urdf(get_urdf_file(self.cfg.out_dir))
        joints = robot.get_manipulatable_joints()
        print("Joints:", joints)
        print("joint.name: joint.type", {joint.name: joint.type for joint in joints.values()})

        return joints.keys()


class JointPredictionMultiModalExamples(InContextExampleModel, JointPredictionActor):
    def _make_system_instruction(self):
        return JointPredictionActor._make_system_instruction(self)

    def get_example_paths(self):
        # paths under `examples_dir/{obj_id}/{joint_id}`
        pattern = join_path(self.cfg.joint_actor.examples_dir, "*", "*")
        return [path for path in glob.glob(pattern) if os.path.isdir(path)]

    def _format_content(self, *args, **kwargs):
        joint_formatter = JointPredictionActor(
            create_task_config(self.cfg, "temp"))
        content = joint_formatter._make_prompt_parts(*args, **kwargs)
        if "expected_joint_pred_path" not in kwargs:
            return content
        joint_pred_func = file_to_string(kwargs["expected_joint_pred_path"])
        content.append(
            f"The correct joint prediction code is\n```python\n{joint_pred_func}\n```"
        )
        return content

    def _extract_example_kwargs(self, example_path):
        joint_id = os.path.basename(example_path)
        gt_video_path = join_path(example_path, (
            f"{'aug_' if self.cfg.joint_actor.use_cotracker else ''}video_{joint_id}_{self.cfg.cam_view}.mp4"
        ))
        link_placement_path = join_path(example_path, "link_placement.py")
        targetted_affordance = file_to_string(join_path(
            example_path, "targetted_affordance.txt"
        ))
        expected_joint_pred_path = join_path(example_path, "joint_pred.py")

        return {
            'link_placement_path': link_placement_path,
            'gt_input': gt_video_path,
            'targetted_affordance': targetted_affordance,
            'expected_joint_pred_path': expected_joint_pred_path,
        }

    def _make_prompt_parts(self, link_placement_path, feedback=None, **kwargs):
        # only use multi-modal in-context prompting for the first iteration.
        # use basic prompting for retry.
        if feedback is None:
            return InContextExampleModel._make_prompt_parts(self, link_placement_path, **kwargs)
        else:
            return JointPredictionActor._make_prompt_parts(self, link_placement_path, feedback, **kwargs)


def make_joint_actor(cfg):
    JointPredCls = {
        "basic": JointPredictionActor,
        "incontext": JointPredictionMultiModalExamples,
    }
    return JointPredCls[cfg.joint_actor.type]
