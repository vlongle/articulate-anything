
import os
import glob
from articulate_anything.agent.multimodal_incontext_agent import InContextExampleModel
from articulate_anything.utils.viz import get_frames_from_video
from articulate_anything.utils.utils import (
    file_to_string,
    join_path,
    save_json,
    load_json,
    create_task_config,
)
import logging
from articulate_anything.agent.agent import Agent
import json
from articulate_anything.api.odio_urdf import get_semantic_joint_id

CRITIC_INSTRUCTION = """
## General Instructions

You are a visual critic expert whose job is to assess the realism of a joint prediction of a 3D model.

You will analyze a candidate function `partnet_{object_id}`. Assess how realistic this model is compared to the ground truth.

You will see two videos: first the ground truth, then the prediction.

Compare these videos and provide feedback on the prediction. Use this format:

```json
{
"gt_description": {describe the gt video},
"pred_description": {describe the prediction video},
"candidate_function_description": {describe the candidate function},
"failure_reason": {one of these "success", "joint_type", "joint_axis", "joint_origin", "joint_limit"},
"improvement_suggestion": {suggestion to improve the prediction},
"realism_rating": {0-10},
}
```

Be concise and specific. When writing the description, compare the predicted video to the ground truth and analyze the `candidate_function` to identify issues.

Important points:

- Evaluate only the joint prediction, not link placement.
- Compare videos first, then examine the candidate function.
- Rate highly if the prediction closely matches the ground truth.
- Identify problems using this checklist, focusing on the most significant error:
  1. Incorrect joint type (e.g., revolute instead of prismatic): Rate 0
  2. Wrong joint axis (e.g., x-axis instead of y-axis): Rate 1
  3. Incorrect joint origin (for **revolute joints** only): Rate 2
  4. Incorrect joint limit (for **revolute joints** only; e.g, the door is opening inward instead of outward): Rate 3
  5. No errors: Rate above 5, mark as "success"
- Your `realism_rating` must match the `failure_reason` according to the ratings specified above.
- Joint axis order is [x, y, z]: 
    - x : forward -- positive x, backward -- negative x
    - y: right -- positive y, left -- negative y
    - z: up -- positive z, down -- negative z
- Use the `candidate_function` to confirm your diagnosis.
- Analyze the videos frame-by-frame if needed. Describe the motion clearly, using terms like "rotates", "slides", or "pivots" to convey the joint behavior.
- **Important**: the groundtruth video might not have the same texture as the prediction video e.g., the gt might be in-the-wild video captured by a phone while prediction is 3D model rendered in a 
physics simulator. Thus, you must correctly describe the motion of the object in the video and compare it with the prediction.
- We will use `json.loads()` to parse your response. Make sure that your response is exactly ```json {your response}```, nothing more, nothing less.
"""


CRITIC_COTRACKER_TRACE = """
## CoTracker Motion Tracing

We also use a motion tracker algorithm (CoTracker) to highlight the moving parts in the videos. Pay close attention to the motion traces annotated in the videos to gain
information about the joint type, axis, origin, and limit.

Important points:

- Ignore traces in the background.
- Sometimes, cotracker might fail to capture traces of moving parts especially when the parts is moving forward and backward. Do your best to detect motion on your own.
- Traces moving in an arc indicates a revolute joint while linear traces indicate a prismatic joint.
- I will tip $200 for each correct analysis of the motion traces.
"""

JOINT_CRITIC_EXAMPLES = """
```json
{"gt_description": "The gt video shows the window pane opens by sliding horizontally along the y-axis in a linear motion.",
 "pred_description": "The pred video shows the window pane opens by sliding horizontally along the frame in a linear motion.",
 "candidate_function_description": "The `candidate_function` has `make_prismatic_joint` and axis is [0, -bbox[`width`], 0], which is horizontal (y-axis) and correct",
 "failure_reason": "success",
 "improvement_suggestion": "None",
 "realism_rating": 10
}
```
```json
{
"gt_description": "The gt video shows the window pane opens by sliding horizontally along the y-axis in a linear motion",
"pred_description": "The pred video shows the window opens by rotating up in an arc.",
"candidate_function_description": "The `candidate_function` has `make_revolute_joint`, which is incorrect.",
"failure_reason": "joint_type",
"improvement_suggestion": "Consider changing the joint type to prismatic to allow sliding motion",
"realism_rating": 0
}
```
```json
{
"gt_description": "The gt video shows the window pane opens by slides horizontally along the y-axis in a linear motion",
"pred_description": "The pred video looks static. We need to see the `candidate_function` to understand the issue.",
"candidate_function_description": "We have `make_prismatic_joint` which is correct but the axis `upper_point=[bbox[`length`], 0, 0]` is along x-axis (front/back) instead of y-axis (left/right).",
"failure_reason": "joint_axis",
"improvement_suggestion": "Consider changing `joint_axis` to slide along the y-axis",
"realism_rating": 1
}
```
```json
{"gt_description":   "The gt video shows the door opens by rotating forward along the vertical axis (z) while the **RIGHT** part fixed to the body",
 "pred_description":  "The pred video shows the door opens by rotating forward along the vertical axis (z) while the **LEFT** part fixed to the body",
 "candidate_function_description": "The `candidate_function` has `make_revolute_joint` and axis is [0, 0, 1], which is vertical (z-axis) and correct. The pivot point is set to Bottom-Front-**RIGHt** which is incorrect. Note that in the groundtruth, the left part of the door is fixed to the body.",
 "failure_reason": "joint_origin",
 "improvement_suggestion": "Try changing the pivot to the left side of the door (e.g. Front-**lEFT**-Bottom) to make the joint more like the groundtruth video.",
 "realism_rating": 2
}
```
```json
{"gt_description":   "The gt video shows the door opens outward. The door rotates outward along the vertical axis (z) while the left part fixed to the body",
 "pred_description":  "The pred video shows the door opens by rotating **inward** along the vertical axis (z) while the left part fixed to the body. The prediction doesn't look similar to the groundtruth as the door appears to be moving inward into the body instead of outward.",
 "candidate_function_description": "The `candidate_function` has `make_revolute_joint` and axis is [0, 0, 1], which is vertical (z-axis). The pivot point is set to Front-Left-Bottom which is correct, keeping the left part of the door fixed to the body. However, the door opens inward instead of outward so this is a joint limit issue.",
 "failure_reason": "joint_limit",
 "improvement_suggestion": "In our convention, left is negative so in order to open outward, the axis must be negative: i.e. [0, 0, -1]. The current axis is [0, 0, 1]. Try negating it",
 "realism_rating": 3
}
```

Important points:

- These examples are far from exhaustive. Use them as a guide to evaluate the realism of the joint.
- Use your own judgement to evaluate. Reason step-by-step.

"""


class JointCritic(Agent):
    OUT_RESULT_PATH = "joint_critic.json"

    def _make_system_instruction(self):
        """
        ## General Instructions
        {...}
        {## CoTracker Motion Tracing}
        {If use_cotracker is True}

        {## Examples. Only for `basic` prompting}
        {...}
        """
        system_instruction = CRITIC_INSTRUCTION
        if self.cfg.joint_critic.use_cotracker:
            system_instruction += CRITIC_COTRACKER_TRACE

        if self.cfg.joint_critic.type == "basic":
            system_instruction += (
                "\n## Examples \n \n Here are some examples of the evaluation of the realism of various joints\n"
                + JOINT_CRITIC_EXAMPLES
                + "\n")
        return system_instruction

    def _make_prompt_parts(
        self,
        candidate_function_path: os.PathLike,
        gt_video_path: os.PathLike,
        pred_video_path: os.PathLike,
        num_frames=5,
        video_encoding_strategy="individual",
    ):
        gt_video = get_frames_from_video(
            gt_video_path,
            num_frames=num_frames,
            video_encoding_strategy=video_encoding_strategy,
            # width=self.cfg.simulator.camera_params.width,
            # height=self.cfg.simulator.camera_params.height,
        )
        pred_video = get_frames_from_video(
            pred_video_path,
            num_frames=num_frames,
            video_encoding_strategy=video_encoding_strategy,
            # width=self.cfg.simulator.camera_params.width,
            # height=self.cfg.simulator.camera_params.height,
        )
        candidate_function = file_to_string(candidate_function_path)
        candidate_function_text = (
            "The candidate function is:\n"
            + "```python\n"
            + candidate_function
            + "\n```"
        )
        prompt_parts = ["The groundtruth video is:\n"] + gt_video
        prompt_parts += ["The prediction video is:\n"] + pred_video
        prompt_parts += [candidate_function_text]
        return prompt_parts

    def parse_response(self, response, realign_score=True, **kwargs):
        # Extract the JSON string from the response text
        json_str = response.text.strip().strip("```json").strip()

        # Parse the JSON string into a dictionary
        parsed_response = json.loads(json_str, strict=False)

        if realign_score:
            scores = {
                "success": 10,
                "joint_type": 0,
                "joint_axis": 1,
                "joint_origin": 2,
                "joint_limit": 3,
            }
            parsed_response["realism_rating"] = scores[
                parsed_response["failure_reason"]
            ]
            if int(parsed_response["realism_rating"]) > 5:
                parsed_response["failure_reason"] = "success"

        logging.info(f"Joint critic response: {parsed_response}")

        # Save the parsed response to a JSON file
        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))


class JointCriticMultiModalExamples(InContextExampleModel, JointCritic):
    def _make_system_instruction(self):
        return JointCritic._make_system_instruction(self)

    def get_example_paths(self):
        # paths under `examples_dir/{failure_reason}/{obj_id}/{joint_id}`
        pattern = join_path(
            self.cfg.joint_critic.examples_dir, "*", "*", "*")
        return [path for path in glob.glob(pattern) if os.path.isdir(path)]

    def _format_content(self, *args, **kwargs):
        joint_formatter = JointCritic(
            create_task_config(self.cfg, "temp"))
        expected_joint_critic_path = kwargs.pop(
            "expected_joint_critic_path", None)
        content = joint_formatter._make_prompt_parts(*args, **kwargs)
        if expected_joint_critic_path is None:
            return content
        joint_critic_response = load_json(expected_joint_critic_path)
        content.append(
            f"The correct response is:\n```json\n{json.dumps(joint_critic_response, indent=2)}\n```"
        )

        return content

    def _extract_example_kwargs(self, example_path):
        joint_id = os.path.basename(example_path)
        obj_id = os.path.basename(os.path.dirname(example_path))

        semantic_joint_id = get_semantic_joint_id(
            obj_id, joint_id,
            # input_dir=os.path.dirname(self.cfg.dataset_dir),
        )

        gt_video_name = (
            f"{'aug_' if self.cfg.joint_critic.use_cotracker else ''}video_{joint_id}_{self.cfg.cam_view}.mp4"
        )
        pred_video_name = f"{'aug_' if self.cfg.joint_critic.use_cotracker else ''}video_{semantic_joint_id}_{self.cfg.cam_view}.mp4"

        candidate_function_path = join_path(example_path, "joint_pred.py")
        expected_joint_critic_path = join_path(
            example_path, "joint_critic.json")

        gt_video_path = join_path(example_path, gt_video_name)
        pred_video_path = join_path(example_path, pred_video_name)

        return {
            "candidate_function_path": candidate_function_path,
            "gt_video_path": gt_video_path,
            "pred_video_path": pred_video_path,
            "expected_joint_critic_path": expected_joint_critic_path,
        }


def make_joint_critic(cfg):
    JointCriticCls = {
        "basic": JointCritic,
        "incontext": JointCriticMultiModalExamples,
    }
    return JointCriticCls[cfg.joint_critic.type]
