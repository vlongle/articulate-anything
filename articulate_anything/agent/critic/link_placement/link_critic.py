from articulate_anything.utils.utils import (
    save_json,
    file_to_string,
    join_path,
)
import json
import logging
from PIL import Image
from articulate_anything.agent.agent import Agent

CRITIC_INSTRUCTION = """
An affordance of an object is a property or a feature that allows for specific actions or interactions. For example, a drawer might be opened and close.
We'd like to build a simple model with joints and links to simulate the object and its affordance. We will provide you
```
object: {description}
affordance: {description}
```
The links represent parts of an object and are already defined using meshes. We'd like you to help us evaluate a simple model with links to represent the object.
"""

CRITIC_INSTRUCTION_CONT = """
A candidate function `partnet_{object_id}` will be provided to you. You are responsible for assessing the realism of this model compared to the groundtruth.
Pay attention to the relative positioning of diferent object parts (i.e. links).

We have run the candidate code, and render the results in PyBullet. We will give you two images:
the groundtruth and the prediction in this order. Please compare the two videos and provide feedback on the prediction.
Your response in this format:
```json
{
"realism_rating": {0-10},
"description": {justify your rating},
}
```
You must first visually compare the two images. Only when the two images are different, should you look at the `candidate_function` to debug the issues with the prediction.
Your response should be concise and to the point. When writing description and improvements, pay attention to the predicted image compared to the groundtruth
and the provided `candidate_function` to debug the issues with the candidate function (if any).
We will parse your response using `json.loads` so please strictly follow the format.
"""


CRITIC_INSTRUCTION_END = """
Some helpful tips:

- You must give a rating lower than 5 if there is some egregious visual error that can be detected from
the prediction e.g., parts floating in the air or not being attached tightly to each other. Predicted object height
is significantly higher than the groundtruth.
- You must give a rating lower than 5 if there is a major change needed for any part of the object i.e., if
you suggest any change to `place_relative_to` and placement.
- If there are parts that are visible detached from other parts, give it a low rating.

"""

CRITIC_EXAMPLES = """
```json
{
"realism_rating": "4",
"description": "in the groundtruth , the door is placed on the front. Your door is placed on the right.
The candidate function confirms this."
}
```
```json
{
"realism_rating": "10",
"description": "The prediction is visually identical to the groundtruth."
}
```
```json
{
"realism_rating": "1",
"description": "The groundtruth depicts a closed kitchen island with two doors. The prediction shows 4 drawers coming out.
Looking at the candidate function, the `drawer` links are placed `front` of the `furniture_body`, which is wrong.
Try placing them `inside` the `furniture_body`."
}
```
```json
{
"realism_rating": "7",
"description": "The prediction does not look exactly like the groundtruth but it is close enough. The candidate function places the door in `front` of the body,
which is reasonable"
}
```

```json
{
"realism_rating": "4",
"description": "The door in the prediction comes out way more than the groundtruth. Looking at the candidate function,
the `translation_door` is placed `front` of the `furniture_body`, which normally would be correct. But in this case, the visual prediction looks wrong. Try placing it `inside` the `furniture_body`."
}
```
```json
{
"realism_rating": "2",
"description": "In the groundtruth, every part is tightly attached to each other. The prediction shows the lid floating above the air.
Either try to place `inside` in the candidate function or adjust the `clearance` to fix it."
}
```
```json
{
"realism_rating": "4",
"description": "The groundtruth shows two parts being connected. The prediction shows one part to the right of the other part, and the candidate
function confirms this. We should try other `placement` such as `inside` to fix it"
}
```
```json
{
"realism_rating": "4",
"description": "The groundtruth shows a seat inside the leg but in your prediction, the seat is above the body. Try placing the body first
then placing the seat inside the body."
}
```
```json
{
"realism_rating": "2",
"description": "The relative positioning of different parts in the prediction seems correct. However, in the groundtruth, each part is tightly connected
to each other. In the prediction, we can see some visible gaps between the parts (e.g, part floating in the air). The `candidate_function` should try
to include some small `clearance` to fit them together."
}
```
```
```json
{
"realism_rating": "3",
"description": "The prediction shows the handle floating above the body while it is attached to the body in the groundtruth. The `candidate_function` places the
handle `above` the body, which is reasonable but the visual prediction is floating so we should try placing `inside` instead
or adjust the `clearance` to fix it."
}
```
```json
{
"realism_rating": "3",
"description": "The groundtruth object is much shorter than the prediction. The `candidate_function` places the `screen`
`above` the `body`, which is reasonable. But the visual prediction is too tall. Try placing the `screen` `inside` the `body`
or adjust the `clearance` to fix it."
}
```
"""


class LinkCritic(Agent):
    OUT_RESULT_PATH = "link_critic.json"

    def _make_system_instruction(self):
        system_instruction = CRITIC_INSTRUCTION
        if self.cfg.modality == "text":
            # Need to have finer placement because the mesh are all re-centered to the origin
            example = file_to_string(
                'articulate_anything/examples/link_placement_desc_examples.py'
            )
        else:
            # Use the off-centered meshes from PartNet-Mobility so the placement needs not be so precise
            example = file_to_string(
                'articulate_anything/examples/link_placement_examples.py')
        system_instruction += (
            "Here are some examples of creating various objects using our API\n"
            + "```\n"
            + example
            + "\n```\n"
        )
        system_instruction += CRITIC_INSTRUCTION_CONT
        system_instruction += (
            "Here are some examples of the evaluation of the realism of various objects\n"
            + CRITIC_EXAMPLES
            + "\n"
        )
        system_instruction += CRITIC_INSTRUCTION_END
        return system_instruction

    def _make_prompt_parts(self, link_pred_path, pred_image_path, gt_image_path,
                           **kwargs):

        candidate_function = file_to_string(link_pred_path)
        candidate_function_text = (
            "The candidate function is:\n"
            + "```python\n"
            + candidate_function
            + "\n```"
        )
        gt_img = Image.open(gt_image_path)
        pred_img = Image.open(pred_image_path)
        prompt_parts = [
            candidate_function_text,
            "The groundtruth image is\n",
            gt_img,
            "The prediction image is\n",
            pred_img,
        ]
        return prompt_parts

    def parse_response(self, response, **kwargs):
        # Extract the JSON string from the response text
        json_str = response.text.strip().strip("```json").strip()

        # Parse the JSON string into a dictionary
        parsed_response = json.loads(json_str, strict=False)
        logging.info(f">> LINK CRITIC RESPONSE: {parsed_response}")

        # Save the parsed response to a JSON file
        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))
