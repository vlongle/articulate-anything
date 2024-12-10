from articulate_anything.agent.agent import (
    Agent,
)
import json
from articulate_anything.utils.utils import (
    save_json,
    join_path,
)

TASK_SPECIFIER_SYSTEM_INSTRUCTION = """
You are a expert designer whose role is to take a user's specified object and imagine a detailed description of said object's spatial layouts. 

## Example:

user_description: "a water bottle"
output: "a water bottle with a cap on the top and a cylindrical body on the bottom."

user_description: "a nightstand"
output: "a nightstand with a single drawer."

user_description: "a large China cabinet"
output: "a large China cabinet with two doors on the top and three drawers on the bottom."

user_description: "office chair with wheels"
output: "an office chair with a seat, a leg, and four wheels at the base."

user_description: "a stovetop"
output: "a stovetop with a oven door and a storage drawer on the bottom."

user_description: "a soap dispenser"
output: "a soap dispenser with a pump on the top and a cylindrical body on the bottom."

## Tips:

- Be creative but stay within the realms of reality.
- Only consider the basic physical functions and relative positioning of the object. Do not worry about the object's apperance (color, materials, etc.)
- Specify the number of components for each category. Stay within reasonable numbers - That means NO MORE THAN 5 parts in TOTAL.
- Only consider the OUTSIDE functionalities of the object, DO NOT worry about internal features of the object.
- Do not worry about features that are too small or features that cannot be modeled by articulations
- If two parts are connected by a fixed joint, consider fusing them into one part.
- Output your response in a JSON block. Format:
```json
{
"reasoning": "I should create <total number of parts> parts, including: <what you will include>",
"output": "your output here"
}
```
- Only specify constituent parts for the target object. E.g., do not make background parts like walls, floors, etc.
- Use common sense to only specify the necessary parts. In PartNet-mobility, `body` usually represents the main body of the object. So you shouldn't create `suitcase_body` and `suitcase_base` as separate parts for a suitcase.
"""


class TextTaskSpecifier(Agent):
    OUT_RESULT_PATH = "task_specifier.json"

    def _make_system_instruction(self):
        return TASK_SPECIFIER_SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, text_prompt):
        prompt_parts = ["Now, the user needs you to describe: " + text_prompt]
        return prompt_parts

    def parse_response(self, response):
        json_str = response.text.strip().strip("```json").strip()

        # Parse the JSON string into a dictionary
        parsed_response = json.loads(json_str, strict=False)

        # Save the parsed response to a JSON file
        save_json(parsed_response, join_path(
            self.cfg.out_dir, self.OUT_RESULT_PATH))
