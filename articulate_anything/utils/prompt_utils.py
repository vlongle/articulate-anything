import logging
import random
import astor
import ast
import re
import os
import google.generativeai as genai
from articulate_anything.utils.utils import string_to_file
import textwrap
import markdown2
from io import BytesIO
import base64
from PIL import Image
import anthropic
from openai import OpenAI
 


def setup_gemini(model_name, system_instruction=None, api_key=None):
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)

    # genai.list_models()
    return genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction,
    )




def setup_vlm_model(model_name, system_instruction=None, api_key=None):
    if "gpt" in model_name:
        return setup_gpt(model_name, system_instruction, api_key)
    elif "gemini" in model_name:
        return setup_gemini(model_name, system_instruction, api_key)
    elif "claude" in model_name:
        return setup_claude(model_name, system_instruction, api_key)
    else:
        raise ValueError("Model name must contain 'gpt' or 'gemini'. Got: {}".format(model_name))

def setup_gemini(model_name, system_instruction=None, api_key=None):
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)

    # genai.list_models()
    return genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction,
    )




class ClaudeWrapper:
    def __init__(self, model_name, system_instruction, api_key):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _encode_image_to_base64(self, pil_image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    def _format_content(self, prompt_parts):
        """Format content for Claude API, handling both text and images"""
        if not isinstance(prompt_parts, list):
            return [{"type": "text", "text": prompt_parts}]
        
        formatted_content = []
        
        for part in prompt_parts:
            if isinstance(part, str):
                formatted_content.append({
                    "type": "text",
                    "text": part
                })
            elif isinstance(part, Image.Image):
                base64_image = self._encode_image_to_base64(part)
                formatted_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            elif isinstance(part, dict):
                formatted_content.append(part)
        
        return formatted_content
    
    def generate_content(self, prompt_parts, generation_config={}):
        temperature = generation_config.get("temperature", 0)
        max_tokens = generation_config.get("max_tokens", 1024)
        
        formatted_content = self._format_content(prompt_parts)
        
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.system_instruction,
            messages=[
                {
                    "role": "user",
                    "content": formatted_content
                }
            ]
        )
        print(">>> USAGE", message.usage)
        
        class MockResponse:
            def __init__(self, text):
                self.text = text
        
        return MockResponse(message.content[0].text)
    

def setup_claude(model_name, system_instruction=None, api_key=None):
    """
    Setup function for Claude models.
    
    Args:
        model_name (str): Name of the Claude model (e.g., "claude-3-opus-20240229")
        system_instruction (str, optional): System instruction for the model
        api_key (str, optional): Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY in environment
        
    Returns:
        ClaudeWrapper or None: Initialized wrapper if successful, None if no API key available
    """
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None
    
    return ClaudeWrapper(
        model_name=model_name,
        system_instruction=system_instruction,
        api_key=api_key
    )




class GPTWrapper:
    def __init__(self, model_name, system_instruction, api_key):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image_to_base64(self, pil_image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    def _format_content(self, prompt_parts):
        """Format content for OpenAI API, handling both text and images"""
        if not isinstance(prompt_parts, list):
            return prompt_parts  # Return as is if it's just a string

        formatted_content = []
        
        for part in prompt_parts:
            if isinstance(part, str):
                formatted_content.append(part)
            elif isinstance(part, Image.Image):
                # Convert PIL Image to base64 and format for OpenAI
                base64_image = self._encode_image_to_base64(part)
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }
                })
            elif isinstance(part, dict) and part.get("type") == "image_url":
                # If it's already in the correct format, pass it through
                formatted_content.append(part)
        
        return formatted_content

    def generate_content(self, prompt_parts, generation_config={}):
        temperature = generation_config.get("temperature", 0.5)
        
        # Format the content for OpenAI
        formatted_content = self._format_content(prompt_parts)
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature)
            
        # Format response to match Gemini's format
        response_text = completion.choices[0].message.content
        
        # Create a MockResponse class to mimic the structure expected by parse_response
        class MockResponse:
            def __init__(self, text):
                self.text = text
        return MockResponse(response_text)

def setup_gpt(model_name, system_instruction=None, api_key=None):
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        return None
    
    return GPTWrapper(model_name=model_name, 
                      system_instruction=system_instruction, 
                      api_key=api_key)
    





def save_prompt_parts_as_html(prompt_parts, html_file_path):
    html_content = prompt_parts_to_html(prompt_parts)
    string_to_file(html_content, html_file_path)


def prompt_parts_to_html(prompt_parts, max_image_width=300, max_image_height=300):
    if isinstance(prompt_parts, (str, Image.Image)):
        # If a single string or image is passed, convert to a list
        prompt_parts = [prompt_parts]

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.3.1/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        <style>
            .image-row {{
                display: flex;
                flex-wrap: nowrap;
                margin-bottom: 20px;
            }}
            .image-row img {{
                margin-right: 1px;
                max-width: {max_width}px;
                max-height: {max_height}px;
                height: auto;
            }}
        </style>
    </head>
    <body>
    """.format(
        max_width=max_image_width, max_height=max_image_height
    )

    image_row_open = False

    for part in prompt_parts:
        if isinstance(part, str):
            if image_row_open:
                html_content += "</div>"
                image_row_open = False
            formatted_text = to_markdown(part)
            html_content += f"<p>{markdown2.markdown(formatted_text, extras=['fenced-code-blocks', 'code-friendly'])}</p>"
        elif isinstance(part, Image.Image):
            if not image_row_open:
                html_content += '<div class="image-row">'
                image_row_open = True
            buffered = BytesIO()
            part.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            html_content += f'<img src="data:image/png;base64,{img_str}">'

    if image_row_open:
        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    return html_content


def to_markdown(text):
    text = text.replace("•", "  *")
    return textwrap.indent(text, "> ", predicate=lambda _: True)


def extract_code_from_string(code_string):
    """Extracts the Python code found between triple backticks (```) in a string."""
    pattern = r"```python\n([\s\S]*?)```"
    matches = re.findall(pattern, code_string, re.DOTALL)

    if matches:
        # Return the first match of the Python code block, removing whitespace
        return matches[0].strip()
    else:
        return None  # Indicate no code found


def remove_lines_containing(content: str, keyword: str) -> str:
    """Removes lines containing a specific keyword from the content string.

    Args:
        content (str): The content string to process.
        keyword (str): The keyword to search for.

    Returns:
        str: The content string with lines containing the keyword removed.
    """
    lines = content.split("\n")
    filtered_lines = [line for line in lines if keyword not in line]
    return "\n".join(filtered_lines)


def categorize_nodes(python_code: str):
    tree = ast.parse(python_code)

    imports = []
    other_top_level = []
    functions = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        else:
            other_top_level.append(node)

    return imports, other_top_level, functions


def create_new_module(imports, other_top_level, selected_functions):
    module = ast.Module(body=imports + other_top_level +
                        selected_functions, type_ignores=[])
    return astor.to_source(module)


def select_random_functions(functions, num_examples):
    num_examples = min(num_examples, len(functions))
    logging.info(
        f"Selecting {num_examples} examples")
    return random.sample(functions, num_examples)


def get_n_examples_from_python_code(python_code, num_examples):
    if isinstance(num_examples, int) and num_examples >= 0:
        imports, other_top_level, functions = categorize_nodes(python_code)
        selected_functions = select_random_functions(functions, num_examples)
        python_code = create_new_module(
            imports, other_top_level, selected_functions)
    return python_code
