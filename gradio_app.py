import gradio as gr
import logging
from pathlib import Path
import json
from typing import Dict, Any
from dataclasses import dataclass
from omegaconf import OmegaConf

from articulate_anything.utils.utils import load_config
from articulate_anything.utils.cotracker_utils import make_cotracker
from articulate import ArticulationPipeline

logging.basicConfig(level=logging.INFO)

class ArticulationUI:
    def __init__(self):
        self.pipeline = None
        self.joint_data = None

    def setup_pipeline(self, prompt: str, api_key: str, model_name: str, use_cotracker: bool,
                       max_points: int, grid_size: int, topk: int, frame_index: int,
                       max_iter: int, ray_tracing: bool, floor_texture: str,
                       actor_type: str, critic_type: str, modality: str) -> ArticulationPipeline:
        cfg = load_config()
        cfg.api_key = api_key

        if modality == "visual":
            media_path = prompt
            task = Path(media_path).stem
            file_extension = Path(media_path).suffix.lower()
            if file_extension in ['.jpg', '.jpeg', '.png']:
                cfg.modality = 'image'
            elif file_extension in ['.mp4', '.avi', '.mov', '.mpeg']:
                cfg.modality = 'video'
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            cfg.prompt = media_path
            cfg.out_dir = f"results/{cfg.modality}/{task}"

            if cfg.modality == 'video' and use_cotracker:
                self._setup_cotracker(cfg, media_path, max_points, grid_size)
        else:
            # Text modality
            cfg.modality = 'text'
            cfg.prompt = prompt
            task = "text_task"
            cfg.out_dir = f"results/text/{task}"
            # For text modality, no CoTracker, no model parameters, no critic setting
            use_cotracker = False
            topk = 1
            frame_index = -1
            max_iter = 1
            critic_type = "basic"
            actor_type = "basic"  # Forced to basic

        cfg.model_name = model_name
        cfg.category_selector.topk = topk
        cfg.obj_selector.frame_index = frame_index
        cfg.actor_critic.max_iter = max_iter

        # Simulator settings
        cfg.simulator.ray_tracing = ray_tracing
        cfg.simulator.floor_texture = floor_texture
        cfg.simulator.flip_video = False

        self._setup_actor_critic_config(cfg, use_cotracker, actor_type, critic_type, cfg.modality)

        return ArticulationPipeline(cfg)

    def _setup_cotracker(self, cfg, media_path, max_points, grid_size):
        cfg.cotracker.max_moving_points = max_points
        cfg.cotracker.grid_size = grid_size
        cfg.cotracker.mode = "online"

    def _setup_actor_critic_config(self, cfg, use_cotracker, actor_type, critic_type, modality):
        # Actor settings
        cfg.joint_actor.mode = modality
        cfg.joint_actor.use_cotracker = use_cotracker if modality == 'video' else False
        cfg.joint_actor.type = actor_type
        cfg.joint_actor.targetted_affordance = modality == "video"
        cfg.joint_actor.examples_dir = "datasets/multi_modal_incontext_examples/joint_actor/in_context_actor_examples_datasets"

        # Critic settings
        cfg.joint_critic.mode = modality
        cfg.joint_critic.use_cotracker = use_cotracker if modality == 'video' else False
        cfg.joint_critic.type = critic_type
        cfg.joint_critic.examples_dir = "datasets/multi_modal_incontext_examples/joint_critic/in_context_examples_datasets"

        # Other settings
        cfg.actor_critic.actor_only = "auto"

    def build_interface(self) -> gr.Blocks:
        media_width = 800
        media_height = 400

        with gr.Blocks(title="Articulate Anything", theme=gr.themes.Soft()) as app:
            gr.Markdown("# Articulate Anything")
            gr.Markdown("Project website: [articulate-anything.github.io](https://articulate-anything.github.io)") 
            gr.Markdown("🚀**HOW TO USE:** Select the modality (visual or text). If visual, provide the target video or image AND your Google's Gemini API key.\
                        If text, provide a text prompt and your API key. Set hyper-parameters as needed (only for visual) and hit 'Start Processing' to begin the articulation process.")

            with gr.Row():
                # Left Column
                with gr.Column(scale=1):
                    modality_choice = gr.Radio(
                        choices=["visual", "text"],
                        value="visual",
                        label="Input Modality"
                    )

                    media_input = gr.File(
                        label="Input Media", file_types=['image', 'video'],
                        visible=True
                    )
                    text_input = gr.Textbox(label="Text Prompt", visible=False)

                    def toggle_inputs(modality):
                        # For text mode: hide media_input, show text_input
                        if modality == "visual":
                            return (
                                gr.update(visible=True),  # media_input
                                gr.update(visible=False)  # text_input
                            )
                        else:
                            return (
                                gr.update(visible=False),
                                gr.update(visible=True)
                            )

                    modality_choice.change(
                        fn=toggle_inputs,
                        inputs=[modality_choice],
                        outputs=[media_input, text_input]
                    )

                    api_key = gr.Textbox(label="VLM API Key", type="password",)
                    model_name = gr.Dropdown(
                        choices=["gemini-1.5-flash-latest", "claude-3-5-sonnet-20241022", "gpt-4o"],
                        value="gemini-1.5-flash-latest",
                        label="Select Model"
                    )

                    use_cotracker = gr.Checkbox(label="Use CoTracker", value=True, visible=True)
                    max_points = gr.Slider(minimum=5, maximum=50, value=20, label="Max Moving Points", visible=True)
                    grid_size = gr.Slider(minimum=10, maximum=50, value=30, label="Grid Size", visible=True)

                    topk = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                     label="Category Selector Top-K", visible=True)
                    frame_index = gr.Slider(minimum=-1, maximum=10, value=-1, step=1,
                                            label="Object Selector Frame Index", visible=True)
                    max_iter = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                         label="Actor-Critic Max Iterations", visible=True)

                    ray_tracing = gr.Checkbox(label="Enable Ray Tracing", value=False)
                    floor_texture = gr.Radio(choices=["plain", "checkerboard"],
                                             value="plain", label="Floor Texture")

                    # For visual: user can choose actor & critic type
                    actor_type = gr.Radio(choices=["basic", "incontext"],
                                          value="basic", label="Actor Prompting Type", visible=True)
                    critic_type = gr.Radio(choices=["basic", "incontext"],
                                           value="incontext", label="Critic Prompting Type", visible=True)

                    status = gr.Markdown("Status: Ready")
                    process_btn = gr.Button("Start Processing")

                # Right Column
                with gr.Column(scale=2):
                    gr.Markdown("### Media Preview")
                    media_preview_image = gr.Image(visible=False, width=media_width, height=media_height)
                    media_preview_video = gr.Video(visible=False, width=media_width, height=media_height)

                    gr.Markdown("### 1. Mesh Retrieval")
                    with gr.Row():
                        with gr.Column():
                            category_output = gr.TextArea(label="Category")
                            object_output = gr.TextArea(label="Selected Object")
                        mesh_render = gr.Image(label="Mesh Rendering")

                    gr.Markdown("### 2. Link Articulation")
                    link_gallery = gr.Gallery(label="Link Predictions")
                    with gr.Row():
                        link_code = gr.Code(
                            label="Link Placement Code",
                            language="python",
                            show_label=True,
                        )
                        link_feedback = gr.TextArea(label="Link Feedback")

                    gr.Markdown("### 3. Affordance Extraction")
                    with gr.Row():
                        affordance_output = gr.TextArea(label="Extracted Affordance")
                        affordance_render = gr.Image(label="Affordance Visualization")

                    gr.Markdown("### 4. Joint Articulation")
                    joint_dropdown = gr.Dropdown(label="Select Joint", choices=[], visible=False)
                    joint_gallery = gr.Gallery(label="Joint Predictions")
                    with gr.Row():
                        joint_code = gr.Code(
                            label="Joint Articulation Code",
                            language="python",
                            show_label=True,
                        )
                        joint_feedback = gr.TextArea(label="Joint Feedback")


            def hide_visual_features_on_text(modality):
                # If text mode: hide co-tracker and model parameters, critic_type and actor_type
                if modality == "text":
                    return (
                        gr.update(visible=False),  # use_cotracker
                        gr.update(visible=False),  # max_points
                        gr.update(visible=False),  # grid_size
                        gr.update(visible=False),  # topk
                        gr.update(visible=False),  # frame_index
                        gr.update(visible=False),  # max_iter
                        gr.update(visible=False),  # critic_type
                        gr.update(visible=False),  # actor_type
                    )
                else:
                    # visual mode: all visible
                    return (
                        gr.update(visible=True),  # use_cotracker
                        gr.update(visible=True),  # max_points
                        gr.update(visible=True),  # grid_size
                        gr.update(visible=True),  # topk
                        gr.update(visible=True),  # frame_index
                        gr.update(visible=True),  # max_iter
                        gr.update(visible=True),  # critic_type
                        gr.update(visible=True),  # actor_type
                    )

            modality_choice.change(
                fn=hide_visual_features_on_text,
                inputs=[modality_choice],
                outputs=[use_cotracker, max_points, grid_size, topk, frame_index, max_iter, critic_type, actor_type]
            )

            def show_cotracker_params(use_ct, mod):
                # If text mode or CoTracker not selected, hide co-tracker sliders
                if mod != "visual" or not use_ct:
                    return gr.update(visible=False), gr.update(visible=False)
                else:
                    return gr.update(visible=True), gr.update(visible=True)

            use_cotracker.change(
                fn=show_cotracker_params,
                inputs=[use_cotracker, modality_choice],
                outputs=[max_points, grid_size]
            )

            def display_media_preview(media_input):
                if media_input is None:
                    return gr.update(visible=False), gr.update(visible=False)
                media_path = media_input.name
                file_extension = Path(media_path).suffix.lower()
                if file_extension in ['.jpg', '.jpeg', '.png']:
                    return gr.update(visible=True, value=media_path), gr.update(visible=False)
                elif file_extension in ['.mp4', '.avi', '.mov', '.mpeg']:
                    return gr.update(visible=False), gr.update(visible=True, value=media_path)
                else:
                    return gr.update(visible=False), gr.update(visible=False)

            media_input.change(
                fn=display_media_preview,
                inputs=[media_input],
                outputs=[media_preview_image, media_preview_video]
            )

            def process_mesh_retrieval(modality, media_input, text_input_value, api_key, model_name, use_cotracker_val, max_points_val, grid_size_val,
                                       topk_val, frame_index_val, max_iter_val, ray_tracing_val, floor_texture_val,
                                       actor_type_val, critic_type_val):
                if modality == "visual":
                    if media_input is None:
                        return {status: "Status: No media input provided"}
                    prompt = media_input.name
                else:
                    if not text_input_value.strip():
                        return {status: "Status: No text prompt provided"}
                    prompt = text_input_value.strip()

                if not api_key:
                    return {status: "Status: API Key is required"}

                # For text modality, actor_type must be basic and no critic_type allowed, so we override here:
                if modality == "text":
                    actor_type_val = "basic"
                    critic_type_val = "basic"
                    use_cotracker_val = False
                    topk_val = 1
                    frame_index_val = -1
                    max_iter_val = 1

                self.pipeline = self.setup_pipeline(
                    prompt, api_key, model_name, use_cotracker_val, max_points_val, grid_size_val,
                    topk_val, frame_index_val, max_iter_val, ray_tracing_val, floor_texture_val,
                    actor_type_val, critic_type_val, modality
                )

                result = self.pipeline.process_mesh_retrieval()
                if result.success:
                    if modality == "text":
                        return {
                            status: "Status: Text processing completed (no mesh for text modality)",
                            category_output: "Not applicable for text",
                            object_output: "Not applicable for text",
                            mesh_render: None
                        }
                    else:
                        mesh_data = result.data
                        return {
                            status: "Status: Mesh retrieval completed",
                            category_output: mesh_data["Category Selection"].load_prediction(),
                            object_output: mesh_data["Object Selection"].load_prediction(),
                            mesh_render: mesh_data["Object Selection"].load_predicted_rendering()
                        }
                return {status: f"Status: Error in mesh retrieval - {result.error_msg}"}

            def process_link_articulation():
                if not self.pipeline:
                    return {status: "Status: Pipeline not initialized"}

                # if self.pipeline.cfg.modality == "text":
                #     return {
                #         status: "Status: Link articulation skipped for text input",
                #         link_gallery: None,
                #         link_code: None,
                #         link_feedback: "Not applicable for text modality"
                #     }

                result = self.pipeline.process_link_articulation()
                if result.success:
                    link_data = result.data
                    try:
                        link_feedback_value = json.dumps(link_data["Link critic"][-1].load_prediction(), indent=4)
                    except:
                        link_feedback_value = "Not applicable for this input"
                    return {
                        status: "Status: Link articulation completed",
                        link_gallery: [actor.load_predicted_rendering() for actor in link_data["Link actor"]],
                        link_code: link_data["Link actor"][-1].load_prediction() if link_data["Link actor"] else None,
                        link_feedback: link_feedback_value
                    }
                return {status: f"Status: Error in link articulation - {result.error_msg}"}

            def process_affordance():
                if not self.pipeline:
                    return {status: "Status: Pipeline not initialized"}

                if self.pipeline.cfg.modality == "text":
                    return {
                        status: "Status: Affordance extraction skipped for text input",
                        affordance_output: "Not applicable for text modality",
                        affordance_render: None
                    }

                result = self.pipeline.process_affordance_extraction()
                if result.success:
                    if result.data:
                        return {
                            status: "Status: Affordance extraction completed",
                            affordance_output: result.data.load_prediction(),
                            affordance_render: result.data.render_prediction()
                        }
                    return {
                        status: "Status: Affordance extraction skipped (not applicable)",
                        affordance_output: "Not applicable for this input",
                        affordance_render: None
                    }
                return {status: f"Status: Error in affordance extraction - {result.error_msg}"}

            def process_joint_articulation():
                if not self.pipeline:
                    return {status: "Status: Pipeline not initialized"}

                # if self.pipeline.cfg.modality == "text":
                #     return {
                #         status: "Status: Joint articulation skipped for text input",
                #         joint_gallery: None,
                #         joint_code: None,
                #         joint_feedback: "Not applicable for text modality",
                #         joint_dropdown: gr.update(choices=[], visible=False)

                #     }

                result = self.pipeline.process_joint_articulation()
                if result.success:
                    self.joint_data = result.data
                    try:
                        joint_feedback_value = json.dumps(self.joint_data["Joint critic"][-1].load_prediction(), indent=4)
                    except:
                        joint_feedback_value = "Not applicable for this input"

                    joint_names = self.joint_data["Joint actor"][-1].get_all_joint_names()
                    return {
                        status: "Status: Processing completed",
                        joint_gallery: [actor.load_predicted_rendering(use_gif=True) for actor in self.joint_data["Joint actor"]],
                        joint_code: self.joint_data["Joint actor"][-1].load_prediction() if self.joint_data["Joint actor"] else None,
                        joint_feedback: joint_feedback_value,
                        joint_dropdown: gr.Dropdown(choices=joint_names, visible=True, interactive=True)
                    }
                return {status: f"Status: Error in joint articulation - {result.error_msg}"}

            def load_joint_rendering(joint_name: str):
                if not self.pipeline or not self.joint_data:
                    return None
                try:
                    gallery = [actor.load_predicted_rendering(joint_name=joint_name, use_gif=True)
                               for actor in self.joint_data["Joint actor"]]
                    return gallery
                except Exception as e:
                    logging.error(f"Error loading renderings for joint {joint_name}: {e}")
                    return None

            mesh_result = process_btn.click(
                fn=process_mesh_retrieval,
                inputs=[
                    modality_choice, media_input, text_input, api_key, model_name, use_cotracker, max_points, grid_size,
                    topk, frame_index, max_iter, ray_tracing, floor_texture,
                    actor_type, critic_type
                ],
                outputs=[status, category_output, object_output, mesh_render]
            ).then(
                fn=process_link_articulation,
                inputs=[],
                outputs=[status, link_gallery, link_code, link_feedback]
            ).then(
                fn=process_affordance,
                inputs=[],
                outputs=[status, affordance_output, affordance_render]
            ).then(
                fn=process_joint_articulation,
                inputs=[],
                outputs=[status, joint_gallery, joint_code, joint_feedback, joint_dropdown]
            )

            joint_dropdown.change(
                fn=load_joint_rendering,
                inputs=[joint_dropdown],
                outputs=[joint_gallery],
                queue=True,
            )

        app.queue()
        return app

def create_app():
    ui = ArticulationUI()
    return ui.build_interface()

if __name__ == "__main__":
    app = create_app()
    app.launch(share=True, server_name="0.0.0.0")
