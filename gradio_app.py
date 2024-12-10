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

    def setup_pipeline(self, media_path: str, api_key: str, use_cotracker: bool,
                      max_points: int, grid_size: int, topk: int, frame_index: int,
                      max_iter: int, ray_tracing: bool, floor_texture: str,
                      actor_type: str, critic_type: str) -> ArticulationPipeline:
        cfg = load_config()
        cfg.api_key = api_key
        task = Path(media_path).stem
        file_extension = Path(media_path).suffix.lower()
        if file_extension in ['.jpg', '.jpeg', '.png']:
            modality = 'image'
        elif file_extension in ['.mp4', '.avi', '.mov', '.mpeg']:
            modality = 'video'
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        cfg.modality = modality
        cfg.prompt = media_path
        cfg.out_dir = f"results/{modality}/{task}"

        # CoTracker setup
        if modality == 'video' and use_cotracker:
            self._setup_cotracker(cfg, media_path, max_points, grid_size)

        # Basic configuration
        cfg.model_name = "gemini-1.5-flash-latest"
        cfg.category_selector.topk = topk
        cfg.obj_selector.frame_index = frame_index
        cfg.actor_critic.max_iter = max_iter

        # Simulator settings
        cfg.simulator.ray_tracing = ray_tracing
        cfg.simulator.floor_texture = floor_texture
        cfg.simulator.flip_video = False

        # Actor/critic settings
        self._setup_actor_critic_config(cfg, use_cotracker, actor_type, critic_type, modality)

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
        with gr.Blocks(title="Articulate Anything") as app:
            gr.Markdown("# Articulate Anything")
            gr.Markdown("Please provide the target video or image AND your Google's Gemini API key.\
                        Set hyper-parameters as needed and hit 'Start Processing' to begin the articulation process.")

            # Create all required components
            with gr.Row():
                with gr.Column(scale=1):
                    media_input = gr.File(
                        label="Input Media", file_types=['image', 'video'],
                    )
                    api_key = gr.Textbox(label="Gemini API Key", type="password",)

                    # CoTracker settings
                    with gr.Group():
                        use_cotracker = gr.Checkbox(label="Use CoTracker", value=True)
                        with gr.Column(visible=True) as cotracker_params:
                            max_points = gr.Slider(minimum=5, maximum=50, value=20,
                                                label="Max Moving Points")
                            grid_size = gr.Slider(minimum=10, maximum=50, value=30,
                                                label="Grid Size")

                    # Additional parameters
                    with gr.Group():
                        gr.Markdown("### Model Parameters")
                        topk = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                       label="Category Selector Top-K")
                        frame_index = gr.Slider(minimum=-1, maximum=10, value=-1, step=1,
                                              label="Object Selector Frame Index")
                        max_iter = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                           label="Actor-Critic Max Iterations")

                        gr.Markdown("### Simulator Settings")
                        ray_tracing = gr.Checkbox(label="Enable Ray Tracing", value=False)
                        floor_texture = gr.Radio(choices=["plain", "checkerboard",],
                                               value="plain", label="Floor Texture")

                        gr.Markdown("### Prompting Settings")
                        actor_type = gr.Radio(choices=["basic", "incontext"],
                                            value="basic", label="Actor Prompting Type")
                        critic_type = gr.Radio(choices=["basic", "incontext"],
                                             value="incontext", label="Critic Prompting Type")

                    status = gr.Markdown("Status: Ready")
                    process_btn = gr.Button("Start Processing")

                with gr.Column(scale=2):
                    # Media Preview
                    gr.Markdown("### Media Preview")
                    with gr.Group():
                        media_preview_image = gr.Image(visible=False)
                        media_preview_video = gr.Video(visible=False)

                    # Mesh Retrieval Section
                    with gr.Group():
                        gr.Markdown("### 1. Mesh Retrieval")
                        with gr.Row():
                            with gr.Column():
                                category_output = gr.TextArea(label="Category")
                                object_output = gr.TextArea(label="Selected Object")
                            mesh_render = gr.Image(label="Mesh Rendering")

                    # Link Articulation Section
                    with gr.Group():
                        gr.Markdown("### 2. Link Articulation")
                        link_gallery = gr.Gallery(label="Link Predictions")
                        with gr.Row():
                            link_code = gr.Code(
                                label="Link Placement Code",
                                language="python",
                                show_label=True,
                            )
                            link_feedback = gr.TextArea(label="Link Feedback")

                    # Affordance Section
                    with gr.Group():
                        gr.Markdown("### 3. Affordance Extraction")
                        with gr.Row():
                            affordance_output = gr.TextArea(label="Extracted Affordance")
                            affordance_render = gr.Image(label="Affordance Visualization")

                    # Joint Articulation Section
                    with gr.Group():
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


            # Handle CoTracker parameter visibility
            use_cotracker.change(
                fn=lambda x: gr.Column.update(visible=x),
                inputs=[use_cotracker],
                outputs=[cotracker_params]
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

            def process_mesh_retrieval(media_input, api_key, use_cotracker, max_points, grid_size,
                                    topk, frame_index, max_iter, ray_tracing, floor_texture,
                                    actor_type, critic_type):
                if media_input is None:
                    return {status: "Status: No media input provided"}
                media_path = media_input.name
                if not api_key:
                    return {status: "Status: API Key is required"}
                self.pipeline = self.setup_pipeline(
                    media_path, api_key, use_cotracker, max_points, grid_size,
                    topk, frame_index, max_iter, ray_tracing, floor_texture,
                    actor_type, critic_type
                )
                result = self.pipeline.process_mesh_retrieval()
                if result.success:
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

                result = self.pipeline.process_affordance_extraction()
                if result.success:
                    if result.data:  # Check if affordance extraction was applicable
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
                    # Create gallery of renderings for the selected joint from all actors
                    gallery = [actor.load_predicted_rendering(joint_name=joint_name, use_gif=True) 
                            for actor in self.joint_data["Joint actor"]]
                    return gallery
                except Exception as e:
                    logging.error(f"Error loading renderings for joint {joint_name}: {e}")
                    return None

            # Chain the processing steps
            mesh_result = process_btn.click(
                fn=process_mesh_retrieval,
                inputs=[
                    media_input, api_key, use_cotracker, max_points, grid_size,
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


