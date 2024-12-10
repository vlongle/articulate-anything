import json
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict, Any
from actor_critic import (
    actor_critic_loop,
    error_handler,
)
from articulate_anything.utils.utils import (
    create_task_config,
    join_path,
    Steps,
)
from articulate_anything.agent.actor.joint_prediction.joint_actor import make_joint_actor
from articulate_anything.agent.critic.joint_prediction.joint_critic import make_joint_critic
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_retrieval import (
    add_mesh_to_out_folder,
)
from articulate_anything.utils.cotracker_utils import make_cotracker
import os
from articulate_anything.preprocess.preprocess_partnet import (
    get_urdf_file,
    render_object,
)
from articulate_anything.api.odio_urdf import (
    get_semantic_joint_id,
    get_semantic_joint_from_child_name,
    get_joint_id,
)


def preprocess(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> Dict[str, Any]:
    modality_processors = {
        "video": process_visual,
        "image": process_image,
        "partnet": process_partnet,
        "text": process_text
    }
    processor = modality_processors.get(cfg.modality)
    if not processor:
        raise ValueError(f"Preprocess failed. Unsupported modality: {cfg.modality}")

    link_actor = steps["Link Articulation"]["Link actor"][-1]
    print("No. of link steps:", len(steps["Link Articulation"]["Link actor"]))
    print("Picking the last link step")
    cfg.joint_actor.link_placement_path = join_path(
        link_actor.cfg.out_dir, link_actor.OUT_RESULT_PATH)

    print("Link placement path:", cfg.joint_actor.link_placement_path)

    cfg = processor(prompt, steps, gpu_id, cfg)

    steps.add_step("Joint actor", [])  # list, one per iteration
    steps.add_step("Joint critic", [])  # list, one per iteration
    return {"cfg": cfg, "prompt": prompt}


def process_image(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """
    Process the image prompt. No need to preprocess the object.
    """
    return cfg

def process_partnet(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """
    Set the targetted joint for the joint actor. Must be provided for PartNet.
    Then, process the gt video.
    """
    cfg.prompt = str(cfg.prompt)
    cfg.out_dir = join_path(cfg.out_dir, cfg.prompt)

    if cfg.joint_actor.targetted_affordance and cfg.additional_prompt:
        cfg.joint_actor.targetted_joint = cfg.additional_prompt

    assert cfg.joint_actor.targetted_joint or cfg.additional_prompt, "PartNet: Please provide a targetted joint or additional prompt"

    cfg.joint_actor.targetted_semantic_joint = get_semantic_joint_id(
        obj_id=cfg.prompt, joint_id=cfg.joint_actor.targetted_joint)

    cfg.joint_actor.targetted_semantic_part = cfg.joint_actor.targetted_semantic_joint.split(
        "to_")[-1]

    cfg.simulator.simulation_params.joint_name = cfg.joint_actor.targetted_joint
    cfg.prompt = join_path(cfg.dataset_dir,
                           f"video_{cfg.joint_actor.targetted_joint}_{cfg.cam_view}.mp4")  # gt_video

    render_object(get_urdf_file(cfg.dataset_dir), gpu_id, cfg.simulator,
                  "move")

    if cfg.joint_actor.use_cotracker:
        # render the partnet obj segmentation mask for more accurate cotracker annotation result
        temp_cfg = OmegaConf.create(cfg)
        temp_cfg.simulator.use_segmentation = True
        temp_cfg.simulator.ray_tracing = False
        render_object(get_urdf_file(temp_cfg.dataset_dir), gpu_id, temp_cfg.simulator,
                      "stationary")

        cotracker_model = make_cotracker(cfg)
        seg_mask_path = join_path(
            cfg.dataset_dir, f"robot_{cfg.cam_view}_seg.png")

        cotracker_model.forward(
            cfg.prompt, seg_mask_path=seg_mask_path,
            **cfg.cotracker)

        cfg.prompt = join_path(
            os.path.dirname(cfg.prompt), f"aug_{os.path.basename(cfg.prompt)}")



    cfg.simulator.simulation_params.joint_name = cfg.joint_actor.targetted_semantic_joint
    return cfg


def process_visual(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """
    Apply cotracker to gt_video if enabled
    """

    if cfg.modality == "video" and cfg.joint_actor.targetted_affordance:
        extractor = steps["Affordance Extraction"]
        cfg.joint_actor.targetted_semantic_part = extractor.load_prediction()[
            "part_name"]
        obj_selector = steps["Mesh Retrieval"]["Object Selection"]
        # we will articulate the object that was selected by template match
        selected_obj_id = obj_selector.load_prediction()["obj_id"]

        cfg.joint_actor.targetted_semantic_joint = get_semantic_joint_from_child_name(
            selected_obj_id, child_part_name=cfg.joint_actor.targetted_semantic_part)

        cfg.joint_actor.targetted_joint = get_joint_id(
            selected_obj_id, cfg.joint_actor.targetted_semantic_joint)

    if cfg.joint_actor.use_cotracker:
        # render the partnet obj segmentation mask for more accurate cotracker annotation result
        temp_cfg = OmegaConf.create(cfg)
        temp_cfg.simulator.use_segmentation = True
        temp_cfg.simulator.ray_tracing = False
        render_object(get_urdf_file(temp_cfg.dataset_dir), gpu_id, temp_cfg.simulator,
                      "stationary")

        video_file_name = os.path.basename(cfg.prompt)
        out_video_path = join_path(
            os.path.dirname(cfg.prompt), f"aug_{video_file_name}"
        )

        if not os.path.exists(out_video_path):
            cotracker_model = make_cotracker(cfg)
            cotracker_model.forward(prompt, **cfg.cotracker)
        cfg.prompt = join_path(
            os.path.dirname(cfg.prompt), f"aug_{os.path.basename(cfg.prompt)}")

    return cfg


def process_text(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    # IMPORTANT: move meshes to the joint_actor directory
    layout_planner = steps["Mesh Retrieval"]["Box Layout"]
    box_layout = layout_planner.load_prediction()

    mesh_searcher = steps["Mesh Retrieval"]["Mesh Retrieval"]
    mesh_info = mesh_searcher.load_prediction()
    meshes = {k: v["mesh_file"] for k, v in mesh_info.items()}

    out_joint_actor_dir = join_path("joint_actor", "iter_0", "seed_0")
    add_mesh_to_out_folder(out_joint_actor_dir, box_layout,
                           meshes,
                           cfg.out_dir)
    return cfg


def actor_function(iteration: int, seed: int, cfg: DictConfig,
                   prompt: str, gpu_id: str, retry_kwargs: dict) -> Dict[str, Any]:
    joint_actor = make_joint_actor(cfg)(create_task_config(cfg, join_path(
        "joint_actor", f"iter_{iteration}", f"seed_{seed}"))
    )
    joint_actor.generate_prediction(gt_input=cfg.prompt,
                                    **retry_kwargs, **cfg.gen_config)
    joint_actor.render_prediction(gpu_id)
    video = joint_actor.load_predicted_rendering()

    if cfg.modality != "text" and cfg.joint_actor.targetted_affordance:
        gt_joint_diff = joint_actor.compute_gt_diff()
        logging.info(f"GT joint diff is {gt_joint_diff}")

    result = {
        "candidate_function_path": join_path(joint_actor.cfg.out_dir, joint_actor.OUT_RESULT_PATH),
        "pred_video_path": video,
    }
    return result


def is_actor_only(cfg):
    # cfg.actor_critic.actor_only is either a boolean or a string "auto"
    # if "auto" then we use critic only if cfg.modality == "video"
    return cfg.actor_critic.actor_only if isinstance(cfg.actor_critic.actor_only, bool) else cfg.modality != "video"

def critic_function(iteration: int, seed: int, cfg: DictConfig, prompt: str, actor_result: Dict[str, Any]) -> Dict[str, Any]:
    if is_actor_only(cfg):
        return {
            "feedback_score": 10
        }
    joint_critic = make_joint_critic(cfg)(create_task_config(cfg, join_path(
        "joint_critic", f"iter_{iteration}", f"seed_{seed}"))
    )
    joint_critic.generate_prediction(gt_video_path=cfg.prompt,
                                     **actor_result, **cfg.gen_config)
    feedback = joint_critic.load_prediction()
    if cfg.joint_actor.targetted_affordance:
        assert cfg.joint_actor.targetted_semantic_part is not None
    return {
        "feedback_score": int(feedback['realism_rating']),
        "feedback":  json.dumps(feedback, indent=4),
        "link_placement_path": cfg.joint_actor.link_placement_path,  # will be automatically
        # populated by the preprocess function
        "targetted_affordance": cfg.joint_actor.targetted_semantic_part,
    }


def post_process_iter(best_result, cfg, steps):
    iteration = best_result["iteration"]
    seed = best_result["seed"]
    joint_actor = make_joint_actor(cfg)(create_task_config(cfg, join_path(
        "joint_actor", f"iter_{iteration}", f"seed_{seed}")))

    joint_critic = make_joint_critic(cfg)(create_task_config(cfg, join_path(
        "joint_critic", f"iter_{iteration}", f"seed_{seed}")))

    steps["Joint actor"].append(joint_actor)
    steps["Joint critic"].append(joint_critic)

    return steps


def articulate_joint(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> Dict[str, Any]:
    preprocess_result = preprocess(prompt, steps, gpu_id, cfg)
    cfg = preprocess_result["cfg"]

    if cfg.joint_actor.targetted_affordance:
        assert cfg.joint_actor.targetted_semantic_part is not None, "Targetted semantic part is not provided but the `joint_actor.targetted_affordance` is True. \
            For text modality. Must set `joint_actor.targetted_affordance` to False."

    retry_kwargs = {
        "link_placement_path": cfg.joint_actor.link_placement_path,  # will be automatically
        # populated by the preprocess function
        "targetted_affordance": cfg.joint_actor.targetted_semantic_part,
    }
    if cfg.joint_actor.mode != "video":
        actor_result = actor_function(iteration=0, seed=0,
                                      cfg=cfg, prompt=prompt,
                                      gpu_id=gpu_id,
                                      retry_kwargs=retry_kwargs)
        best_result = {**actor_result, "iteration": 0,
                       "seed": 0, "feedback_score": 10}
        post_process_iter(best_result, cfg, steps)
    else:
        best_result = actor_critic_loop(
            cfg,
            lambda i, s, r: actor_function(i, s, cfg, prompt, gpu_id, r),
            lambda i, s, a: critic_function(i, s, cfg, prompt, a),
            steps=steps,
            error_handler=lambda e, i, s: error_handler(
                e, "joint_error", i, s, cfg),
            retry_kwargs=retry_kwargs,
            post_process_iter=post_process_iter,
        )

    return steps
