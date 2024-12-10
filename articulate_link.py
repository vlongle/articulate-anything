import os
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
from articulate_anything.agent.actor.link_placement.link_placement import LinkPlacementActor
from articulate_anything.agent.critic.link_placement.link_critic import LinkCritic
from articulate_anything.preprocess.preprocess_partnet import (
    render_partnet_obj,
    preprocess_partnet_object,
)
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_retrieval import (
    add_mesh_to_out_folder,
    write_box_layout_to_link_summary,
)
from articulate_anything.agent.actor.mesh_retrieval.obj_selector import (
    make_obj_selector,
)


def preprocess(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> Dict[str, Any]:
    """Prepare the environment and configuration based on the modality."""
    modality_processors = {
        "partnet": process_partnet,
        "image": process_visual,
        "video": process_visual,
        "text": process_text
    }
    processor = modality_processors.get(cfg.modality)
    if not processor:
        raise ValueError(f"Preprocess failed. Unsupported modality: {cfg.modality}")

    cfg = processor(prompt, steps, gpu_id, cfg)

    steps.add_step("Link actor", [])  # list, one per iteration
    steps.add_step("Link critic", [])  # list, one per iteration
    return {"cfg": cfg, "prompt": prompt}


def process_visual(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    obj_selector = steps["Mesh Retrieval"]["Object Selection"]
    # we will articulate the object that was selected by template match
    selected_obj_id = obj_selector.load_prediction()["obj_id"]
    render_partnet_obj(selected_obj_id, gpu_id, cfg, "stationary")
    cfg.dataset_dir = join_path(cfg.dataset_dir, selected_obj_id)
    cfg = OmegaConf.create(cfg)  # copy for link_placement
    cfg.prompt = selected_obj_id
    temp_cfg = OmegaConf.create(cfg)
    temp_cfg.dataset_dir = os.path.dirname(cfg.dataset_dir)  # because
    # preprocess_partnet_object will change link_cfg.dataset_dir
    preprocess_partnet_object(selected_obj_id, gpu_id, temp_cfg)
    return cfg


def process_partnet(obj_id: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """Process partnet modality."""
    render_partnet_obj(obj_id, gpu_id, cfg, "stationary")
    preprocess_partnet_object(obj_id, gpu_id, cfg)
    cfg.dataset_dir = join_path(cfg.dataset_dir, obj_id)
    cfg.out_dir = join_path(cfg.out_dir, obj_id)
    return cfg


def process_text(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """Process text modality."""
    cfg.link_actor.mode = "text"
    layout_planner = steps["Mesh Retrieval"]["Box Layout"]
    box_layout = layout_planner.load_prediction()

    mesh_searcher = steps["Mesh Retrieval"]["Mesh Retrieval"]
    mesh_info = mesh_searcher.load_prediction()
    meshes = {k: v["mesh_file"] for k, v in mesh_info.items()}

    out_link_actor_dir = join_path("link_placement", "iter_0", "seed_0")
    # IMPORTANT: move meshes to the link_placement directory
    new_box_layout = add_mesh_to_out_folder(out_link_actor_dir, box_layout,
                                            meshes,
                                            cfg.out_dir)
    link_summary_path = join_path(
        cfg.out_dir, out_link_actor_dir, "link_summary.txt")
    write_box_layout_to_link_summary(new_box_layout, link_summary_path)

    cfg.link_actor.link_summary_path = link_summary_path
    cfg.link_actor.new_box_layout = new_box_layout
    return cfg


def actor_function(iteration: int, seed: int, cfg: DictConfig, prompt: str, gpu_id: str, retry_kwargs: dict) -> Dict[str, Any]:
    """Execute the actor part of the pipeline."""
    link_placement_actor = LinkPlacementActor(
        create_task_config(cfg, join_path(
            "link_placement", f"iter_{iteration}", f"seed_{seed}"))
    )
    link_placement_actor.generate_prediction(
        **cfg.gen_config, **retry_kwargs)

    result = {
        "pred_image_path": link_placement_actor.load_predicted_rendering(),
        "link_pred_path": join_path(link_placement_actor.cfg.out_dir, link_placement_actor.OUT_RESULT_PATH),
    }

    link_placement_actor.render_prediction(gpu_id)
    if cfg.modality != "text":
        gt_link_diff = link_placement_actor.compute_gt_diff()
        logging.info(f"GT link diff is {gt_link_diff}")
        result["gt_link_diff"] = gt_link_diff

    return result




def is_actor_only(cfg):
    # cfg.actor_critic.actor_only is either a boolean or a string "auto"
    # if "auto" then we should critic AND actor only if cfg.modality == "image" or "video".
    # i.e., actor_only is when cfg.modality == "text"
    return cfg.actor_critic.actor_only if isinstance(cfg.actor_critic.actor_only, bool) else cfg.modality == "text"




def critic_function(iteration: int, seed: int, cfg: DictConfig, prompt: str, actor_result: Dict[str, Any]) -> Dict[str, Any]:
    if is_actor_only(cfg):
        return {
            "feedback_score": 10,
        }

    """Execute the critic part of the pipeline."""
    if cfg.modality == "text":
        return {"feedback_score": 10, "feedback_path": None}

    link_critic = LinkCritic(create_task_config(cfg, join_path(
        "link_critic", f"iter_{iteration}", f"seed_{seed}")))
    link_critic.generate_prediction(
        gt_image_path=join_path(
            cfg.dataset_dir, f"robot_{cfg.cam_view}.png"),
        **actor_result,
        **cfg.gen_config,
    )
    feedback = link_critic.load_prediction()

    return {
        "feedback_score": int(feedback['realism_rating']),
        "feedback_path": join_path(link_critic.cfg.out_dir, link_critic.OUT_RESULT_PATH),
        "link_summary_path": cfg.link_actor.link_summary_path,  # will be automatically
        # populated by the preprocess function
    }


def post_process_iter(best_result: Dict[str, Any], cfg: DictConfig, steps) -> Dict[str, Any]:
    iteration = best_result["iteration"]
    seed = best_result["seed"]
    link_critic = LinkCritic(create_task_config(cfg, join_path(
        "link_critic", f"iter_{iteration}", f"seed_{seed}")))

    link_actor = LinkPlacementActor(create_task_config(cfg, join_path(
        "link_placement", f"iter_{iteration}", f"seed_{seed}")))

    steps["Link critic"].append(link_critic)
    steps["Link actor"].append(link_actor)
    return steps


def articulate_link(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> Dict[str, Any]:
    """Main function to articulate links based on the given prompt and configuration."""
    # Pre-processing
    preprocess_result = preprocess(prompt, steps, gpu_id, cfg)
    cfg = preprocess_result["cfg"]

    # Actor-Critic loop or single actor run
    if cfg.modality == "text":
        retry_kwargs = {
            "link_summary_path": cfg.link_actor.link_summary_path,  # will be automatically
            # populated by the preprocess function
        }
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
                e, "link_error", i, s, cfg),
            post_process_iter=post_process_iter,
        )

    return steps
