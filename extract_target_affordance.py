from omegaconf import DictConfig, OmegaConf
from articulate_anything.agent.actor.joint_prediction.affordance_extractor import TargettedAffordanceExtractor
from articulate_anything.utils.utils import create_task_config, Steps, join_path
from articulate_anything.preprocess.preprocess_partnet import render_object, get_urdf_file
import os
from articulate_anything.utils.cotracker_utils import make_cotracker


def extract_affordance(cfg: DictConfig, steps: Steps, gpu_id: str):
    extractor = TargettedAffordanceExtractor(
        create_task_config(cfg, "affordance_extractor"))
    cfg = OmegaConf.create(cfg)  # copy for affordance_extractor
    cfg.simulator.use_segmentation = True
    cfg.simulator.ray_tracing = False
    cfg.simulator.object_white = False
    render_object(get_urdf_file(cfg.dataset_dir), gpu_id, cfg.simulator,
                  "stationary")

    video = cfg.prompt
    if cfg.joint_actor.use_cotracker:
        video_file_name = os.path.basename(cfg.prompt)
        video = join_path(
            os.path.dirname(cfg.prompt), f"aug_{video_file_name}"
        )
        if not os.path.exists(video):
            cotracker_model = make_cotracker(cfg)
            cotracker_model.forward(cfg.prompt, **cfg.cotracker)
    extractor.generate_prediction(video)

    return extractor
