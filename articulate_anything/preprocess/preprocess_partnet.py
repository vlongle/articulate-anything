import shutil
import hydra
from omegaconf import OmegaConf, DictConfig
import os
from articulate_anything.utils.parallel_utils import process_tasks
from articulate_anything.utils.partnet_utils import get_obj_type
from articulate_anything.utils.utils import (
    run_subprocess,
    config_to_command,
    make_cmd,
    join_path,
    string_to_file,
)
from articulate_anything.preprocess.preprocess_utils import (
    mask_urdf,
)
from typing import Optional
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_annotator import combine_meshes
from articulate_anything.api.odio_urdf import process_urdf
import logging


def get_urdf_file(directory: str, filename: str = "mobility.urdf") -> str:
    return join_path(directory, filename)


def render_object(urdf_file: str, gpu_id: str, simulator_cfg: DictConfig, render_mode: str = "move"):
    """
    Render an object based on its URDF file.

    :param urdf_file: Path to the URDF file
    :param gpu_id: GPU ID to use for rendering
    :param cfg: Configuration object
    :param render_mode: 'stationary' for photo rendering, 'move' for joint movement rendering
    """
    simulator_cfg = OmegaConf.create(simulator_cfg)
    simulator_cfg.urdf.file = urdf_file
    simulator_cfg.simulation_params.stationary_or_move = render_mode

    command = config_to_command(
        simulator_cfg,
        "articulate_anything/physics/sapien_simulate.py",
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_subprocess(command, env=env)


def render_partnet_obj(obj_id: str, gpu_id: str, cfg: DictConfig, render_mode: str = "move", urdf_file: Optional[str] = None):
    """
    Render a PartNet object, handling PartNet-specific logic.

    :param obj_id: PartNet object ID
    :param gpu_id: GPU ID to use for rendering
    :param cfg: Configuration object
    :param render_mode: 'stationary' for photo rendering, 'move' for joint movement rendering
    :param urdf_file: Optional path to URDF file. If not provided, it will be derived from obj_id
    """
    obj_dir = join_path(cfg.dataset_dir, obj_id)
    if urdf_file is None:
        urdf_file = get_urdf_file(obj_dir)

    simulator_cfg = OmegaConf.create(cfg.simulator)
    if get_obj_type(obj_id) == "Chair":
        simulator_cfg.urdf.raise_distance_offset = 0.15

    rotate_urdf(urdf_file)
    render_object(urdf_file, gpu_id, simulator_cfg, render_mode)
    combine_meshes(obj_dir)


def rotate_urdf(urdf_file: str):
    # first, make copy of the urdf file name urdf_file + ".backup"
    backup_file = urdf_file + ".legacy"
    if not os.path.exists(backup_file):
        logging.info(
            f"Rotating URDF file {urdf_file} by 270 degrees. Backup file: {backup_file}")
        shutil.copy2(urdf_file, backup_file)
        robot = process_urdf(urdf_file)
        joint = robot.get_joint_for_parent("base")
        angle_degree = 180
        robot.rotate_joint(joint.name, axis="y", angle_degree=angle_degree)
        string_to_file(str(robot), urdf_file)


def render_parts(obj_id: str, gpu_id: str, cfg: DictConfig):
    """
    Create a robot_{cfg.simulator.camera_params.views.cam_view}.png per part meshes.
    Preprocessing step for visual inputs
    """
    obj_dir = join_path(cfg.dataset_dir, obj_id)
    cmd_args = [obj_dir]
    if cfg.render_part_views:
        cmd_args.append("--render_part_views")

    command = make_cmd(
        script_path="articulate_anything/agent/actor/mesh_retrieval/partnet_mesh_annotator.py",
        cmd_args=[obj_dir]
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_subprocess(command, env=env)


def preprocess_partnet_object(obj_id: str, gpu_id: str, cfg: DictConfig,):
    """
    Get the link summary for the partnet object
    """
    obj_dir = join_path(cfg.dataset_dir, obj_id)
    urdf_file = get_urdf_file(obj_dir)
    robot = mask_urdf(urdf_file)
    link_summary = robot.summarize()
    link_summary = f"object_id: {obj_id}\n\n{link_summary}"
    string_to_file(link_summary, join_path(obj_dir, "link_summary.txt"))


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def preprocess_objects(cfg: DictConfig):
    if isinstance(cfg.obj_ids, int):
        cfg.obj_ids = [cfg.obj_ids]
    elif isinstance(cfg.obj_ids, str):
        cfg.obj_ids = [obj_id.strip() for obj_id in cfg.obj_ids.split(',')]
    if cfg.obj_ids is None:
        cfg.obj_ids = os.listdir(cfg.dataset_dir)
    cfg.obj_ids = list(map(str, cfg.obj_ids))

 
    process_function = {
        "text": render_parts,
        "image": lambda obj_id, gpu_id, cfg: render_partnet_obj(obj_id, gpu_id, cfg, render_mode="stationary"),
        "video": lambda obj_id, gpu_id, cfg: render_partnet_obj(obj_id, gpu_id, cfg, render_mode="stationary"),
        # "video": lambda obj_id, gpu_id, cfg: render_partnet_obj(obj_id, gpu_id, cfg, render_mode="move"),
        "partnet": preprocess_partnet_object
    }.get(cfg.modality)

    if process_function:
        process_tasks(cfg.obj_ids, process_function, num_workers=cfg.parallel,
                      max_load_per_gpu=cfg.max_load_per_gpu,
                      cfg=cfg)
    else:
        raise ValueError(
            f"Invalid modality {cfg.modality}. Supported modalities are 'text', 'image', 'video', and 'partnet'.")


if __name__ == "__main__":
    preprocess_objects()
