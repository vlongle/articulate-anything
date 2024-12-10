import hydra
from omegaconf import OmegaConf, DictConfig
import os
from articulate_anything.utils.parallel_utils import process_tasks
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_annotator import PartNetMeshAnnotator


def annotate_partnet_part(obj_id, gpu_id, cfg):
    partnet_dir = join_path(cfg.dataset_dir, obj_id)
    cfg = OmegaConf.create(cfg)  # copy the configuration
    cfg.out_dir = partnet_dir
    partnet_annotator = PartNetMeshAnnotator(cfg)
    partnet_annotator.generate_prediction(partnet_dir=partnet_dir)


@ hydra.main(version_base=None, config_path="../../conf", config_name="config")
def preprocess_partnet(cfg: DictConfig):
    obj_ids = os.listdir(cfg.dataset_dir)
    process_tasks(obj_ids, annotate_partnet_part, num_workers=cfg.parallel,
                  max_load_per_gpu=cfg.max_load_per_gpu,
                  cfg=cfg)


if __name__ == "__main__":
    preprocess_partnet()
