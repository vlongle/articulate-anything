import datetime
import hydra
import logging
from typing import Dict, Optional, Callable, Any
import time
from omegaconf import DictConfig

from articulate_anything.utils.utils import seed_everything, Steps
from articulate_anything.mesh_retrieval import mesh_retrieval
from articulate_link import articulate_link
from articulate_joint import articulate_joint
from extract_target_affordance import extract_affordance
import traceback 

logging.basicConfig(level=logging.INFO)

class ProcessingResult:
    """Tracks execution results and timing"""
    def __init__(
        self, 
        success: bool, 
        data: Any, 
        time_taken: float, 
        error_msg: Optional[str] = None
    ):
        self.success = success
        self.data = data
        self.time_taken = time_taken
        self.error_msg = error_msg

    @classmethod
    def success(cls, data: Any, time_taken: float) -> 'ProcessingResult':
        return cls(success=True, data=data, time_taken=time_taken, error_msg=None)

    @classmethod
    def failure(cls, error: str, time_taken: float) -> 'ProcessingResult':
        return cls(success=False, data=None, time_taken=time_taken, error_msg=error)

class ArticulationPipeline:
    """Manages the articulation processing pipeline while maintaining Steps compatibility"""
    
    def __init__(self, cfg: DictConfig, progress_callback: Optional[Callable] = None):
        self.cfg = cfg
        self.steps = Steps()
        self.progress_callback = progress_callback
        seed_everything(0)

    def notify_progress(self, message: str) -> None:
        """Update progress if callback is provided"""
        if self.progress_callback:
            self.progress_callback(message)

    def execute_stage(self, name: str, func: Callable, *args, **kwargs) -> ProcessingResult:
        """Execute a processing stage with timing and error handling"""
        start_time = time.time()
        try:
            logging.info(f"Starting {name}")
            self.notify_progress(f"Processing {name}...")
            
            result = func(*args, **kwargs)
            time_taken = time.time() - start_time
            
            # Add to steps in the original format
            self.steps.add_step(name, result)
            
            return ProcessingResult.success(result, time_taken)
            
        except Exception as e:
            time_taken = time.time() - start_time
            error_msg = f"{name} failed: {traceback.format_exc()}"
            logging.error(error_msg)
            return ProcessingResult.failure(error_msg, time_taken)

    def process_mesh_retrieval(self) -> ProcessingResult:
        """Process mesh retrieval stage"""
        return self.execute_stage(
            "Mesh Retrieval",
            mesh_retrieval,
            self.cfg
        )

    def process_link_articulation(self) -> ProcessingResult:
        """Process link articulation stage"""
        def _process_link():
            original_out = self.cfg.out_dir
            
            # Handle partnet mode
            if self.cfg.modality == "partnet" and self.cfg.link_placement_path:
                self.cfg.out_dir = self.cfg.link_placement_path
            
            try:
                return articulate_link(
                    str(self.cfg.prompt),
                    self.steps,
                    str(self.cfg.gpu_id),
                    self.cfg
                )
            finally:
                self.cfg.out_dir = original_out
                
        return self.execute_stage(
            "Link Articulation",
            _process_link
        )

    def process_affordance_extraction(self) -> ProcessingResult:
        """Process affordance extraction stage"""
        def _should_extract_affordance():
            return (
                self.cfg.modality == "video" and 
                self.cfg.joint_actor.targetted_affordance
            )

        if not _should_extract_affordance():
            return ProcessingResult.success({}, 0.0)

        return self.execute_stage(
            "Affordance Extraction",
            extract_affordance,
            self.cfg,
            self.steps,
            str(self.cfg.gpu_id)
        )

    def process_joint_articulation(self) -> ProcessingResult:
        """Process joint articulation stage"""
        return self.execute_stage(
            "Joint Articulation",
            articulate_joint,
            str(self.cfg.prompt),
            self.steps,
            str(self.cfg.gpu_id),
            self.cfg
        )

    def run_pipeline(self) -> Steps:
        """Run the complete pipeline while maintaining Steps compatibility"""
        stages = [
            self.process_mesh_retrieval,
            self.process_link_articulation,
            self.process_affordance_extraction,
            self.process_joint_articulation
        ]
        
        for stage_func in stages:
            result = stage_func()
            if not result.success:
                break
                
        # Log summary
        self._log_processing_summary()
        
        # Return steps object for compatibility
        return self.steps

    def _log_processing_summary(self) -> None:
        """Log processing summary information"""
        logging.info(f"Pipeline execution completed")
        logging.info(f"Steps executed: {', '.join(self.steps.order)}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def articulate(cfg: DictConfig, progress_callback: Optional[Callable] = None) -> Steps:
    """
    Main articulation function that maintains backward compatibility
    Returns the Steps object containing all processing results
    """
    pipeline = ArticulationPipeline(cfg, progress_callback)
    return pipeline.run_pipeline()

if __name__ == "__main__":
    articulate()
