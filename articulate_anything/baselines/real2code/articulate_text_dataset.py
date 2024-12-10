from torch.utils.data import Dataset as TorchDataset
from articulate_anything.utils.utils import load_config, join_path, file_to_string, load_json
from articulate_anything.agent.actor.joint_prediction.joint_actor import JointPredictionActor

class LLamaInstructArticulateAnythingDataset(TorchDataset):
    def __init__(self, tasks, result_dir="results"):
        self.tasks = tasks
        self.result_dir = result_dir
        
        # Load configuration and initialize JointPredictionActor
        cfg = load_config()
        cfg.joint_actor.mode = "text"
        cfg.joint_actor.type = "llama_finetune"
        cfg.joint_actor.targetted_affordance = True
        self.joint_actor = JointPredictionActor(cfg)
        self.system_instruction = self.joint_actor._make_system_instruction()
        
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        obj_id, joint_id, semantic_joint_id = task["obj_id"], task["joint_id"], task["semantic_joint_id"]
        
        link_result = task["link_diff"]
        link_result_dir = join_path(self.result_dir, obj_id, "link_placement", f"iter_{link_result['iter_num']}",
                  link_result["seed"])
        link_placement_path =join_path(link_result_dir, "link_placement.py")
        targetted_affordance = semantic_joint_id.split("to_")[-1]

        joint_result = task["joint_diff"]
        joint_result_dir = join_path(self.result_dir, obj_id, "joint_pred", 
                                        joint_id,
                                     f"iter_{joint_result['iter_num']}",
                                    joint_result["seed"])
        completion = file_to_string(join_path(joint_result_dir, "joint_pred.py"))
        # Generate the prompt using JointPredictionActor
        prompt_parts = self.joint_actor._make_prompt_parts(link_placement_path,
                            targetted_affordance=targetted_affordance)
        user_prompt = " ".join(prompt_parts)
        user_prompt += "\n Generate the joint prediction code:"
        
        prompt = f"[INST]\n<<SYS>>\n{self.system_instruction}\n<</SYS>>\n\n{user_prompt}[/INST]"

        
        return {
            "prompt": prompt,
            "completion": completion,
            "text": f"<s>{prompt}\n{completion}</s>"
        }


def make_articulate_anything_dataset(result_dir="results"):
    train_tasks = load_json("train_tasks.json")
    val_tasks = load_json("val_tasks.json")
    # train_tasks = load_json("train_all_tasks.json")
    # val_tasks = load_json("val_all_tasks.json")
    train_dataset = LLamaInstructArticulateAnythingDataset(train_tasks, 
                                                           result_dir=result_dir)
    val_dataset = LLamaInstructArticulateAnythingDataset(val_tasks, 
                                                         result_dir=result_dir)
    return train_dataset, val_dataset