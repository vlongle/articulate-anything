import os
import json
from glob import glob
import torch
from torch.utils.data import Dataset
import random
from articulate_anything.utils.utils import (
    load_json,
    seed_everything,
)
import torch
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel
import datetime
import time
import argparse
from articulate_anything.baselines.real2code.dataset import make_real2code_dataset
from articulate_anything.baselines.real2code.articulate_text_dataset import make_articulate_anything_dataset
from articulate_anything.baselines.real2code.utils import create_model_and_tokenizer, convert_to_hf_dataset
from articulate_anything.utils.utils import join_path

def get_latest_checkpoint(output_dir):
    checkpoints = glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def main(model_size, dataset_type):
    start = time.time()
    output_dir = f"finetune_{dataset_type}_results_{model_size}"
    model_base = f"meta-llama/CodeLlama-{model_size}-Instruct-hf"
    num_train_epochs = 3
    save_freq = 100
    seed = 0
    cluster_prefix = "/mnt/kostas-graid/datasets/vlongle/articulate-anything/"
    seed_everything(0)
    
    # Choose dataset based on dataset_type
    if dataset_type == "real2code":
        train_dataset, val_dataset = make_real2code_dataset(join_path(cluster_prefix, "real2code_results/dataset"))
    elif dataset_type == "articulate":
        train_dataset, val_dataset = make_articulate_anything_dataset(join_path(cluster_prefix, "results"))
    else:
        raise ValueError("Invalid dataset_type. Choose 'real2code' or 'articulate'.")
    
    train_dataset = convert_to_hf_dataset(train_dataset)
    val_dataset = convert_to_hf_dataset(val_dataset)
    print("len(dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    
    # Set up the model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_base)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    latest_checkpoint = get_latest_checkpoint(output_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = PeftModel.from_pretrained(
            model,
            latest_checkpoint,
            is_trainable=True
        )
    else:
        print("Starting training from scratch")
    
    # Set up training configuration
    train_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=save_freq,
        eval_steps=save_freq,
        evaluation_strategy="steps",
        save_steps=save_freq,
        fp16=True,
        optim="adamw_torch",
        report_to="tensorboard",
        seed=seed,
    )
    
    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        args=train_config,
        tokenizer=tokenizer,
    )
    
    # Resume training
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    
    # Save the final model
    trainer.model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print("Training completed and model saved.")
    
    end = time.time()
    print(f"Experiment runs took {datetime.timedelta(seconds=end-start)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune CodeLlama model")
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b", "34b", "70b"], 
                        help="Size of the CodeLlama model to use (default: 7b)")
    parser.add_argument("--dataset_type", type=str, default="articulate", choices=["real2code", "articulate"],
                        help="Type of dataset to use for training (default: articulate)")
    args = parser.parse_args()
    
    main(args.model_size, args.dataset_type)