import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from articulate_anything.utils.utils import load_json, save_json, seed_everything
from articulate_anything.baselines.real2code.dataset import make_real2code_dataset
from articulate_anything.baselines.real2code.articulate_text_dataset import make_articulate_anything_dataset
from articulate_anything.baselines.real2code.utils import load_model_and_tokenizer
import os
from dotenv import load_dotenv
import argparse
from glob import glob
import json

def generate_response(model, tokenizer, prompt, temperature=0.001):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=temperature,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()
    



def evaluate(model, tokenizer, sample, temperature=0.001):
    prompt = sample['prompt']
    ground_truth = sample['completion']
    generated_response = generate_response(model, tokenizer, prompt, temperature=temperature)
    result = {
        'prompt': prompt,
        'ground_truth': ground_truth,
        'generated_response': generated_response
    }
    return result

def main(model_size, checkpoint_type, dataset_type):
    output_dir = f"finetune_{dataset_type}_results_{model_size}"
    model_base = f"meta-llama/CodeLlama-{model_size}-Instruct-hf"
    
    if checkpoint_type == "final":
        checkpoint_path = f"{output_dir}/final_model"
    elif checkpoint_type == "best":
        checkpoint_path = get_best_checkpoint(output_dir)
    else:
        checkpoint_path = None
    
    result_dir = "/mnt/kostas-graid/datasets/vlongle/articulate-anything/results" 
    seed_everything(0)
    
    # Choose dataset based on dataset_type
    if dataset_type == "real2code":
        train_dataset, val_dataset = make_real2code_dataset()
    elif dataset_type == "articulate":
        train_dataset, val_dataset = make_articulate_anything_dataset(result_dir)
    else:
        raise ValueError("Invalid dataset_type. Choose 'real2code' or 'articulate'.")
    
    model, tokenizer = load_model_and_tokenizer(model_base, checkpoint_path)
    model.eval()
    
    val_sample = val_dataset[0]
    val_result = evaluate(model, tokenizer, val_sample)
    save_json(val_result, f"{dataset_type}_val_sample_result_{model_size}_{checkpoint_type}.json")
    
    train_sample = train_dataset[0]
    train_result = evaluate(model, tokenizer, train_sample)
    save_json(train_result, f"{dataset_type}_train_sample_result_{model_size}_{checkpoint_type}.json")

def get_best_checkpoint(output_dir):
    checkpoints = glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    best_checkpoint = None
    best_loss = float('inf')
    
    for checkpoint in checkpoints:
        trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            
            # Get the last evaluation loss
            if 'log_history' in trainer_state:
                eval_losses = [log['eval_loss'] for log in trainer_state['log_history'] if 'eval_loss' in log]
                if eval_losses:
                    last_eval_loss = eval_losses[-1]
                    if last_eval_loss < best_loss:
                        best_loss = last_eval_loss
                        best_checkpoint = checkpoint
    
    if best_checkpoint:
        print(f"Best checkpoint found: {best_checkpoint} with validation loss: {best_loss}")
    else:
        print("No valid checkpoint found with evaluation loss information.")
    
    return best_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CodeLlama model")
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b", "34b", "70b"], 
                        help="Size of the CodeLlama model to use (default: 7b)")
    parser.add_argument("--checkpoint_type", type=str, default="none", choices=["final", "best", "none"],
                        help="Type of checkpoint to use: 'final' for final_model, 'best' for best checkpoint, 'none' for base model (default: none)")
    parser.add_argument("--dataset_type", type=str, default="articulate", choices=["real2code", "articulate"],
                        help="Type of dataset to use for evaluation (default: articulate)")
    args = parser.parse_args()
    
    main(args.model_size, args.checkpoint_type, args.dataset_type)