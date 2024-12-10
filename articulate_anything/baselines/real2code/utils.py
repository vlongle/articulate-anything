
import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import Dataset as HFDataset

def create_model_and_tokenizer(model_name="meta-llama/CodeLlama-7b-Instruct-hf"):
    load_dotenv() # load env variables from ".env" file that contains the HF_ACCESS_TOKEN
    token = os.getenv("HF_ACCESS_TOKEN")
    print("token:", token)
    from huggingface_hub import login
    login(token=token)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
    )
 
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=token,
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
 
    return model, tokenizer

def convert_to_hf_dataset(dataset):
    data = []
    num_samples = len(dataset)
    for idx in tqdm(range(num_samples), desc=f"Converting dataset to HF format"):
        item = dataset[idx]
        data.append(item)
    return HFDataset.from_list(data)

## set to eval mode
def load_model_and_tokenizer(model_base, checkpoint_path=None):
    model, tokenizer = create_model_and_tokenizer(model_base)
    
    if checkpoint_path:
        print("Loading checkpoint from:", checkpoint_path)
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        print("No checkpoint provided. Using the base model.")
    
    model.eval()
    return model, tokenizer