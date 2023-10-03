# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from transformers import pipeline

import json


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf", metadata={"help": "the model name"})
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the SFT model name or path"},
    )

def load_model(model_name, model_name_or_path):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # loads model
    path = model_name_or_path
    if path is not None and path != "" and path != "None":
        # This is necesarry to fix the josn since for some reason, the DPO trainer does not include the base model.
        # This is a hack to fix that
        # If this crashes, ensure the model_name you choose matches the base model.
        if path[:3] == "dpo":
            json_path = os.path.join(path, "adapter_config.json")
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
            data["base_model_name_or_path"] = model_name
            with open(json_path, 'w') as json_file:
                json.dump(data, json_file)
        
        # load the model
        model = AutoPeftModelForCausalLM.from_pretrained(
            path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map='auto',
        )
        model.config.use_cache = True
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto', # automatically uses accelerate to put on multiple gpus
            trust_remote_code=True,
            use_auth_token=True,

        )
        model.config.use_cache = True
        model.eval()
    print(f"Testing {model_name}\nPretrained model: {model_name_or_path}\n\n")
    return model, tokenizer

def chat(question, model, tokenizer):
    toks = tokenizer(question, return_tensors="pt", padding=True).to("cuda")
    out_toks = model.generate(**toks, max_length=300, do_sample=True, repetition_penalty=1.1)
    out = tokenizer.batch_decode(out_toks, skip_special_tokens=True)[0]
    return out


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model, tokenizer = load_model(script_args.model_name, script_args.model_name_or_path)
    
#     question = """<s>[INST] <<SYS>>
# You are a helpful assistant. Always answer as helpfully as possible, while being safe.  Your answers should be detailed.
# <</SYS>>

# Define the steps of "turn left at traffic light with left-turn light": [/INST]
# """
    question = """<s>[INST] <<SYS>>
You are a helpful assistant. Always answer as helpfully as possible, while being safe.  Your answers should be detailed.
<</SYS>>

Define 3 steps of "turn left at traffic light with left-turn light": [/INST]
"""


    
    answer = chat(question, model, tokenizer)
    print("\n\n\n", answer)