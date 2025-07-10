import pandas as pd
import json
import fire
import yaml
import os
import subprocess
from pathlib import Path
from common import *
from config import TASK_CONFIG, MODELS_CONFIG
from datetime import datetime

def save_dataset(conversations: list, output_path: str):
    """Save conversations to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(conversations, f, indent=2)
    print(f"Saved {len(conversations)} examples to {output_path}")

def create_yaml_config(model: str, dataset_name: str, output_dir: str, base_adapter: str = None, config_name: str = "sft") -> str:
    """Create training config from template"""
    # Load template
    with open(f'config/{config_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['model_name_or_path'] = model
    config['dataset'] = f'{dataset_name}'
    config['output_dir'] = output_dir
    if base_adapter:
        config['adapter_name_or_path'] = base_adapter

    # Save updated config
    config_path = f'{output_dir}/{config_name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

# def train_sft(dataset: str, model: str = None, adapter: str = None, base_adapter: str = None, samples_list: list = None, config_name: str = "sft"):
#     """
#     Train SFT model using LLaMA-Factory
#     Args:
#         samples_list: List of lists, where each inner list contains conversation samples
#     """
#     if dataset not in TASK_CONFIG:
#         raise ValueError(f"Dataset {dataset} not found in config")
#     task_config = TASK_CONFIG[dataset]

#     assert samples_list is not None, "samples_list is required"
    
#     # Process each list of samples separately
#     dataset_paths = []
#     model_nickname = model.split('/')[-1]
#     output_dir = f"save/{model_nickname}/{dataset}/labeled" if not adapter else adapter
#     os.makedirs(output_dir, exist_ok=True)
    
#     for idx, samples in enumerate(samples_list):
#         conversations = [
#             format_question_alpaca(conv, format_question_vanilla, task_config)
#                 for conv in samples
#         ]

#         # print(conversations[0])
#         # exit()
        
#         # Save each converted dataset with unique timestamp
#         timestamp = datetime.now().strftime("%d%H%M%S")
#         dataset_name = f'{dataset}_{timestamp}_{idx}'
#         dataset_path = f"{output_dir}/{dataset_name}.json"
#         dataset_absolute_path = str(Path(dataset_path).absolute())
#         dataset_paths.append(dataset_name)
        
#         save_dataset(conversations, dataset_path)

#         # Update dataset info
#         dataset_info_path = 'data/dataset_info.json'
#         with open(dataset_info_path, 'r') as f:
#             dataset_info = json.load(f)
#         dataset_info[dataset_name] = {
#             "file_name": dataset_absolute_path,
#         }
#         with open(dataset_info_path, 'w') as f:
#             json.dump(dataset_info, f, indent=2)

#     # Create config with comma-separated dataset names
#     combined_dataset_name = ','.join(dataset_paths)
#     config_path = create_yaml_config(model, combined_dataset_name, output_dir, base_adapter, config_name)

#     # Train using LLaMA-Factory CLI
#     cmd = ["llamafactory-cli", "train", config_path]
#     print(f"Running command: {' '.join(cmd)}")
#     subprocess.run(cmd)

# TODO FT modify below
def train_sft_factuality(dataset: str, model: str = None, adapter: str = None, base_adapter: str = None, samples_list: list = None, config_name: str = "sft"):
    """
    Train SFT model using LLaMA-Factory
    Args:
        samples_list: List of lists, where each inner list contains conversation samples
    """
    if dataset not in TASK_CONFIG:
        raise ValueError(f"Dataset {dataset} not found in config")
    task_config = TASK_CONFIG[dataset]

    if samples_list == None:
        return "samples_list is required"
    
    # Process each list of samples separately
    dataset_paths = []
    model_nickname = model.split('/')[-1]
    output_dir = f"save/{model_nickname}/{dataset}/labeled" if not adapter else adapter
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, samples in enumerate(samples_list):
        
        conversations = [
            format_question_factuality(conv, format_factuality_vanilla, task_config)
                for conv in samples
        ]

        # Save each converted dataset with unique timestamp
        timestamp = datetime.now().strftime("%d%H%M%S")
        dataset_name = f'{dataset}_{timestamp}_{idx}'
        dataset_path = f"{output_dir}/{dataset_name}.json"
        dataset_absolute_path = str(Path(dataset_path).absolute())
        dataset_paths.append(dataset_name)
        
        save_dataset(conversations, dataset_path)

        # Update dataset info
        dataset_info_path = 'data/dataset_info.json'
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        dataset_info[dataset_name] = {
            "file_name": dataset_absolute_path,
        }
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)

    # Create config with comma-separated dataset names
    combined_dataset_name = ','.join(dataset_paths)
    config_path = create_yaml_config(model, combined_dataset_name, output_dir, base_adapter, config_name)

    # Train using LLaMA-Factory CLI
    cmd = ["llamafactory-cli", "train", config_path]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    fire.Fire(train_sft_factuality)
