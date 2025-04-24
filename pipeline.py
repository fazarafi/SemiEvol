import os
import fire
import pandas as pd
import random
import copy
import numpy as np
from train_sft import train_sft
from common import *
from eval import Evaluator, EvalConfig
from eval import eval_model
from config import TASK_CONFIG, MODELS_CONFIG
import datetime
from tqdm import tqdm

def calculate_entropy(probs):
    prob_list = np.array(probs)
    entropy = - np.sum(prob_list) / len(prob_list)
    return entropy

def configure_model(model: str) -> dict:
    """Configure model settings"""
    if model in MODELS_CONFIG:
        config = MODELS_CONFIG[model]
        if 'url' in config:
            os.environ['LLM_BASE_URL'] = config['url']
        if 'OPENAI_API_KEY' in config:
            os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
        return config
    return {
        'name': model
    }

def load_samples(data_path: str, task_config: dict) -> list:
    df = pd.read_csv(data_path)
    samples = []
    for _, row in df.iterrows():
        sample = row.to_dict()
        sample.update({
            'question_type': task_config['question_type'],
            'additional_prompt': task_config['additional_prompt']
        })
        samples.append(sample)
    return samples

def prepare_data(task, task_config, labeled_path=None, unlabeled_path=None, output_dir=None):
    """Prepare and embed data for inference"""
    if not labeled_path:
        labeled_path = f'data/{task}/labeled.csv'
    if not unlabeled_path:
        unlabeled_path = f'data/{task}/unlabeled.csv'
    
    # nr.embed_data_path(labeled_path, f'{output_dir}/{task}_labeled')
    
    labeled_data = load_samples(labeled_path, task_config)
    unlabel_data = load_samples(unlabeled_path, task_config)
    
    return labeled_data, unlabel_data

def run_multiple_inference(all_data: list, num_infer: int, model_config: dict, task: str, base_adapter: str = None) -> list:
    """Run multiple inferences with progress tracking"""
    all_predictions = []
    
    for i in tqdm(range(num_infer), desc="Running inferences"):
        current_seed = int(datetime.datetime.now().timestamp() * 1000) + i
        random.seed(current_seed)
        
        print(f"Running inference {i+1} with seed {current_seed}")

        eval_instance = Evaluator(
            task=task,
            config=EvalConfig(
                model=model_config.get('name', ''),
                temperature=1.0,
                max_tokens=1024,
                logprobs=True,
                lora_path=base_adapter,
                seed=current_seed
            ),
            samples=copy.deepcopy(all_data)
        )
        
        _ = eval_instance.run_inference(
            format_fn=format_question_vanilla,
            extract_fn=extract_result
        )
        all_predictions.append(eval_instance.samples)

        del eval_instance
        clear_mem()
    
    return all_predictions

def process_results(unlabel_data, inference_list, num_infer=4):
    """Process inference results to generate pseudo-labels"""
    save_data = copy.deepcopy(unlabel_data)
    num_examples = len(unlabel_data)
    
    conf_samples = []
    unconf_samples = []
    # unconsis_indexs = []

    for idx in range(num_examples):
        pred_list = []
        for i in range(num_infer):
            pred = inference_list[i][idx]['PredAnswer']
            if type(pred) == list:
                pred = str(pred[0])
            pred_list.append(pred)

        entropy = calculate_entropy(inference_list[0][idx]['logprobs'])

        if len(set(pred_list)) > 1:
            save_data[idx]['consist'] = 0
            save_data[idx]['entropy'] = entropy
            save_data[idx]['PredAnswers'] = pred_list
            save_data[idx]['Preds'] = [inference_list[i][idx]['Pred'] for i in range(num_infer)]
            unconf_samples.append(save_data[idx])
        else:
            save_data[idx]['PseudoLabel'] = pred_list[0]
            save_data[idx]['consist'] = 1
            save_data[idx]['entropy'] = entropy
            conf_samples.append(save_data[idx])
    
    print(f'Consistent Rate: {len(conf_samples) / num_examples:.4f}')
    print(f'Inconsistent Rate: {len(unconf_samples) / num_examples:.4f}')
    
    return conf_samples, unconf_samples

def resolve_inconsistencies(unconf_samples, model_config, task, base_adapter=None):
    """Resolve inconsistent predictions with additional inference"""
    if not unconf_samples:
        return []

    eval_instance = Evaluator(
            task=task,
            config=EvalConfig(
                model=model_config.get('name', ''),
                temperature=1.0,
                max_tokens=4096,
                logprobs=True,
                lora_path=base_adapter,
        ),
        samples=copy.deepcopy(unconf_samples)
    )

    _ = eval_instance.run_inference(
        format_fn=format_reflection,
        extract_fn=extract_result
    )

    unconsis_preds = eval_instance.samples

    for i in range(len(unconsis_preds)):
        unconsis_preds[i]['PseudoLabel'] = unconsis_preds[i]['PredAnswer']
        unconsis_preds[i]['entropy'] = calculate_entropy(unconsis_preds[i]['logprobs'])
    
    entropy_values = [s['entropy'] for s in unconsis_preds]
    entropy_threshold = np.percentile(entropy_values, 30)
    resolved_samples = [s for s in unconsis_preds if s['entropy'] < entropy_threshold]

    return resolved_samples

def calculate_accuracy(save_data):
    """Calculate accuracy of pseudo-labels"""
    for i, s in enumerate(save_data):
        if "answer" in s and "PseudoLabel" in s:
            score = 1.0 if s["PseudoLabel"] == s["answer"] else 0.0
            save_data[i]["Accuracy"] = score
    
    return save_data

def save_results(save_data, task, model, output_dir=None):
    """Save results to CSV"""
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        output_dir = f"save/{model}"
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{task}_pseudo_{timestamp}.csv")
    
    save_df = pd.DataFrame(save_data)
    save_df.to_csv(save_path, index=False)
    print(f"Saved pseudo-labels to {save_path}")
    
    return save_path

def run_pipeline(
    task: str = "mmlu",
    model: str = "llama3.1",
    num_infer: int = 4,
    base_model_path: str = None,
    output_dir: str = None,
    labeled_path: str = None,
    unlabeled_path: str = None
):
    """
    Run the complete SemiEvol pipeline
    
    Args:
        task: Task name (e.g. 'mmlu', 'arc')
        model: Model name from config
        num_infer: Number of inference iterations
        base_model_path: Path to base model (optional)
        output_dir: Directory to save results (optional)
        labeled_path: Path to labeled data (optional)
        unlabeled_path: Path to unlabeled data (optional)
    """
    
    if task not in TASK_CONFIG:
        raise ValueError(f"Task {task} not found in config")
        
    if model not in MODELS_CONFIG:
        raise ValueError(f"Model {model} not found in config")

    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        output_dir = f"save/{model}/{task}_{timestamp}"
    ## For debug use
    # output_dir = f"save/{model}/{task}_test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the environment
    task_config = TASK_CONFIG[task]
    model_config = configure_model(model)
    model_path = base_model_path if base_model_path else model_config['name']
    
    print("\n=== Step 1: Training SFT Model ===")
    sft_output_dir = os.path.join(output_dir, "sft")
    
    # Use paths from task config if not provided
    labeled_data, unlabel_data = prepare_data(task, task_config, labeled_path, unlabeled_path, output_dir)
    
    # Train the SFT model if not already trained
    if not os.path.exists(sft_output_dir):
        train_sft(
            dataset=task,
            model=model_path,
            adapter=sft_output_dir,
            samples_list=[labeled_data]
        )
    else:
        print(f"SFT model already exists at {sft_output_dir}, skipping training")

    # Step 2: Run multiple inferences
    print(f"\n=== Step 2: Running Inference ({num_infer} times) ===")
    inference_list = run_multiple_inference(unlabel_data, num_infer, model_config, task, sft_output_dir)
    
    # Step 3: Process results
    print("\n=== Step 3: Processing Inference Results ===")
    conf_samples, unconf_samples = process_results(unlabel_data, inference_list, num_infer)
    
    # Step 4: Resolve inconsistencies
    print("\n=== Step 4: Resolving Inconsistent Predictions ===")
    resolved_samples = resolve_inconsistencies(
        unconf_samples, model_config, task, sft_output_dir
    )

    print("\n=== Step 5: Training Clean SFT Model ===")
    semievol_output_dir = os.path.join(output_dir, f"semi_evol")
    
    train_sft(
        dataset=task,
        model=model_path,
        adapter=semievol_output_dir,
        base_adapter=sft_output_dir,
        samples_list=[conf_samples + resolved_samples],
        config_name="sft-ft"
    )

    print("\nEvaluating SemiEvol Model:")
    eval_model(
        task=task,
        model=model_path,
        adapter=semievol_output_dir
    )

if __name__ == "__main__":
    fire.Fire(run_pipeline)
