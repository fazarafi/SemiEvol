import random
import pandas as pd
import sys
from pathlib import Path
import json
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from datetime import datetime
import fire
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
# sys.path.append('..')
from common import *
from sft_utils import *
from config import *

# update again

@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    model: str
    temperature: float = 0.5
    max_tokens: int = 1000
    logprobs: bool = True
    lora_path: Optional[str] = None
    seed: int = 0

class ResultSaver:
    """Handle saving evaluation results"""
    def __init__(self, task: str, save_dir: str = "./save"):
        self.task = task
        self.save_dir = Path(save_dir)
        self.test_dir = self.save_dir / "test" / task
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, results: List[Dict], accuracy: float):
        """Save detailed results to CSV"""
        res_df = pd.DataFrame(results)
        res_df.to_csv(self.test_dir / f"{self.task}_{accuracy:.2f}.csv", index=False)
    
    def update_global_results(self, model: str, adapter: Optional[str], accuracy: float):
        """Update global results file"""
        adapter_name = os.path.basename(adapter) if adapter else "no_adapter"
        with open(self.save_dir / "global_res.txt", "a") as f:
            f.write(f"{self.task},{model},{adapter_name},{accuracy:.4f}\n")

class SampleProcessor:
    """Process and prepare evaluation samples"""
    @staticmethod
    def load_samples(task_config: Dict, num_samples: Optional[int] = None) -> List[Dict]:
        df = pd.read_csv(task_config['test_path'])
        if num_samples:
            df = df.sample(n=num_samples, random_state=0)
            
        samples = []
        for _, row in df.iterrows():
            sample = row.to_dict()
            sample.update({
                'question_type': task_config['question_type'],
                'additional_prompt': task_config['additional_prompt']
            })
            samples.append(sample)
        return samples

    @staticmethod
    def load_samples_factuality(task_config: Dict, num_samples: Optional[int] = None) -> List[Dict]:
        samples = load_processed_dataset(dataset_name="xsum_factuality", set_type="test")
        return samples

class Evaluator:
    """Main evaluation class"""
    def __init__(
        self,
        task: str,
        config: EvalConfig,
        samples: List[Dict],
    ):
        self.task = task
        self.config = config
        self.samples = samples
        self.model = config.model.replace('/', '_')
        self.timestamp = datetime.now().strftime("%m%d_%H")
        self.output_path = f'./save/infer/{task}_{self.model}_{self.timestamp}.json'
        self.gptreq = None
        
    def _init_model(self):
        """Initialize model if not already done"""
        if not self.gptreq:
            self.gptreq = LocalRequest(
                model_path=self.config.model,
                lora_path=self.config.lora_path
            )
    
    def _process_responses(self, res_list: List[Dict], extract_fn: Callable):
        """Process model responses"""
        assert len(res_list) == len(self.samples)
        for i, (sample, result) in enumerate(zip(self.samples, res_list)):
            response = result['response']

            # print("@@@@@@@@@")
            # print("----")
            # # print(response)

            # print("self.samples[",i,"] BEFORE:", self.samples[i])
            # print("->>")
            # print("type", type(self.samples[i]))
            # print("Pred before::", response)
            # print("PredAnswer before::", extract_fn(response))
            # print("->>")
            

            self.samples[i]["Pred"] = response
            # print("self.samples[",i,"] AFTER:", self.samples[i])
            # print("Pred", self.samples[i]["Pred"])
            
            self.samples[i]["PredAnswer"] = extract_fn(response)
            # print("PredAnswer", self.samples[i]["PredAnswer"])

            # s = dict(self.samples[i])  # make a copy just for visibility
            # s["Pred"] = response
            # s["PredAnswer"] = extract_fn(response)
            # print("FORCED UPDATE:", s)
            # self.samples[i] = s
            # print("self.samples[",i,"] AFTER:", self.samples[i])
            
            
            if "logprobs" in result:
                self.samples[i]["logprobs"] = result["logprobs"]
            # print("(after) sample ",i,"-th:",self.samples[i])

        print("======================")
        print("PRED SAMPLE", self.samples[0])
        print("======================")
    
    def run_inference(self, format_fn: Callable, extract_fn: Callable) -> float:
        """Run inference and calculate accuracy"""
        print(f'Formatting {len(self.samples)} questions ...')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            instances = list(tqdm(
                executor.map(
                    lambda x: [{"role": "user", "content": format_fn(x)}],
                    self.samples
                ),
                total=len(self.samples)
            ))
        
        print('Beginning inference ...')
        self._init_model()
        config_dict = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "logprobs": self.config.logprobs,
            "seed": self.config.seed
        }
        res_list = self.gptreq.batch_req(instances, config_dict, save=True, save_dir=self.output_path)
        self._process_responses(res_list, extract_fn)
        
        return self.calculate_accuracy(TASK_CONFIG[self.task]['check_fn'])
    
    def calculate_accuracy(self, check_fn: Callable) -> float:
        """Calculate accuracy of predictions"""
        
        # print("======================")
        # print("Len Sample:", len(self.samples))
        # print("sample[0]:", self.samples[0])
        # print("======================")
        # exit()
        scores = [
            1.0 if check_fn(s['Pred'], 1) else 0.0 # 1 -> s["answer"]
            for s in self.samples
        ]
        return np.mean(scores)

def eval_model(
    task: str,
    model: str,
    adapter: Optional[str] = None,
    num_samples: Optional[int] = None
):
    """
    Run evaluation with specified parameters
    
    Args:
        task: Task name (e.g. 'arc')
        model: Model path or name
        adapter: Optional LoRA adapter path
        num_samples: Optional number of samples to evaluate
    """
    if task not in TASK_CONFIG:
        raise ValueError(f'Task {task} not found in TASK_CONFIG')

    if model in MODELS_CONFIG:
        adapter = MODELS_CONFIG[model]['adapter']
        model = MODELS_CONFIG[model]['name']
    
    # random seed
    seed = random.randint(0, 1000000)
    
    # Initialize components
    config = EvalConfig(
        model=model,
        lora_path=adapter,
        seed=seed
    )
    
    samples = SampleProcessor.load_samples_factuality(TASK_CONFIG[task], num_samples)
    evaluator = Evaluator(task, config, samples)
    saver = ResultSaver(task)
    
    # Run evaluation
    accuracy = evaluator.run_inference(
        format_fn=format_factuality_vanilla,
        extract_fn=extract_result
    )
    
    print(f'Accuracy: {accuracy}')
    
    # Save results
    saver.save_results(evaluator.samples, accuracy)
    saver.update_global_results(model, adapter, accuracy)

if __name__ == "__main__":
    fire.Fire(eval_model)

__all__ = ['Evaluator', 'EvalConfig', 'ResultSaver', 'SampleProcessor']