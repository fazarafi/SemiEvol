from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import time
import json
import os 
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# GLOBAL_REASONING_ENHANCE_INDEX = 0

def get_tmp_file_path():
    created_time = time.time()
    created_time = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    res_file_path = './tmp/batch_res_' + timestamp + '.jsonl'
    return res_file_path


class LocalRequest:
    def __init__(self, model_path=None, lora_path=None):
        # Initialize base model with LoRA support if lora_path is provided
        enable_lora = lora_path is not None
        self.model = LLM(
            model=model_path,
            enable_lora=enable_lora,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=50000,
            dtype="float16"
        )
        
        # Store LoRA config if provided
        self.lora_request = None
        if lora_path:
            self.lora_request = LoRARequest(
                "lora_adapter",
                1,
                lora_path
            )
            print(f"LoRA adapter loaded from: {lora_path}")
        else:
            print("No LoRA adapter loaded - using base model only")

        self.results_list = []
        # self.global_reasoning_enhance_index = 0
        self.data_lines = []

    # def single_req(self, msg, config, logprobs=False):
    #     sampling_params = SamplingParams(
    #         temperature=config.get('temperature', 0),
    #         max_tokens=config.get('max_tokens', 100),
    #         logprobs=logprobs
    #     )

    #     # Use chat() method instead of generate()
    #     outputs = self.model.chat(
    #         messages=msg,
    #         sampling_params=sampling_params,
    #         lora_request=self.lora_request
    #     )
        
    #     output = outputs[0]
    #     if logprobs:
    #         return output.outputs[0].text, output.outputs[0].logprobs
    #     return output.outputs[0].text

    def batch_req(self, messages_list, config, save=False, save_dir=''):
        sampling_params = SamplingParams(
            temperature=config.get('temperature', 0),
            max_tokens=config.get('max_tokens', 100),
            logprobs=config.get('logprobs', 0),
            seed=config.get('seed', 0)
        )

        # print sampling_params
        print(sampling_params)

        # Ensure each message in the list is properly formatted
        formatted_messages = []
        for messages in messages_list:
            if isinstance(messages, list):
                formatted_messages.append(messages)
            else:
                formatted_messages.append([messages])

        outputs = self.model.chat(
            messages=formatted_messages,
            sampling_params=sampling_params,
            lora_request=self.lora_request,
            use_tqdm=True
        )

        res_list = []
        for output in outputs:
            result = {
                "response": output.outputs[0].text,
            }
            if config.get('logprobs'):
                try:
                    # 添加错误处理
                    logprobs = output.outputs[0].logprobs
                    result["logprobs"] = [next(iter(p.values())).logprob for p in logprobs]
                except (AttributeError, TypeError) as e:
                    print(f"Warning: Could not process logprobs: {e}")
                    result["logprobs"] = []
            res_list.append(result)

        self.results_list = res_list

        if save:
            output_path = save_dir if save_dir else get_tmp_file_path()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as file:
                for obj in self.results_list:
                    file.write(json.dumps(obj) + '\n')

        return res_list
