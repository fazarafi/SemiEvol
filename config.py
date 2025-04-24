from common import *

MODELS_CONFIG = {
    "llama3.1": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "adapter": ""
    },
    "llama3.2": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "adapter": ""
    },
    "gemma2": {
        "name": "google/gemma-2-9b-it",
        "adapter": ""
    },
}


TASK_CONFIG = {
    "mmlu": {
        "dataset_name": "mmlu",
        "test_path": "./data/mmlu/test.csv",
        "labeled_path": "./data/mmlu/labeled.csv",
        "unlabeled_path": "./data/mmlu/unlabeled.csv",
        "question_type": "multi-choice",
        "additional_prompt": "Your response should be of the following format: 'Answer: LETTER' (without quotes, The LETTER should be in A,B,C,D).",
        "check_fn": check_answer
    },
    "mmlu_pro": {
        "dataset_name": "mmlu_pro",
        "test_path": "./data/mmlu_pro/test.csv",
        "labeled_path": "./data/mmlu_pro/labeled.csv",
        "unlabeled_path": "./data/mmlu_pro/unlabeled.csv",
        "question_type": "multi-choice",
        "additional_prompt": "",
        "check_fn": check_answer
    },
    "arc": {
        "dataset_name": "arc",
        "test_path": "./data/arc/test.csv",
        "labeled_path": "./data/arc/labeled.csv",
        "unlabeled_path": "./data/arc/unlabeled.csv",
        "noisy_path": "./data/arc/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "",
        "check_fn": check_answer
    },
    "fpb": {
        "dataset_name": "fpb",
        "test_path": "./data/fpb/test.csv",
        "labeled_path": "./data/fpb/labeled.csv",
        "unlabeled_path": "./data/fpb/unlabeled.csv",
        "noisy_path": "./data/fpb/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "",
        "check_fn": check_answer
    },
    "pubmedqa": {
        "dataset_name": "pubmedqa",
        "test_path": "./data/pubmedqa/test.csv",
        "labeled_path": "./data/pubmedqa/labeled.csv",
        "unlabeled_path": "./data/pubmedqa/unlabeled.csv",
        "question_type": "multi-choice",
        "additional_prompt": "",
        "check_fn": check_answer
    },
    "usmle": {
        "dataset_name": "usmle",
        "test_path": "./data/usmle/test.csv",
        "labeled_path": "./data/usmle/labeled.csv",
        "unlabeled_path": "./data/usmle/unlabeled.csv",
        "question_type": "multi-choice",
        "additional_prompt": "",
        "check_fn": check_answer
    },
    "convfinqa": {
        "dataset_name": "convfinqa",
        "test_path": "./data/convfinqa/test.csv",
        "labeled_path": "./data/convfinqa/labeled.csv",
        "unlabeled_path": "./data/convfinqa/unlabeled.csv",
        "question_type": "math and value extraction",
        "additional_prompt": "The answer should be in digits.",
        "check_fn": check_answer_value
    },
    "gsm8k": {
        "dataset_name": "gsm8k",
        "test_path": "./data/gsm8k/test.csv",
        "labeled_path": "./data/gsm8k/labeled.csv",
        "unlabeled_path": "./data/gsm8k/unlabeled.csv",
        "question_type": "math",
        "additional_prompt": "The answer should be in digits.",
        "check_fn": check_answer_value
    },
    "drop": {
        "dataset_name": "drop",
        "test_path": "./data/drop/test.csv",
        "labeled_path": "./data/drop/labeled.csv",
        "unlabeled_path": "./data/drop/unlabeled.csv",
        "question_type": "reading comprehension",
        "additional_prompt": "The answer should be a single word or in digits.",
        "check_fn": check_answer_fuzzy
    }
}

