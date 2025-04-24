import re
import json
import ast
import gc
import torch
import pandas as pd

"""
Vanilla Inference
"""

QUERY_TEMPLATE_MULTICHOICE = """
Answer the {question_type} question.
{additional_prompt}

Question: 
{question}

{options_str}
""".strip()

# ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-Z])"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\s\n]+)"

def format_option_str(row):
    if not 'options' in row:
        return ''
    options = row['options']
    if type(options) == str:
        options = json.loads(options)
    options = [f"{chr(65+i)}. {option}" for i, option in enumerate(options)]
    options_str = "\n".join(options)
    return f'Options:\n{options_str}\n'

def format_question_vanilla(row):
    question = row['question']
    options_str = format_option_str(row)
    question_type = row['question_type']
    additional_prompt = row['additional_prompt']
    return QUERY_TEMPLATE_MULTICHOICE.format(question=question, options_str=options_str, question_type=question_type, additional_prompt=additional_prompt)

def get_the_shortest_str_inlist(str_list):
    return min(str_list, key=len)

def extract_result(res):
    # 如果输入为空或None，返回空字符串
    if not res:
        return ''
    
    match = re.search(ANSWER_PATTERN, res)
    extracted_answer = match.group(1) if match else ''
    # ' res[0].upper()
    # if len(extracted_answer) > 1:
    #     extracted_answer = extracted_answer[0].upper()
    # if not extracted_answer in ['A', 'B', 'C', 'D', 'E']:
    #     return ''
    return extracted_answer

def parse_string(s):
    s = s.replace("array(", "").replace(", dtype=object)", "")
    return ast.literal_eval(s)

def clean_string(pl):
    if "' '" in pl or '" "' in pl:
        pl = pl.strip('[]').split("' '")
        pl = [item.strip("' ") for item in pl]
    if isinstance(pl, str) and pl.startswith('['):
        pl = parse_string(pl)
    if isinstance(pl, list):
        pl = get_the_shortest_str_inlist(pl)
    pl = pl.replace('"', '').replace("'", '').replace('[]', '')
    pl = pl.replace('.', '').strip()

    return pl

def pack_answer(row):
    if 'PseudoLabel' in row:
        pl = row['PseudoLabel']
    else:
        pl = row['answer']

    if type(pl) != str:
        pl = parse_string(row['answers_spans'])['spans'][0]
    
    pl = clean_string(pl)
    return f'Answer: {pl}'

"""
Check Answer
"""

def check_consistency(res, gt):
    pred = extract_result(res)
    gt_pred = extract_result(gt)
    return pred == gt_pred

def check_answer(res, gt):
    pred = extract_result(res)
    # if length not same, cut to the same length
    if len(pred) < len(gt):
        pred = pred[:len(gt)]
    elif len(pred) > len(gt):
        gt = gt[:len(pred)]
    return pred == gt

# for value inference
def normoalize_num(num):
    def eval_num(num):
        num = num.replace('%','/100').replace(',','')
        try:
            num = eval(num)
        except Exception as e:
            num = float('inf')
            pass
        return num
    VALUE_PATTERRN = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?[%]*"
    val_reg = re.compile(VALUE_PATTERRN)
    return [eval_num(num) for num in val_reg.findall(num)]


def check_value_equal(res_arr, gt_arr):
    import math
    for gt_num in gt_arr:
        for pred_num in res_arr:
            if math.isclose(pred_num, gt_num, rel_tol=1e-2):
                return True
    return False

def check_answer_value(res, gt):
    pred = normoalize_num(extract_result(res))
    gt = normoalize_num(gt)
    return check_value_equal(pred, gt)

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    import string
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1

def check_answer_fuzzy(res: str, gt: list):
    pred = extract_result(res)
    match_list = [fuzzy_match(pred, gt_item) for gt_item in gt]
    return True in match_list

def convert_to_conversation_format(row, task_config):
    # Get question_type and additional_prompt from task config
    row['question_type'] = task_config['question_type']
    row['additional_prompt'] = task_config['additional_prompt']
    
    messages = [
        {"role": "user", "content": format_question_vanilla(row)},
        {"role": "assistant", "content": pack_answer(row)}
    ]

    if "system_prompt" in task_config:
        messages.insert(0, {"role": "system", "content": ""})

    return messages

def format_question_alpaca(row, format_fn=format_question_vanilla, task_config=None):
    row['question_type'] = task_config['question_type']
    row['additional_prompt'] = task_config['additional_prompt']
    input_text = format_fn(row)
    output_test = pack_answer(row)
    return {
        "instruction": input_text,
        "input": '',
        "output": output_test
    }

def clear_mem(verbose: bool = False) -> None:
    """
    This function is used to clear the memory allocated by PyTorch.
    It does so by calling the garbage collector to release unused GPU memory.
    After clearing the memory, it prints the current amount of memory still allocated by PyTorch (post-clean).

    Parameters:
    verbose (bool): Whether to print additional information.
    """

    gc.collect()
    torch.cuda.empty_cache()

    def try_attr(x, a):
        try:
            return getattr(x, a)
        except Exception:
            return None

    if verbose:
        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)

    print(f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

FEW_SHOT_SYSTEM = """
You are an expert in the multiple choice question. Below are some examples of questions and their corresponding answer.

{reference}
""".strip()

REFLECTION = """Here are the multiple answers of the multiple choice question.  Please consider them thoroughly and give me the correct answer. Your response should be of the following format: 'Answer: LETTER' (without quotes).

Question: 
{question}

Options:
{options}

Multiple Answers:
{answers}

Now, please directly give me the final correct answer:
"""


def format_reflection(data):
    question = data['question']
    options = data['options']
    preds = '\n'.join(data['Preds'])
    return REFLECTION.format(question=question, options=options, answers=preds)

