from typing import Callable, Generator
from string import Template
from threading import Lock
from datasets import load_dataset
from transformers import AutoTokenizer
from common_types import InferenceConfig, RolloutResult
from sandbox import Sandbox
import json
import os
import re

dataset = None
NUM_RUNS = 4

def rollout(
    problem_id: str, 
    inference_config: InferenceConfig, 
    tokenizer: AutoTokenizer,
    completion_fn: Callable[[str, InferenceConfig], Generator],
    dump_path: str | None = None, 
    dataset_lock: Lock = Lock()
) -> RolloutResult:
    global dataset
    if dataset is None:
        dataset_lock.acquire()
        if dataset is None:
            tmp_dataset = load_dataset("parquet", data_files="datasets/eval.parquet")
            tmp_dataset = tmp_dataset["train"]
            dataset = {}
            for problem in tmp_dataset:
                idx = problem["id"]
                dataset[idx] = problem
                dataset[idx]["text"] = problem["pretty_content"][0]
                dataset[idx]["prompt"] = problem["prompt"]
                dataset[idx]["test_cases"] = json.loads(problem["test_cases"])
                dataset[idx]["convert_offline"] = problem["convert_offline"]
                dataset[idx]["evaluate_offline"] = problem["evaluate_offline"]
                dataset[idx]["entry_point"] = problem["entry_point"]
                dataset[idx]["solution_index"] = 0
                dataset[idx]["solution"] = problem["solutions"][0]["solution"]
                dataset[idx]["timeout"] = 32
        dataset_lock.release()

    prompt_format = inference_config.prompt_format
    system_prompt_path = f'prompts/{prompt_format}/system.txt'
    user_prompt_path = f'prompts/{prompt_format}/user.txt'

    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    with open(user_prompt_path, 'r') as f:
        user_prompt = f.read()
    user_prompt = Template(user_prompt).substitute(problem_text=dataset[problem_id]["text"], prompt=dataset[problem_id]["prompt"])

    msg_list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        msg_list,
        tokenize=False, 
        add_generation_prompt=True
    )

    completion = completion_fn(prompt, inference_config)

    num_tokens = 0
    new_msg = ''
    for chunk in completion:
        if chunk.usage:
            num_tokens = chunk.usage.completion_tokens
        if len(chunk.choices) > 0:
            content = chunk.choices[0].text
            new_msg += content

    def parse_last_code_block(markdown_text):
        pattern = r"```(?:\w*\n)?(.*?)```"
        blocks = re.findall(pattern, markdown_text, flags=re.DOTALL)
        return blocks[-1] if blocks else None

    code = parse_last_code_block(new_msg)

    sample = {
        "solution": code,
        "convert_offline": dataset[problem_id]["convert_offline"],
        "evaluate_offline": dataset[problem_id]["evaluate_offline"],
        "entry_point": dataset[problem_id]["entry_point"],
        "test_cases": dataset[problem_id]["test_cases"],
        "solution_index": dataset[problem_id]["solution_index"],
        "timeout": dataset[problem_id]["timeout"],
    }

    sandbox = Sandbox()
    results = sandbox.run_samples(samples=[sample] * NUM_RUNS, n_workers=1)
    
    print(f"results: {results}", flush=True)
    sol_passed = any(result["result"] == 'passed' for result in results)
    sol_exec_time = sum(result["runtime"] for result in results) / NUM_RUNS

    if dump_path is not None:
        msg_list.append({"role": "assistant", "content": new_msg})

        # make parent dir if not exists
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, 'w') as f:
            dump_data = {
                
                'traj': msg_list,
                'num_tokens': num_tokens,
                'sol_passed': sol_passed,
                'sol_exec_time': sol_exec_time,
            }
            json.dump(dump_data, f, indent=4)

    return RolloutResult(
        problem_id=problem_id,
        num_tokens=num_tokens,
        sol_passed=sol_passed,
        sol_exec_time=sol_exec_time
    )
    