from argparse import ArgumentParser
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
import logging
from threading import Lock
import random
import math
import gc
import uuid

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from peft import (
    get_peft_model_state_dict, 
    get_peft_model, 
)
from peft.peft_model import PeftModelForCausalLM
from accelerate import Accelerator, InitProcessGroupKwargs
from trl import DataCollatorForCompletionOnlyLM
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from datasets import Dataset
from dotenv import load_dotenv
import numpy as np
import openai

from rollout import rollout
from replay_buffers import UnitSizeReplayBuffer
from constants import RESPONSE_TEMPLATE, INSTRUCTION_TEMPLATE, OUTPUT_DIR, EPSILON
from configs import MAP_VERSION_TO_TRAIN_CONFIG
from common_types import InferenceConfig
from common_utils import APIWrapper, logprobs_from_logits


def save_all(
    save_dir: str,
    accelerator: Accelerator,
    model, 
    optimizer,
    replay_buffer,
):
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f'optimizer_{accelerator.process_index}.pt'))
    replay_buffer.save(os.path.join(save_dir, f'replay_buffer_{accelerator.process_index}.json'))
    if accelerator.is_main_process:
        model.save_pretrained(save_dir)
        os.remove(os.path.join(save_dir, 'adapter_model.safetensors'))
        lora_state_dict = get_peft_model_state_dict(unwrapped_model, state_dict)

        save_file(lora_state_dict, os.path.join(save_dir, 'adapter_model.safetensors'))


def main():
    parser = ArgumentParser()
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--max_workers', type=int, default=256)
    parser.add_argument('--name', type=str, required=True)

    args = parser.parse_args()
    train_config = MAP_VERSION_TO_TRAIN_CONFIG[args.version]

    BATCH_SIZE = train_config.batch_size
    GRAD_ACC_STEPS = BATCH_SIZE // train_config.minibatch_size
    TRAJS_DIR = os.path.join(OUTPUT_DIR, args.name, 'trajs')
    BATCH_SIZE_PER_GPU = BATCH_SIZE // args.num_gpus

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name, 
        truncation_side='right',
        model_max_length=train_config.max_train_context_length,
        trust_remote_code=True
    )
    tokenizer.pad_token = '<|finetune_right_pad_id|>'

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE, 
        instruction_template=INSTRUCTION_TEMPLATE, 
        tokenizer=tokenizer, 
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACC_STEPS // args.num_gpus, 
        project_dir=os.path.join(OUTPUT_DIR, args.name),
        kwargs_handlers=[
            InitProcessGroupKwargs(
                timeout=datetime.timedelta(seconds=720000)
            )
        ]
    )

    # setup logging
    debug_logger = logging.getLogger('debug')
    info_logger = logging.getLogger('info')
    debug_logger.setLevel(logging.DEBUG)
    info_logger.setLevel(logging.INFO)

    os.makedirs(os.path.join(OUTPUT_DIR, args.name, 'logs'), exist_ok=True)
    debug_logger.addHandler(logging.FileHandler(os.path.join(OUTPUT_DIR, args.name, 'logs', f'debug_{accelerator.process_index}.log')))
    info_logger.addHandler(logging.FileHandler(os.path.join(OUTPUT_DIR, args.name, 'logs', f'info_{accelerator.process_index}.log')))

    debug_lock = Lock()
    info_lock = Lock()

    def my_log(msg, level: str):
        if level == 'debug':
            with debug_lock:
                debug_logger.debug(msg)
        elif level == 'info':
            with info_lock:
                info_logger.info(msg)
        else:
            raise ValueError(f'Invalid log level: {level}')
        
    resume_ckpt_dir = None
    resume_ckpt_num = None
    maybe_ckpt_dir = os.path.join(OUTPUT_DIR, args.name, 'models')
    if os.path.exists(maybe_ckpt_dir):
        ckpt_dirs = [int(d.split('-')[-1]) for d in os.listdir(maybe_ckpt_dir) if d.startswith('checkpoint-')]
        if len(ckpt_dirs) > 0:
            latest_ckpt = max(ckpt_dirs)
            resume_ckpt_dir = os.path.join(maybe_ckpt_dir, f'checkpoint-{latest_ckpt}')
            resume_ckpt_num = latest_ckpt

    base_model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        attn_implementation='flash_attention_2'
    )
    base_model.gradient_checkpointing_enable()
    if resume_ckpt_dir is not None:
        lora_model = PeftModelForCausalLM.from_pretrained(
            base_model, 
            resume_ckpt_dir, 
            is_trainable=True
        )
    else:
        lora_model = get_peft_model(base_model, train_config.lora_config)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in lora_model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": train_config.weight_decay,
        },
        {
            "params": [p for n, p in lora_model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        params=optimizer_grouped_parameters, 
        lr=train_config.lr, 
        weight_decay=train_config.weight_decay
    )

    lora_model, optimizer = accelerator.prepare(
        lora_model, optimizer
    )
    if resume_ckpt_dir is not None:
        state_dict_list = []
        for i in range(accelerator.num_processes):
            state_dict = torch.load(os.path.join(resume_ckpt_dir, f'optimizer_{i}.pt'))
            state_dict_list.append(state_dict)
        optimizer.load_state_dict(state_dict_list)
    lora_model.train()

    replay_buffer = UnitSizeReplayBuffer(
        train_config.unit_replay_buffer_size, 
        seed=random.randint(0, 1000000)
    )
    if resume_ckpt_dir is not None:
        replay_buffer = UnitSizeReplayBuffer.load(os.path.join(resume_ckpt_dir, f'replay_buffer_{accelerator.process_index}.json'))
    
    VLLM_SERVER_HOST = os.getenv('VLLM_SERVER_HOST')
    VLLM_SERVER_PORT = os.getenv('VLLM_SERVER_PORT')
    VLLM_MODEL_PORT = os.getenv('VLLM_MODEL_PORT')
    VLLM_SERVER_URL = f'http://{VLLM_SERVER_HOST}:{VLLM_SERVER_PORT}'

    vllm_api_wrapper = APIWrapper(VLLM_SERVER_URL)
    vllm_inited = False

    for global_step in range(train_config.global_steps):
        if (resume_ckpt_num is not None) and global_step < resume_ckpt_num:
            continue
        
        my_log(f'Global step: {global_step}', 'info')

        if global_step >= train_config.warmup_steps: 
            save_path = os.path.join(OUTPUT_DIR, args.name, 'models', f'checkpoint-{global_step}')
            os.makedirs(save_path, exist_ok=True)
            # we don't save the model on the checkpoint where we resume from
            if (resume_ckpt_num is None) or global_step > resume_ckpt_num:
                save_all(
                    save_path,
                    accelerator, 
                    lora_model,
                    optimizer,
                    replay_buffer,
                )
            accelerator.wait_for_everyone()
        
        if global_step >= train_config.warmup_steps:
            if not vllm_inited:
                if accelerator.is_main_process: 
                    vllm_api_wrapper.api_post(
                        'shutdown',
                        data={}
                    )
                    vllm_api_wrapper.api_post(
                        'set_base_model', 
                        {'base_model': train_config.model_name}
                    )
                    vllm_api_wrapper.api_post(
                        'set_quantization',
                        {'quantization': 'fp8'}
                    )
                vllm_inited = True
            accelerator.wait_for_everyone()

            cur_trajs_dir = os.path.join(TRAJS_DIR, f'global_step_{global_step}')
            if accelerator.is_main_process: 
                vllm_api_wrapper.api_post(
                    'set_lora_dir', 
                    {'lora_dir': os.path.dirname(save_path)}
                )
                vllm_api_wrapper.api_post(
                    'set_checkpoint_list',
                    {'checkpoint_list': [global_step]}
                )
                vllm_api_wrapper.api_post(
                    'set_checkpoint', 
                    {'checkpoint': global_step}
                )
                res_gen = vllm_api_wrapper.api_post_stream(
                    'init', 
                    data={}
                )
                for chunk in res_gen:
                    my_log(chunk, 'debug')

                openai_client = openai.OpenAI(
                    base_url=f'http://{VLLM_SERVER_HOST}:{VLLM_MODEL_PORT}/v1',
                )

                num_rollouts_per_problem = {}
                for problem_id in train_config.problem_ids:
                    num_rollouts_per_problem[problem_id] = 0
                    for root, dirs, files in os.walk(os.path.join(cur_trajs_dir, problem_id)):
                        num_rollouts_per_problem[problem_id] += len(files)
                
                def completion_fn(prompt: str, inference_config: InferenceConfig):
                    return openai_client.completions.create(
                        model=f'checkpoint-{global_step}',
                        prompt=prompt,
                        max_tokens=inference_config.max_tokens,
                        temperature=inference_config.temperature,
                        stream=True, 
                        stream_options={'include_usage': True}
                    )

                rollout_args = []
                dataset_lock = Lock()
                for problem_id in train_config.problem_ids:
                    for _ in range(train_config.rollout_per_problem - num_rollouts_per_problem[problem_id]):
                        rollout_args.append({
                            'problem_id': problem_id,
                            'inference_config': train_config.inference_config,
                            'tokenizer': tokenizer,
                            'completion_fn': completion_fn,
                            'dump_path': os.path.join(cur_trajs_dir, problem_id, f'{uuid.uuid4()}.json'),
                            'dataset_lock': dataset_lock
                        })

                random.shuffle(rollout_args)

                executor = ThreadPoolExecutor(args.max_workers)
                futures = [executor.submit(rollout, **args) for args in rollout_args]

                num_passed = 0
                num_failed = 0
                total_passed_time = 0
                for future in as_completed(futures):
                    result = future.result()
                    if result.sol_passed:
                        num_passed += 1
                        total_passed_time += result.sol_exec_time
                    else:
                        num_failed += 1

                    my_log(f'Passed: {num_passed}, Failed: {num_failed}, Avg passed time: {total_passed_time / num_passed if num_passed > 0 else 0}', 'info')

                executor.shutdown(wait=True)

                vllm_api_wrapper.api_post(
                    'shutdown',
                    data={}
                )
            accelerator.wait_for_everyone()
            
            dump_results = []
            for root, dirs, files in os.walk(cur_trajs_dir):
                for file in files:
                    if file.endswith('.json'):
                        try:
                            result = json.load(open(os.path.join(root, file)))
                            result['problem_id'] = os.path.basename(root)
                            dump_results.append(result)
                        except Exception as e:
                            my_log(f'Error loading {os.path.join(root, file)}: {e}', 'debug')
            
            random.shuffle(dump_results)
            for result in dump_results:
                if result['sol_passed']:
                    replay_buffer.add_traj(result['traj'], {
                        'problem_id': result['problem_id'],
                        'num_tokens': result['num_tokens'],
                        'sol_exec_time': result['sol_exec_time'],
                        'sol_passed': result['sol_passed'],
                    })
            
            local_dataset_w_info = replay_buffer.sample(BATCH_SIZE * train_config.local_steps)
            local_dataset = [traj for traj, _ in local_dataset_w_info]

            local_dataset = tokenizer.apply_chat_template(
                conversation=local_dataset,
                add_generation_prompt=True,
                tokenize=True,
                padding='max_length',
                truncation=True,
                max_length=train_config.max_train_context_length, 
                return_tensors='pt', 
                return_dict=True
            )

            local_dataset = {k: v.tolist() for k, v in local_dataset.items()}

            local_dataset['advantage'] = []
            for _, info in local_dataset_w_info:
                problem_id = info['problem_id']
                times = [x['sol_exec_time'] for x in replay_buffer.info_buffer[problem_id]]
                mean_time = sum(times) / len(times)
                std_time = math.sqrt(sum((time - mean_time) ** 2 for time in times) / len(times))
                local_dataset['advantage'].append(
                    -(info['sol_exec_time'] - mean_time) / (std_time + EPSILON)
                )

            local_dataset = Dataset.from_dict(local_dataset)
            local_dataloader = DataLoader(
                local_dataset, 
                batch_size=train_config.minibatch_size, 
                collate_fn=data_collator, 
                shuffle=True
            )
            local_dataloader = accelerator.prepare(local_dataloader)

            batched_loss = 0.0
            for local_step, batch in enumerate(local_dataloader):
                with accelerator.accumulate(lora_model):
                    outputs = lora_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                    )
                    logits = outputs.logits
                    logprobs = logprobs_from_logits(logits[:, :-1, :], batch['input_ids'][:, 1:])
                    valid_indices = (batch['labels'][:, 1:] != -100).nonzero(as_tuple=True)[1]
                    valid_logprobs = logprobs[:, valid_indices]
                    weights = torch.exp(batch['advantage'][:, None] / train_config.awr_temp)
                    my_log(f'Weights: {weights}', 'debug')
                    loss = -(valid_logprobs * weights).mean(dim=-1).mean()
                    my_log(f'Loss: {loss.item()}', 'debug')
                    batched_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if (local_step+1) % BATCH_SIZE_PER_GPU == 0:
                    my_log(f'Local step: {local_step//BATCH_SIZE_PER_GPU}, Loss: {batched_loss / BATCH_SIZE_PER_GPU}', 'info')
                    batched_loss = 0.0

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    load_dotenv()
    main()