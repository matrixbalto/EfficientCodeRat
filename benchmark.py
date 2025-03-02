from rollout import rollout
from common_types import InferenceConfig, RolloutResult
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from dotenv import load_dotenv
from openai import OpenAI
from threading import Lock
from constants import EVAL_PROBLEM_IDS, TRAIN_PROBLEM_IDS
from transformers import AutoTokenizer
import os
import json
import time
import uuid
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--port_shift', type=int, default=0)
    parser.add_argument('--problem_set', type=str, choices=['eval', 'train', 'all'], default='eval',
                        help='Which problem set to evaluate on: eval_problems, train_problems, or all')
    parser.add_argument('--samples_per_problem', type=int, default=4,
                        help='Number of samples to take from each problem')
    parser.add_argument('--output_file', type=str, default='benchmark_results.json',
                        help='Path to save the benchmark results')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of worker threads')

    args = parser.parse_args()

    VLLM_MODEL_PORT = os.getenv('VLLM_MODEL_PORT')
    openai_client = OpenAI(
        base_url=f'http://0.0.0.0:{int(VLLM_MODEL_PORT) + args.port_shift}/v1',
    )

    inference_config = InferenceConfig(
        temperature=0.5,
        max_tokens=4096, 
        prompt_format='v1'
    )

    # Determine which problem set to use
    if args.problem_set == 'eval':
        problem_ids = EVAL_PROBLEM_IDS
    elif args.problem_set == 'train':
        problem_ids = TRAIN_PROBLEM_IDS
    else:  # 'all'
        problem_ids = list(set(EVAL_PROBLEM_IDS + TRAIN_PROBLEM_IDS))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Create output directory for detailed results
    output_dir = f"benchmark_results_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)

    def completion_fn(prompt: str, inference_config: InferenceConfig):
        return openai_client.completions.create(
            model=args.model_path,
            prompt=prompt,
            max_tokens=inference_config.max_tokens,
            temperature=inference_config.temperature,
            stream=True,
            stream_options={'include_usage': True}
        )

    # Prepare rollout arguments
    rollout_args = []
    dataset_lock = Lock()
    
    for problem_id in problem_ids:
        problem_dir = os.path.join(output_dir, problem_id)
        os.makedirs(problem_dir, exist_ok=True)
        
        for _ in range(args.samples_per_problem):
            rollout_args.append({
                'problem_id': problem_id,
                'inference_config': inference_config,
                'tokenizer': tokenizer,
                'completion_fn': completion_fn,
                'dump_path': os.path.join(problem_dir, f'{uuid.uuid4()}.json'),
                'dataset_lock': dataset_lock
            })

    # Run rollouts
    results = {}
    for problem_id in problem_ids:
        results[problem_id] = {
            'passed': 0,
            'failed': 0,
            'avg_exec_time': 0,
            'total_exec_time': 0
        }

    print(f"Running benchmark on {len(problem_ids)} problems with {args.samples_per_problem} samples per problem...")
    
    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    futures = []
    
    for args_dict in rollout_args:
        futures.append(executor.submit(rollout, **args_dict))
    
    # Process results as they complete
    for future in tqdm(futures):
        result = future.result()
        problem_id = result.problem_id
        
        if result.sol_passed:
            results[problem_id]['passed'] += 1
            results[problem_id]['total_exec_time'] += result.sol_exec_time
        else:
            results[problem_id]['failed'] += 1
    
    # Calculate average execution time for passed attempts
    for problem_id in results:
        if results[problem_id]['passed'] > 0:
            results[problem_id]['avg_exec_time'] = results[problem_id]['total_exec_time'] / results[problem_id]['passed']
    
    # Calculate overall statistics
    total_passed = sum(results[p]['passed'] for p in results)
    total_attempts = sum(results[p]['passed'] + results[p]['failed'] for p in results)
    overall_pass_rate = total_passed / total_attempts if total_attempts > 0 else 0
    
    # Prepare final results
    final_results = {
        'overall': {
            'total_problems': len(problem_ids),
            'total_attempts': total_attempts,
            'total_passed': total_passed,
            'pass_rate': overall_pass_rate
        },
        'problems': results
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Benchmark completed. Results saved to {args.output_file}")
    print(f"Detailed results saved to {output_dir}")
    print(f"Overall pass rate: {overall_pass_rate:.2%}")


if __name__ == "__main__":
    load_dotenv()
    main()