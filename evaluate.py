from datasets import load_dataset
from sandbox import Sandbox
import json

data = load_dataset("Elfsong/Mercury")["eval"] # Dataset object

def evaluate(problem_id, src):
    problems = data.filter(lambda x: x["id"] == problem_id)
    if len(problems) == 0:
        raise ValueError(f"Problem {problem_id} not found")
    instance = problems[0]

    sandbox = Sandbox()
    
    sample = {
        "solution": src,
        "convert_offline": instance['convert_offline'],
        "evaluate_offline": instance['evaluate_offline'],
        "entry_point": instance['entry_point'],
        "test_cases": json.loads(instance['test_cases']),
        "solution_index": 0,
        "timeout": 30,
    }

    result = sandbox.run_sample(sample)
    print(result)

    return {
        "slug_name": instance["slug_name"],
        "result":  result,
        # "solution": src,
    }

model_solution = data[0]["solutions"][0]["solution"]
# print(data[0])
# print(model_solution)


if __name__ == "__main__":
    print(evaluate("54", model_solution))
