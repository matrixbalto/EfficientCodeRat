from dataclasses import dataclass, field
from peft import LoraConfig
from constants import DEFAULT_LORA_CONFIG, TRAIN_PROBLEM_IDS

@dataclass
class InferenceConfig:
    temperature: float
    max_tokens: int
    prompt_format: str


@dataclass
class RolloutResult:
    problem_id: str
    num_tokens: int
    sol_passed: bool
    sol_exec_time: float


@dataclass
class TrainConfig:
    model_name: str = 'meta-llama/Llama-3.1-8B-Instruct'
    problem_ids: list[str] = field(default_factory=lambda: TRAIN_PROBLEM_IDS)

    lr: float = 1e-4
    weight_decay: float = 0.01
    awr_temp: float = 1.0
    
    minibatch_size: int = 2
    batch_size: int = 32

    rollout_per_problem: int = 8

    warmup_steps: int = 0
    local_steps: int = 32
    global_steps: int = 999999999
    unit_replay_buffer_size: int = 16
    max_train_context_length: int = 8192

    inference_config: InferenceConfig = field(default_factory=lambda: InferenceConfig(
        temperature=1.0, 
        max_tokens=4096, 
        prompt_format='v1'
    ))

    lora_config: LoraConfig = field(default_factory=lambda: DEFAULT_LORA_CONFIG)