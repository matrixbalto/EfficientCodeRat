from abc import ABC, abstractmethod
from random import Random
import json


class AbstractReplayBuffer(ABC): 

    @abstractmethod
    def add_traj(self, traj: list[dict], info: dict = {}) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, batch_size: int) -> list[tuple[list[dict], dict]]:
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError
    
    @staticmethod
    def load(path: str) -> 'AbstractReplayBuffer':
        raise NotImplementedError


class UnitSizeReplayBuffer(AbstractReplayBuffer):
    '''
    Local replay buffers for each problem_id
    '''

    def __init__(
        self, 
        unit_capacity: int,
        seed: int,
        buffer: dict[str, list[list[dict]]] = None,
        info_buffer: dict[str, list[dict]] = None,
        idx: dict[str, int] = None
    ):
        self.unit_capacity = unit_capacity
        self.buffer = buffer if buffer is not None else {}
        self.info_buffer = info_buffer if info_buffer is not None else {}
        self.idx = idx if idx is not None else {}
        self.seed = seed
        self.sampler = Random(seed)

    def add_traj(self, traj: list[dict], info: dict = {}) -> None:
        if 'problem_id' not in info:
            raise ValueError('info must contain problem_id')
        problem_id = info['problem_id']
        if problem_id not in self.buffer:
            self.buffer[problem_id] = []
            self.info_buffer[problem_id] = []
            self.idx[problem_id] = 0
        if len(self.buffer[problem_id]) < self.unit_capacity:
            self.buffer[problem_id].append(traj)
            self.info_buffer[problem_id].append(info)
        else:
            self.buffer[problem_id][self.idx[problem_id]] = traj
            self.info_buffer[problem_id][self.idx[problem_id]] = info
            self.idx[problem_id] = (self.idx[problem_id] + 1) % self.unit_capacity
    
    def sample(self, batch_size: int) -> list[tuple[list[dict], dict]]:
        # we first find the list of problem_ids that are not empty
        non_empty_problem_ids = [problem_id for problem_id in self.buffer if len(self.buffer[problem_id]) > 0]
        # we then sample problem_ids with replacement
        sampled_problem_ids = self.sampler.choices(non_empty_problem_ids, k=batch_size)
        # we then sample 1 trajectory from each sampled problem_id
        return [(self.sampler.choice(list(zip(self.buffer[problem_id], self.info_buffer[problem_id])))) for problem_id in sampled_problem_ids]
    
    def save(self, path: str) -> None:
        serialized_obj = dict({
            'unit_capacity': self.unit_capacity,
            'seed': self.seed,
            'buffer': self.buffer,
            'info_buffer': self.info_buffer,
            'idx': self.idx
        })
        with open(path, 'w') as f:
            json.dump(serialized_obj, f)
    
    @staticmethod
    def load(path: str) -> 'UnitSizeReplayBuffer':
        with open(path, 'r') as f:
            serialized_obj = json.load(f)
        buffer = serialized_obj['buffer']
        info_buffer = serialized_obj['info_buffer']
        unit_capacity = serialized_obj['unit_capacity']
        seed = serialized_obj['seed']
        idx = serialized_obj['idx']
        return UnitSizeReplayBuffer(unit_capacity, seed, buffer, info_buffer, idx)