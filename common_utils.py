import requests
import json
import time
import torch
import torch.nn.functional as F


class APIWrapper:

    MAX_RETRIES = 5

    def __init__(self, api_url: str):
        self._api_url = api_url

    def api_post(self, endpoint: str, data: dict, timeout: int | None = None):
        for _ in range(APIWrapper.MAX_RETRIES):
            try:
                response = requests.post(
                    url=f"{self._api_url}/{endpoint}",
                    json=data,
                    timeout=timeout
                )
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)
        response.raise_for_status()
        return response.json()
    
    def api_post_stream(self, endpoint: str, data: dict, timeout: int | None = None):
        for _ in range(APIWrapper.MAX_RETRIES):
            try:
                response = requests.post(
                    url=f"{self._api_url}/{endpoint}",
                    json=data,
                    stream=True,
                    timeout=timeout
                )
                response.raise_for_status()
                for chunk in response.iter_lines():
                    yield json.loads(chunk)
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy