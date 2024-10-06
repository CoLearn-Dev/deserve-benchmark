import argparse
from typing import Any

from transformers import AutoTokenizer  # type: ignore
from vllm import LLM, SamplingParams

from ..rater import Rater, RaterTimeLimitExceeded, Response
from ..workload.oasst1 import Oasst1Dataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


class OfflineVLLMClient:
    def __init__(
        self,
        model: str,
        workload: list[list[dict[str, str]]],
        time_limit: float,
        url: str,
        batch_size: int,
        max_tokens: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ):
        self.url = url
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.rater = Rater(workload=workload, time_limit=time_limit)
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

    def speedtest(self) -> dict[str, Any]:
        while True:
            requests = self.rater.get(self.batch_size)
            if len(requests) == 0:
                # no more requests
                break
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    request.history, tokenize=False, add_generation_timestamp=True
                )
                for request in requests
            ]
            results = self.llm.generate(
                formatted_prompts, SamplingParams(max_tokens=self.max_tokens)
            )
            results = [output.outputs[0].text for output in results]
            try:
                for history, result in zip(requests, results):
                    self.rater.post(
                        Response(id=history.id, payload=result, finished=True)
                    )
            except RaterTimeLimitExceeded:
                break
        return self.rater.dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    workload = Oasst1Dataset().into_workload()
    client = OfflineVLLMClient(
        model=args.model_name,
        workload=workload,
        time_limit=args.time_limit,
        url="http://localhost:8000/v1",
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )
    result = client.speedtest()
    print(result)
