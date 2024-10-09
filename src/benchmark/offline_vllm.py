import argparse
import json
from typing import Any

from transformers import AutoTokenizer  # type: ignore
from vllm import LLM, SamplingParams

from src.workload.sharegpt import ShareGptDataset
from src.workload.utils import Workload

from ..rater import Rater, RaterTimeLimitExceeded, Response
from ..workload.oasst1 import Oasst1Dataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


class OfflineVLLMClient:
    def __init__(
        self,
        model: str,
        workload: Workload,
        time_limit: int,
        url: str,
        batch_size: int,
        max_tokens: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        trace: bool,
    ):
        self.url = url
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.rater = Rater(workload=workload, time_limit=time_limit, trace=trace)
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

    def speedtest(self) -> dict[str, Any]:
        try:
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
                texts = [output.outputs[0].text for output in results]
                try:
                    for history, text in zip(requests, texts):
                        self.rater.post(
                            Response(id=history.id, payload=text, finished=True)
                        )
                except RaterTimeLimitExceeded:
                    break
        except KeyboardInterrupt:
            pass
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
    parser.add_argument("--workload", type=str, default="oasst1")
    parser.add_argument("--trace", action="store_true", default=False)
    args = parser.parse_args()

    if args.workload == "oasst1":
        workload = Oasst1Dataset().into_workload()
    elif args.workload == "sharegpt":
        workload = ShareGptDataset().into_workload()
    else:
        raise ValueError(f"Unknown workload: {args.workload}")

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
        trace=args.trace,
    )
    result = client.speedtest()
    print(json.dumps(result))
