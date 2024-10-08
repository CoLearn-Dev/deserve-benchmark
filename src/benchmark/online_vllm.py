import argparse
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

import requests
from openai import OpenAI, Stream
from openai.types.completion import Completion
from transformers import AutoTokenizer  # type: ignore

from src.workload.utils import Workload

from ..rater import Rater, RaterTimeLimitExceeded, Response
from ..workload.oasst1 import Oasst1Dataset
from ..workload.sharegpt import ShareGptDataset


class OnlineVLLMClient:
    def __init__(
        self,
        model: str,
        workload: Workload,
        time_limit: int,
        url: str,
        batch_size: int,
        max_tokens: int,
    ):
        self.url = url
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.network_executor = ThreadPoolExecutor(max_workers=128)
        self.vllm_executor = ThreadPoolExecutor(max_workers=128)
        self.rater = Rater(workload=workload, time_limit=time_limit)
        self.model = model
        self.openai_client = OpenAI(
            api_key="EMPTY",
            base_url=self.url,
        )

    def polling(self) -> None:
        while True:
            completions = self.rater.get(1)
            if len(completions) == 0:
                # no more requests
                break
            id = completions[0].id
            history = completions[0].history
            try:
                chat_stream: Stream[Completion] = self.openai_client.completions.create(
                    model=self.model,
                    prompt=history,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
            except Exception as e:
                print(e)
                raise e
            for chunk in chat_stream:
                content = chunk.choices[0].text
                if content is None:
                    continue
                try:
                    self.rater.post(Response(id=id, payload=content, finished=False))
                except RaterTimeLimitExceeded as e:
                    return
            self.rater.post(Response(id=id, payload="", finished=True))

    def speedtest(self) -> dict[str, Any]:
        futures = []
        for _ in range(self.batch_size):
            futures.append(self.vllm_executor.submit(self.polling))
        wait(futures, return_when=ALL_COMPLETED)
        return self.rater.dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--workload", type=str, default="oasst1")
    args = parser.parse_args()

    if args.workload == "oasst1":
        workload = Oasst1Dataset().into_workload()
    elif args.workload == "sharegpt":
        workload = ShareGptDataset().into_workload()
    elif args.workload.startswith("fixed-"):
        length = int(args.workload[len("fixed-") :])
    else:
        raise ValueError(f"Unknown workload: {args.workload}")
    client = OnlineVLLMClient(
        model=args.model_name,
        workload=workload,
        time_limit=args.time_limit,
        url=args.url,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
    result = client.speedtest()
    print(result)
