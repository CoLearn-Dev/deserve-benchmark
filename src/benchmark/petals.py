import argparse
import json
import random
import threading
import time
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from petals import AutoDistributedModelForCausalLM  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from vllm import SamplingParams

from src.workload.sharegpt import ShareGptDataset
from src.workload.static import StaticWorkload
from src.workload.utils import Workload

from ..rater import Rater, RaterTimeLimitExceeded, Response
from ..workload.oasst1 import Oasst1Dataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


class PetalsClient:
    def __init__(
        self,
        model: str,
        initial_peers: list[str],
        workload: Workload,
        time_limit: int,
        url: str,
        batch_size: int,
        max_tokens: int,
        trace: bool,
        warmup: int,
    ):
        self.url = url
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.rater = Rater(
            workload=workload, time_limit=time_limit, trace=trace, warmup=warmup
        )
        self.time_limit = time_limit
        self.petals_executor = ThreadPoolExecutor(max_workers=128)
        self.llm = AutoDistributedModelForCausalLM.from_pretrained(
            model, initial_peers=initial_peers
        )
        self.variance = max_tokens // 8

    def polling(self) -> None:
        while True:
            completions = self.rater.get(1)
            if len(completions) == 0:
                # no more requests
                break
            id = completions[0].id
            history = completions[0].history
            try:
                inputs = tokenizer(history, return_tensors="pt")
                output = self.llm.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens
                    + random.randint(-self.variance, self.variance),
                )
            except Exception as e:
                print(e)
                raise e
            try:
                self.rater.post(Response(id=id, payload=output, finished=True))
            except RaterTimeLimitExceeded as e:
                return

    def routine(self) -> None:
        try:
            futures = []
            for _ in range(self.batch_size):
                futures.append(self.petals_executor.submit(self.polling))
            wait(futures, return_when=ALL_COMPLETED)
        except KeyboardInterrupt:
            pass

    def speedtest(self) -> dict[str, Any]:
        routine_thread = threading.Thread(target=self.routine, daemon=True)
        routine_thread.start()
        try:
            for _ in range(self.time_limit):
                time.sleep(1)
                if not routine_thread.is_alive():
                    break
        except KeyboardInterrupt:
            pass
        return self.rater.dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--time-limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--workload", type=str, default="oasst1")
    parser.add_argument(
        "--initial-peer",
        type=str,
        default="/ip4/10.138.0.16/tcp/31337/p2p/QmWKuxaaz4twxP7JMttFZM28BkDctk7CML3ZV6C4iJzt5X",
    )
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()

    if args.workload == "oasst1":
        workload = Oasst1Dataset().into_workload()
    elif args.workload == "sharegpt":
        workload = ShareGptDataset().into_workload()
    elif args.workload.startswith("fixed"):
        raw = args.workload[len("fixed") :]
        size, length, variance = map(int, raw.split(":"))
        workload = StaticWorkload(size, length, variance)
    else:
        raise ValueError(f"Unknown workload: {args.workload}")

    client = PetalsClient(
        model=args.model_name,
        workload=workload,
        initial_peers=[args.initial_peer],
        time_limit=args.time_limit,
        url="http://localhost:8000/v1",
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        trace=args.trace,
        warmup=args.warmup,
    )
    result = client.speedtest()
    print(json.dumps(result))
