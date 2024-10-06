import datetime
import threading
from dataclasses import dataclass
from typing import Any, Optional

import tiktoken
from transformers import AutoTokenizer  # type: ignore

llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B")
openai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class RaterRuntimeError(Exception):
    pass


class RaterTimeLimitExceeded(Exception):
    pass


@dataclass
class Request:
    id: int
    history: list[dict[str, str]]


@dataclass
class Response:
    id: int
    payload: str
    finished: bool

    def count_openai_tokens(self) -> int:
        return len(openai_tokenizer.encode(self.payload))


@dataclass
class BufferedString:
    buffer: str = ""
    time_last_added: Optional[datetime.datetime] = None
    finished: bool = False

    def insert(
        self, offset: int, payload: str, time: datetime.datetime, finished: bool
    ) -> None:
        self.time_last_added = time
        self.finished = finished
        if offset == -1:
            self.buffer = self.buffer + payload
        elif offset <= len(self.buffer):
            self.buffer = self.buffer[:offset] + payload
        else:
            self.buffer = self.buffer + " " * (offset - len(self.buffer)) + payload


@dataclass
class BufferedResponses:
    content: list[BufferedString]

    def __init__(self) -> None:
        self.content = []

    def insert(
        self,
        index: int,
        offset: int,
        payload: str,
        time: datetime.datetime,
        finished: bool,
    ) -> None:
        while index >= len(self.content):
            self.content.append(BufferedString())
        self.content[index].insert(offset, payload, time, finished)

    def into_timestamped_responses(
        self,
    ) -> list[tuple[str, Optional[datetime.datetime]]]:
        return list(map(lambda x: (x.buffer, x.time_last_added), self.content))


class Rater:
    def __init__(self, workload: list[list[dict[str, str]]], time_limit: float):
        self.workload = workload
        self.ptr = 0
        self.time_limit = time_limit
        self.time_first_get: Optional[datetime.datetime] = None
        self.time_last_post: Optional[datetime.datetime] = None
        self.buffered_responses = BufferedResponses()
        self.post_count_history: dict[int, int] = {}
        self.post_finished_history: dict[int, int] = {}
        self.post_count_total = 0
        self.post_finished_total = 0
        self.lock = threading.Lock()

    def get(self, size: int) -> list[Request]:
        if self.time_first_get is None:
            self.time_first_get = datetime.datetime.now()
        if self.ptr >= len(self.workload):
            return []
        last = min(self.ptr + size, len(self.workload))
        ret = self.workload[self.ptr : last]
        self.ptr = last
        return [
            Request(id=id, history=history)
            for id, history in enumerate(ret, start=self.ptr)
        ]

    def post(self, response: Response) -> None:
        now = datetime.datetime.now()
        if self.time_first_get is None:
            raise RaterRuntimeError("First get must be called before post")
        self.time_last_post = now
        if now - self.time_first_get > datetime.timedelta(seconds=self.time_limit):
            raise RaterTimeLimitExceeded("Time limit exceeded")
        self.buffered_responses.insert(
            response.id,
            -1,
            response.payload,
            now,
            response.finished,
        )
        time_delta = int((now - self.time_first_get).total_seconds())

        with self.lock:
            if time_delta not in self.post_count_history:
                self.post_count_history[time_delta] = 0
            self.post_count_history[time_delta] += response.count_openai_tokens()
            self.post_count_total += response.count_openai_tokens()
            if time_delta not in self.post_finished_history:
                self.post_finished_history[time_delta] = 0
            if response.finished:
                self.post_finished_history[time_delta] += 1
                self.post_finished_total += 1

    def dump(self) -> dict[str, Any]:
        return {
            "time_limit": self.time_limit,
            "time_first_get": (
                self.time_first_get.timestamp() if self.time_first_get else None
            ),
            "time_last_post": (
                self.time_last_post.timestamp() if self.time_last_post else None
            ),
            "post_count_total": self.post_count_total,
            "post_finished_total": self.post_finished_total,
            "post_count_history": self.post_count_history,
            "post_finished_history": self.post_finished_history,
        }
