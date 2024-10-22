import random

from src.workload.utils import Workload


class StaticWorkload(Workload):
    def __init__(self, length: int) -> None:
        self.length = length
        self.variance = length // 8

    def get(self, offset: int, length: int) -> list[str]:
        result_length = self.length + random.randint(-self.variance, self.variance)
        return ["a" * 8 * result_length] * length
