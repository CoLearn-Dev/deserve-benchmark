from src.workload.utils import Workload


class StaticWorkload(Workload):
    def __init__(self, length: int) -> None:
        self.length = length

    def get(self, offset: int, length: int) -> list[str]:
        return ["a" * self.length] * length
