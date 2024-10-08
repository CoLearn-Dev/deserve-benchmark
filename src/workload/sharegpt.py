import json

from .utils import cache


class ShareGptDataset:
    def __init__(self) -> None:
        with open(
            "datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
            "r",
        ) as f:
            self.raw_data = json.load(f)

    @cache()
    def into_workload(self) -> list[list[dict[str, str]]]:
        workload = []
        for conversation in self.raw_data:
            messages = []
            for message in conversation["conversations"]:
                role = "user" if message["from"] == "human" else "assistant"
                messages.append({"content": message["value"], "role": role})
                if role == "user":
                    workload.append(messages.copy())
        return workload


if __name__ == "__main__":
    dataset = ShareGptDataset()
    workload = dataset.into_workload()
