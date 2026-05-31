import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from src.feature_extraction.extractor import DARSFeatureExtractor


def load_traces(jsonl_path: str | Path) -> List[dict]:
    traces = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


class AgentDojoDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        extractor: DARSFeatureExtractor | None = None,
        max_seq_len: int = 20,
        fit_extractor: bool = False,
    ):
        self.traces_raw = load_traces(jsonl_path)
        self.labels = [float(trace.get("label", 0)) for trace in self.traces_raw]
        self.scenarios = [trace.get("scenario", "unknown") for trace in self.traces_raw]
        self.domains = [trace.get("domain", "unknown") for trace in self.traces_raw]
        self.extractor = extractor or DARSFeatureExtractor(max_seq_len=max_seq_len)

        if fit_extractor or extractor is None:
            self.extractor.fit(self.traces_raw)

        self.traces = [
            self.extractor.extract_trace_features(trace)
            for trace in self.traces_raw
        ]

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return self.traces[idx], label
