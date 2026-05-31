from statistics import mean, pstdev
from typing import Dict, List

import torch

from src.feature_extraction.bdi_calculator import BDICalculator
from src.feature_extraction.privilege import PrivilegeTracker
from src.feature_extraction.sequence import SequenceContextModel
from src.feature_extraction.token import TokenAnalyzer
from src.feature_extraction.transition import TransitionModel


FEATURE_NAMES = [
    "bdi_deviation",
    "privilege_level",
    "privilege_escalation",
    "transition_anomaly",
    "sequence_anomaly",
    "token_burst",
]


class DARSFeatureExtractor:
    def __init__(self, max_seq_len: int = 20):
        self.max_seq_len = max_seq_len
        self.bdi_calc = BDICalculator()
        self.privilege_tracker = PrivilegeTracker()
        self.transition_model = TransitionModel()
        self.sequence_model = SequenceContextModel()
        self.token_analyzer = TokenAnalyzer()
        self.bdi_mean = 0.0
        self.bdi_std = 1.0
        self.fitted = False

    def fit(self, traces: List[dict]) -> None:
        benign_traces = [trace for trace in traces if int(trace.get("label", 0)) == 0]
        benign_sequences = [self.extract_tools(trace) for trace in benign_traces]
        self.transition_model.fit(benign_sequences)
        self.sequence_model.fit(benign_sequences)
        self.token_analyzer.fit(benign_traces)

        bdi_values = []
        for sequence in benign_sequences:
            for i in range(len(sequence)):
                value = self.bdi_calc.calculate_bdi(sequence[: i + 1])
                bdi_values.append(value)
        if bdi_values:
            self.bdi_mean = mean(bdi_values)
            self.bdi_std = max(pstdev(bdi_values), 1e-6)
        self.fitted = True

    @staticmethod
    def extract_tools(trace: dict) -> List[str]:
        return [call.get("tool", "unknown") for call in trace.get("tool_calls", [])]

    def extract_trace_features(self, trace: dict) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("DARSFeatureExtractor must be fitted before extracting features.")

        calls = trace.get("tool_calls", [])
        sequence = self.extract_tools(trace)
        escalation_scores = self.privilege_tracker.calculate_dynamic_escalation(sequence)
        transition_scores = self.transition_model.score_sequence(sequence)
        sequence_scores = self.sequence_model.score_sequence(sequence)

        rows = []
        cumulative_tokens = 0.0
        for i, tool in enumerate(sequence[: self.max_seq_len]):
            prefix = sequence[: i + 1]
            bdi = self.bdi_calc.calculate_bdi(prefix)
            bdi_deviation = min(abs(bdi - self.bdi_mean) / (3.0 * self.bdi_std), 1.0)
            privilege_level = self.privilege_tracker.get_level(tool) / 5.0
            escalation = escalation_scores[i] if i < len(escalation_scores) else 0.0
            transition = transition_scores[i - 1] if i > 0 and i - 1 < len(transition_scores) else 0.0
            sequence_anomaly = sequence_scores[i] if i < len(sequence_scores) else 0.0
            token_count = float(calls[i].get("tokens", calls[i].get("token_count", 0)) or 0)
            cumulative_tokens += token_count
            token_burst = self.token_analyzer.score_call(token_count, cumulative_tokens)
            rows.append([bdi_deviation, privilege_level, escalation, transition, sequence_anomaly, token_burst])

        while len(rows) < self.max_seq_len:
            rows.append([0.0] * len(FEATURE_NAMES))

        return torch.tensor(rows, dtype=torch.float32)

    def summarize_trace(self, trace: dict) -> Dict[str, float]:
        features = self.extract_trace_features(trace)
        sequence_length = min(len(trace.get("tool_calls", [])), self.max_seq_len)
        if sequence_length == 0:
            return {name: 0.0 for name in FEATURE_NAMES}
        active = features[:sequence_length]
        return {
            "bdi_deviation": float(active[:, 0].max().item()),
            "privilege_level": float(active[:, 1].max().item()),
            "privilege_escalation": float(active[:, 2].max().item()),
            "transition_anomaly": float(active[:, 3].max().item()),
            "sequence_anomaly": float(active[:, 4].max().item()),
            "token_burst": float(active[:, 5].max().item()),
        }

    def to_state(self) -> Dict:
        return {
            "max_seq_len": self.max_seq_len,
            "bdi_mean": self.bdi_mean,
            "bdi_std": self.bdi_std,
            "transition_model": self.transition_model.to_state(),
            "sequence_model": self.sequence_model.to_state(),
            "token_analyzer": self.token_analyzer.to_state(),
        }

    @classmethod
    def from_state(cls, state: Dict) -> "DARSFeatureExtractor":
        extractor = cls(max_seq_len=state.get("max_seq_len", 20))
        extractor.bdi_mean = state.get("bdi_mean", 0.0)
        extractor.bdi_std = state.get("bdi_std", 1.0)
        extractor.transition_model = TransitionModel.from_state(state.get("transition_model", {}))
        extractor.sequence_model = SequenceContextModel.from_state(state.get("sequence_model", {}))
        extractor.token_analyzer = TokenAnalyzer.from_state(state.get("token_analyzer", {}))
        extractor.fitted = True
        return extractor
