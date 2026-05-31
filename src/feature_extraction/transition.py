import math
from collections import defaultdict
from typing import Dict, List


class TransitionModel:
    def __init__(self, rare_threshold: float = 0.01, laplace_smoothing: float = 1.0):
        self.rare_threshold = rare_threshold
        self.laplace_smoothing = laplace_smoothing
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.tool_counts = defaultdict(int)
        self.vocab = set()
        self.max_anomaly = 1.0

    def fit(self, benign_traces: List[List[str]]) -> None:
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.tool_counts = defaultdict(int)
        self.vocab = set()
        for trace in benign_traces:
            self.vocab.update(trace)
            if len(trace) < 2:
                continue
            for i in range(len(trace) - 1):
                t_i = trace[i]
                t_j = trace[i + 1]
                self.transition_counts[t_i][t_j] += 1
                self.tool_counts[t_i] += 1

        observed = []
        for source, targets in self.transition_counts.items():
            for target in targets:
                observed.append(self.calculate_transition_anomaly(source, target))
        self.max_anomaly = max(observed) if observed else 1.0

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), 1)

    def get_probability(self, t_i: str, t_j: str) -> float:
        count_ij = self.transition_counts.get(t_i, {}).get(t_j, 0)
        count_i = self.tool_counts.get(t_i, 0)
        numerator = count_ij + self.laplace_smoothing
        denominator = count_i + self.vocab_size * self.laplace_smoothing
        return numerator / denominator

    def is_rare_transition(self, t_i: str, t_j: str) -> bool:
        return self.get_probability(t_i, t_j) < self.rare_threshold

    def calculate_transition_anomaly(self, t_i: str, t_j: str) -> float:
        prob = max(self.get_probability(t_i, t_j), 1e-12)
        return -math.log(prob)

    def score_sequence(self, tool_sequence: List[str], normalize: bool = True) -> List[float]:
        if len(tool_sequence) < 2:
            return []
        scores = []
        for i in range(len(tool_sequence) - 1):
            score = self.calculate_transition_anomaly(tool_sequence[i], tool_sequence[i + 1])
            if normalize:
                score = min(score / max(self.max_anomaly, 1e-6), 1.0)
            scores.append(score)
        return scores

    def to_state(self) -> Dict:
        return {
            "rare_threshold": self.rare_threshold,
            "laplace_smoothing": self.laplace_smoothing,
            "transition_counts": {k: dict(v) for k, v in self.transition_counts.items()},
            "tool_counts": dict(self.tool_counts),
            "vocab": sorted(self.vocab),
            "max_anomaly": self.max_anomaly,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "TransitionModel":
        model = cls(
            rare_threshold=state.get("rare_threshold", 0.01),
            laplace_smoothing=state.get("laplace_smoothing", 1.0),
        )
        model.transition_counts = defaultdict(lambda: defaultdict(int))
        for source, targets in state.get("transition_counts", {}).items():
            model.transition_counts[source].update(targets)
        model.tool_counts = defaultdict(int, state.get("tool_counts", {}))
        model.vocab = set(state.get("vocab", []))
        model.max_anomaly = state.get("max_anomaly", 1.0)
        return model
