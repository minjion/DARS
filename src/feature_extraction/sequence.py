import math
from collections import defaultdict
from typing import Dict, List, Tuple


class SequenceContextModel:
    def __init__(self, ngram_order: int = 3, laplace_smoothing: float = 1.0):
        self.ngram_order = max(2, ngram_order)
        self.laplace_smoothing = laplace_smoothing
        self.context_counts = defaultdict(lambda: defaultdict(int))
        self.context_totals = defaultdict(int)
        self.vocab = set()
        self.max_anomaly = 1.0

    @property
    def context_size(self) -> int:
        return self.ngram_order - 1

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), 1)

    def _padded(self, sequence: List[str]) -> List[str]:
        return ["<s>"] * self.context_size + list(sequence)

    def fit(self, benign_traces: List[List[str]]) -> None:
        self.context_counts = defaultdict(lambda: defaultdict(int))
        self.context_totals = defaultdict(int)
        self.vocab = set()

        for sequence in benign_traces:
            self.vocab.update(sequence)
            padded = self._padded(sequence)
            for index in range(self.context_size, len(padded)):
                context = tuple(padded[index - self.context_size : index])
                token = padded[index]
                self.context_counts[context][token] += 1
                self.context_totals[context] += 1

        observed = []
        for sequence in benign_traces:
            observed.extend(self.score_sequence(sequence, normalize=False))
        self.max_anomaly = max(observed) if observed else 1.0

    def get_probability(self, context: Tuple[str, ...], token: str) -> float:
        count = self.context_counts.get(context, {}).get(token, 0)
        total = self.context_totals.get(context, 0)
        numerator = count + self.laplace_smoothing
        denominator = total + self.vocab_size * self.laplace_smoothing
        return numerator / denominator

    def score_sequence(self, tool_sequence: List[str], normalize: bool = True) -> List[float]:
        if not tool_sequence:
            return []
        padded = self._padded(tool_sequence)
        scores = []
        for index in range(self.context_size, len(padded)):
            context = tuple(padded[index - self.context_size : index])
            token = padded[index]
            probability = max(self.get_probability(context, token), 1e-12)
            score = -math.log(probability)
            if normalize:
                score = min(score / max(self.max_anomaly, 1e-6), 1.0)
            scores.append(score)
        return scores

    def to_state(self) -> Dict:
        return {
            "ngram_order": self.ngram_order,
            "laplace_smoothing": self.laplace_smoothing,
            "context_counts": {
                "\x1f".join(context): dict(targets)
                for context, targets in self.context_counts.items()
            },
            "context_totals": {
                "\x1f".join(context): count
                for context, count in self.context_totals.items()
            },
            "vocab": sorted(self.vocab),
            "max_anomaly": self.max_anomaly,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "SequenceContextModel":
        model = cls(
            ngram_order=state.get("ngram_order", 3),
            laplace_smoothing=state.get("laplace_smoothing", 1.0),
        )
        model.context_counts = defaultdict(lambda: defaultdict(int))
        for context_key, targets in state.get("context_counts", {}).items():
            model.context_counts[tuple(context_key.split("\x1f"))].update(targets)
        model.context_totals = defaultdict(int)
        for context_key, count in state.get("context_totals", {}).items():
            model.context_totals[tuple(context_key.split("\x1f"))] = count
        model.vocab = set(state.get("vocab", []))
        model.max_anomaly = state.get("max_anomaly", 1.0)
        return model
