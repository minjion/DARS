from statistics import mean, pstdev
from typing import Dict, List


class TokenAnalyzer:
    def __init__(self, burst_z_threshold: float = 3.0, wallet_threshold: int = 2500):
        self.burst_z_threshold = burst_z_threshold
        self.wallet_threshold = wallet_threshold
        self.mean_tokens = 100.0
        self.std_tokens = 1.0

    def fit(self, benign_traces: List[dict]) -> None:
        token_counts = []
        for trace in benign_traces:
            for call in trace.get("tool_calls", []):
                token_counts.append(float(call.get("tokens", call.get("token_count", 0)) or 0))
        if token_counts:
            self.mean_tokens = mean(token_counts)
            self.std_tokens = max(pstdev(token_counts), 1.0)

    def score_call(self, token_count: float, cumulative_tokens: float = 0.0) -> float:
        z_score = max(0.0, (token_count - self.mean_tokens) / self.std_tokens)
        burst_score = min(z_score / self.burst_z_threshold, 1.0)
        wallet_score = min(cumulative_tokens / self.wallet_threshold, 1.0)
        return max(burst_score, wallet_score)

    def to_state(self) -> Dict:
        return {
            "burst_z_threshold": self.burst_z_threshold,
            "wallet_threshold": self.wallet_threshold,
            "mean_tokens": self.mean_tokens,
            "std_tokens": self.std_tokens,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "TokenAnalyzer":
        analyzer = cls(
            burst_z_threshold=state.get("burst_z_threshold", 3.0),
            wallet_threshold=state.get("wallet_threshold", 2500),
        )
        analyzer.mean_tokens = state.get("mean_tokens", 100.0)
        analyzer.std_tokens = state.get("std_tokens", 1.0)
        return analyzer
