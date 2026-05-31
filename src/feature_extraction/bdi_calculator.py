import math
from collections import Counter
from typing import Dict, List

class BDICalculator:
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate_entropy(self, tool_sequence: List[str]) -> float:
        if not tool_sequence: return 0.0
        freqs = Counter(tool_sequence)
        total = len(tool_sequence)
        unique_tools = len(freqs)
        if unique_tools <= 1: return 0.0
        entropy = -sum((count / total) * math.log2(count / total) for count in freqs.values())
        max_entropy = math.log2(unique_tools)
        return entropy / max_entropy

    def calculate_svi(self, tool_sequence: List[str]) -> float:
        if len(tool_sequence) < 2: return 0.0
        bigrams = list(zip(tool_sequence[:-1], tool_sequence[1:]))
        if not bigrams: return 0.0
        bigram_freqs = Counter(bigrams)
        most_common_count = bigram_freqs.most_common(1)[0][1]
        return 1.0 - (most_common_count / len(bigrams))

    def calculate_ads(self, tool_sequence: List[str]) -> float:
        if not tool_sequence: return 0.0
        freqs = Counter(tool_sequence)
        total = len(tool_sequence)
        max_p = max(count / total for count in freqs.values())
        return 1.0 - max_p

    def calculate_bdi(self, tool_sequence: List[str]) -> float:
        h_norm = self.calculate_entropy(tool_sequence)
        svi = self.calculate_svi(tool_sequence)
        ads = self.calculate_ads(tool_sequence)
        return (self.alpha * h_norm) + (self.beta * svi) + (self.gamma * ads)

    def calculate_components(self, tool_sequence: List[str]) -> Dict[str, float]:
        return {
            "entropy": self.calculate_entropy(tool_sequence),
            "svi": self.calculate_svi(tool_sequence),
            "ads": self.calculate_ads(tool_sequence),
            "bdi": self.calculate_bdi(tool_sequence),
        }
