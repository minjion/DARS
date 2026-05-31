import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev


TOTAL_BENIGN = 3500
TOTAL_MALICIOUS = 1800
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
FALSE_POSITIVE_BENIGN_RATIO = 0.40
REQUIRED_SEED_COUNTS = {0: 24, 1: 24}
MALICIOUS_SEEDS_PER_SCENARIO = 4
REQUIRED_DOMAINS = {"workspace", "slack", "data_manipulation", "external_api"}
REQUIRED_MALICIOUS_SCENARIOS = {
    "S1_prompt_injection",
    "S2_privilege_escalation",
    "S3_malicious_api_usage",
    "S4_data_exfiltration",
    "S5_denial_of_wallet",
    "S6_stealth_mimicry",
}
SYNTHETIC_SEED_SOURCES = {"agentdojo_seed_synthetic", "template", "synthetic_template"}

RISKY_TOOLS = {
    "send_email",
    "http_post",
    "call_api",
    "execute_script",
    "change_permissions",
    "root_shell",
    "delete_user",
    "transfer_funds",
}

SCENARIO_SEVERITY = {
    "benign": 0.0,
    "benign_suspicious": 0.25,
    "S1_prompt_injection": 0.70,
    "S2_privilege_escalation": 0.85,
    "S3_malicious_api_usage": 0.75,
    "S4_data_exfiltration": 0.90,
    "S5_denial_of_wallet": 0.65,
    "S6_stealth_mimicry": 0.80,
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Seed file not found: {path}. Run AgentDojo/Groq collection first and "
            "write exactly 48 balanced seed traces to data/real_seed_traces.jsonl."
        )
    traces = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def write_jsonl(path: Path, traces: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def normalize_domain(domain: str) -> str:
    value = (domain or "unknown").strip().lower().replace("-", "_")
    aliases = {
        "slack_suite": "slack",
        "data": "data_manipulation",
        "tools": "data_manipulation",
        "tool": "data_manipulation",
        "api": "external_api",
        "external": "external_api",
        "travel": "data_manipulation",
        "banking": "external_api",
    }
    return aliases.get(value, value)


def validate_seed_traces(seeds: list[dict], allow_synthetic_seeds: bool = False) -> None:
    if len(seeds) != sum(REQUIRED_SEED_COUNTS.values()):
        raise ValueError(f"Expected 48 seed traces, found {len(seeds)}.")

    label_counts = {0: 0, 1: 0}
    domains = set()
    for index, trace in enumerate(seeds):
        label = int(trace.get("label", -1))
        if label not in label_counts:
            raise ValueError(f"Seed {index} has invalid label: {trace.get('label')}.")
        label_counts[label] += 1

        calls = trace.get("tool_calls", [])
        if not calls:
            raise ValueError(f"Seed {index} has no tool_calls; it cannot be used as an AgentDojo trace.")

        source = str(trace.get("source", "")).lower()
        if source in SYNTHETIC_SEED_SOURCES and not allow_synthetic_seeds:
            raise ValueError(
                f"Seed {index} has synthetic source '{source}'. "
                "Paper-mode generation requires genuine AgentDojo/Groq seed traces."
            )

        domains.add(normalize_domain(trace.get("domain", "")))

    if label_counts != REQUIRED_SEED_COUNTS:
        raise ValueError(f"Expected seed labels {REQUIRED_SEED_COUNTS}, found {label_counts}.")

    missing_domains = REQUIRED_DOMAINS - domains
    if missing_domains:
        raise ValueError(f"Seed file is missing required domains: {sorted(missing_domains)}.")

    malicious_scenarios = {
        str(trace.get("scenario", ""))
        for trace in seeds
        if int(trace.get("label", 0)) == 1
    }
    missing_scenarios = REQUIRED_MALICIOUS_SCENARIOS - malicious_scenarios
    if missing_scenarios and not allow_synthetic_seeds:
        raise ValueError(
            "Seed file is missing required malicious scenarios: "
            f"{sorted(missing_scenarios)}."
        )
    scenario_counts = {
        scenario: sum(
            1
            for trace in seeds
            if int(trace.get("label", 0)) == 1 and trace.get("scenario") == scenario
        )
        for scenario in REQUIRED_MALICIOUS_SCENARIOS
    }
    too_few = {
        scenario: count
        for scenario, count in scenario_counts.items()
        if count < MALICIOUS_SEEDS_PER_SCENARIO
    }
    if too_few and not allow_synthetic_seeds:
        raise ValueError(
            "Seed file must contain at least "
            f"{MALICIOUS_SEEDS_PER_SCENARIO} malicious traces per scenario; found {too_few}."
        )


def split_counts(total: int) -> dict[str, int]:
    train = int(total * SPLITS["train"])
    val = int(total * SPLITS["val"])
    return {"train": train, "val": val, "test": total - train - val}


def build_token_profiles(seeds: list[dict]) -> dict[str, tuple[float, float]]:
    values_by_tool: dict[str, list[float]] = {}
    for trace in seeds:
        for call in trace.get("tool_calls", []):
            tool = call.get("tool", "unknown")
            tokens = float(call.get("tokens", call.get("token_count", 0)) or 0)
            if tokens > 0:
                values_by_tool.setdefault(tool, []).append(tokens)

    profiles = {}
    for tool, values in values_by_tool.items():
        profiles[tool] = (mean(values), max(pstdev(values), 10.0))
    return profiles


def token_count(tool: str, profiles: dict[str, tuple[float, float]], multiplier: float = 1.0) -> int:
    base_mean, base_std = profiles.get(tool, (120.0, 40.0))
    return max(5, int(random.gauss(base_mean * multiplier, base_std)))


def copy_call(call: dict, index: int, profiles: dict[str, tuple[float, float]], multiplier: float = 1.0) -> dict:
    new_call = deepcopy(call)
    tool = new_call.get("tool", "unknown")
    new_call["timestamp"] = index
    new_call["tokens"] = token_count(tool, profiles, multiplier)
    new_call.setdefault("success", random.random() > 0.03)
    new_call.setdefault("input", {})
    if isinstance(new_call["input"], dict):
        new_call["input"] = deepcopy(new_call["input"])
        new_call["input"]["synthetic_request_id"] = f"syn_{random.randint(1000, 9999)}"
    return new_call


def tools_from(seeds: list[dict], label: int | None = None) -> list[str]:
    tools = []
    for trace in seeds:
        if label is not None and int(trace.get("label", 0)) != label:
            continue
        tools.extend(call.get("tool", "unknown") for call in trace.get("tool_calls", []))
    return sorted(set(tools))


def malicious_calls(seeds: list[dict]) -> list[dict]:
    calls = []
    for trace in seeds:
        if int(trace.get("label", 0)) == 1:
            calls.extend(trace.get("tool_calls", []))
    return calls


def benign_calls(seeds: list[dict]) -> list[dict]:
    calls = []
    for trace in seeds:
        if int(trace.get("label", 0)) == 0:
            calls.extend(trace.get("tool_calls", []))
    return calls


def choose_risky_call(seeds: list[dict]) -> dict:
    risky = [
        call for call in malicious_calls(seeds)
        if call.get("tool", "unknown") in RISKY_TOOLS
    ]
    pool = risky or malicious_calls(seeds)
    return deepcopy(random.choice(pool))


def choose_benign_anomaly_call(seeds: list[dict]) -> dict:
    benign_pool = benign_calls(seeds)
    suspicious_but_legitimate = [
        call for call in benign_pool
        if call.get("tool", "unknown") not in RISKY_TOOLS
    ]
    pool = suspicious_but_legitimate or benign_pool
    return deepcopy(random.choice(pool))


def seed_call_for_tool(seeds: list[dict], tool_name: str) -> dict:
    for trace in seeds:
        for call in trace.get("tool_calls", []):
            if call.get("tool") == tool_name:
                return deepcopy(call)
    return {"tool": tool_name, "input": {}, "tokens": 120, "success": True}


def append_tool_chain(
    calls: list[dict],
    tool_names: list[str],
    seeds: list[dict],
    profiles: dict[str, tuple[float, float]],
    multiplier: float = 1.0,
) -> None:
    for tool_name in tool_names:
        calls.append(copy_call(seed_call_for_tool(seeds, tool_name), len(calls), profiles, multiplier))


def normalize_seed(seed_trace: dict) -> dict:
    trace = deepcopy(seed_trace)
    trace["domain"] = normalize_domain(trace.get("domain", "unknown"))
    trace["label"] = int(trace.get("label", 0))
    trace.setdefault("scenario", "benign" if trace["label"] == 0 else "S1_prompt_injection")
    return trace


def synthesize_from_seed(
    seed_trace: dict,
    index: int,
    all_seeds: list[dict],
    token_profiles: dict[str, tuple[float, float]],
    benign_anomaly: bool = False,
) -> dict:
    seed_trace = normalize_seed(seed_trace)
    label = int(seed_trace["label"])
    calls = [
        copy_call(call, i, token_profiles)
        for i, call in enumerate(seed_trace.get("tool_calls", []))
    ]

    if label == 0:
        if random.random() < 0.35 and calls:
            insert_at = random.randint(0, len(calls) - 1)
            calls.insert(insert_at, copy_call(calls[insert_at], insert_at, token_profiles))
        if random.random() < 0.25 and calls:
            calls.append(copy_call(random.choice(calls), len(calls), token_profiles))
        if benign_anomaly:
            risky = choose_benign_anomaly_call(all_seeds)
            insert_at = random.randint(1, max(len(calls), 1))
            calls.insert(insert_at, copy_call(risky, insert_at, token_profiles, multiplier=1.15))
    else:
        scenario = seed_trace.get("scenario", "S1_prompt_injection")
        if scenario == "S1_prompt_injection":
            calls.append(copy_call(choose_risky_call(all_seeds), len(calls), token_profiles))
        elif scenario == "S2_privilege_escalation":
            append_tool_chain(
                calls,
                ["read_file", "write_file", "execute_script", "change_permissions", "root_shell"],
                all_seeds,
                token_profiles,
                multiplier=1.10,
            )
        elif scenario == "S3_malicious_api_usage":
            append_tool_chain(
                calls,
                ["http_get", "call_api", "http_post"],
                all_seeds,
                token_profiles,
                multiplier=1.15,
            )
        elif scenario == "S4_data_exfiltration":
            append_tool_chain(
                calls,
                ["read_database", "read_file", "send_email"],
                all_seeds,
                token_profiles,
                multiplier=1.20,
            )
        elif scenario == "S5_denial_of_wallet":
            repeat_call = random.choice(calls)
            for _ in range(random.randint(2, 5)):
                calls.append(copy_call(repeat_call, len(calls), token_profiles, multiplier=2.8))
        elif scenario == "S6_stealth_mimicry":
            benign_seeds = [trace for trace in all_seeds if int(trace.get("label", 0)) == 0]
            mimic_base = random.choice(benign_seeds)
            calls = [
                copy_call(call, i, token_profiles)
                for i, call in enumerate(mimic_base.get("tool_calls", []))
            ]
            if calls:
                replace_at = random.randint(1, len(calls) - 1) if len(calls) > 1 else 0
                calls[replace_at] = copy_call(choose_risky_call(all_seeds), replace_at, token_profiles)
        else:
            calls.append(copy_call(choose_risky_call(all_seeds), len(calls), token_profiles))

    for i, call in enumerate(calls):
        call["timestamp"] = i

    split_label = "benign" if label == 0 else "malicious"
    trace = {
        "agent_id": f"agent-{split_label}-{index:04d}",
        "domain": seed_trace["domain"],
        "scenario": "benign_suspicious" if benign_anomaly else seed_trace.get("scenario", "benign"),
        "source": "agentdojo_seed_synthesized",
        "seed_agent_id": seed_trace.get("agent_id", f"seed-{index}"),
        "tool_calls": calls[:20],
        "label": label,
        "benign_anomaly": benign_anomaly,
        "severity": SCENARIO_SEVERITY.get(
            "benign_suspicious" if benign_anomaly else seed_trace.get("scenario", "benign"),
            0.75 if label else 0.0,
        ),
    }
    return trace


def select_seed(seeds: list[dict], label: int, scenario: str | None = None) -> dict:
    pool = [trace for trace in seeds if int(trace.get("label", 0)) == label]
    if scenario is not None:
        scenario_pool = [trace for trace in pool if trace.get("scenario") == scenario]
        if scenario_pool:
            pool = scenario_pool
    return random.choice(pool)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the paper dataset from 48 balanced genuine AgentDojo/Groq seed traces."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--seed-file", type=Path, default=Path(__file__).parent / "real_seed_traces.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--allow-synthetic-seeds",
        action="store_true",
        help="Development-only escape hatch. Do not use for paper results.",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    seeds = [normalize_seed(trace) for trace in load_jsonl(args.seed_file)]
    validate_seed_traces(seeds, allow_synthetic_seeds=args.allow_synthetic_seeds)
    token_profiles = build_token_profiles(seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benign_counts = split_counts(TOTAL_BENIGN)
    malicious_counts = split_counts(TOTAL_MALICIOUS)
    benign_index = 0
    malicious_index = 0

    for split_name in ["train", "val", "test"]:
        traces = []
        benign_total = benign_counts[split_name]
        malicious_total = malicious_counts[split_name]
        anomalous_total = int(benign_total * FALSE_POSITIVE_BENIGN_RATIO)
        anomaly_flags = [True] * anomalous_total + [False] * (benign_total - anomalous_total)
        random.shuffle(anomaly_flags)

        for anomalous in anomaly_flags:
            traces.append(
                synthesize_from_seed(
                    select_seed(seeds, 0),
                    benign_index,
                    seeds,
                    token_profiles,
                    benign_anomaly=anomalous,
                )
            )
            benign_index += 1

        scenario_cycle = sorted(REQUIRED_MALICIOUS_SCENARIOS)
        for _ in range(malicious_total):
            scenario = scenario_cycle[malicious_index % len(scenario_cycle)]
            traces.append(
                synthesize_from_seed(
                    select_seed(seeds, 1, scenario=scenario),
                    malicious_index,
                    seeds,
                    token_profiles,
                )
            )
            malicious_index += 1

        random.shuffle(traces)
        write_jsonl(args.output_dir / f"{split_name}_traces.jsonl", traces)
        print(
            f"{split_name}: {len(traces)} traces "
            f"({benign_total} benign, {malicious_total} malicious)"
        )

    write_jsonl(args.output_dir / "seed_traces.jsonl", seeds)
    print(f"Copied validated AgentDojo/Groq seeds from {args.seed_file} to data/seed_traces.jsonl.")


if __name__ == "__main__":
    main()
