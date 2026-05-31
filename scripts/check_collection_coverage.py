import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
DOMAIN_ALIASES = {
    "slack_suite": "slack",
    "data": "data_manipulation",
    "tools": "data_manipulation",
    "tool": "data_manipulation",
    "api": "external_api",
    "external": "external_api",
    "travel": "data_manipulation",
    "banking": "external_api",
}


def normalize_domain(domain: str) -> str:
    value = (domain or "unknown").strip().lower().replace("-", "_")
    return DOMAIN_ALIASES.get(value, value)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_labels(traces: list[dict]) -> Counter:
    return Counter(int(trace.get("label", 0)) for trace in traces)


def coverage_status(traces: list[dict]) -> tuple[bool, list[str]]:
    labels = count_labels(traces)
    domains = Counter(normalize_domain(str(trace.get("domain", ""))) for trace in traces)
    scenarios = Counter(
        str(trace.get("scenario", "unknown"))
        for trace in traces
        if int(trace.get("label", 0)) == 1
    )

    messages = []
    ok = True

    for label, required in REQUIRED_SEED_COUNTS.items():
        count = labels.get(label, 0)
        if count < required:
            ok = False
            messages.append(f"missing label={label}: need {required}, have {count}")

    missing_domains = REQUIRED_DOMAINS - set(domains)
    if missing_domains:
        ok = False
        messages.append(f"missing domains: {', '.join(sorted(missing_domains))}")

    missing_scenarios = REQUIRED_MALICIOUS_SCENARIOS - set(scenarios)
    if missing_scenarios:
        ok = False
        messages.append(f"missing malicious scenarios: {', '.join(sorted(missing_scenarios))}")
    too_few_scenarios = {
        scenario: scenarios.get(scenario, 0)
        for scenario in REQUIRED_MALICIOUS_SCENARIOS
        if scenarios.get(scenario, 0) < MALICIOUS_SEEDS_PER_SCENARIO
    }
    if too_few_scenarios:
        ok = False
        messages.append(
            "need at least "
            f"{MALICIOUS_SEEDS_PER_SCENARIO} traces per malicious scenario: {too_few_scenarios}"
        )

    return ok, messages


def print_counter(title: str, counter: Counter, required: set[str] | None = None) -> None:
    print(f"\n{title}")
    if required:
        keys = sorted(required | set(counter))
    else:
        keys = sorted(counter)
    if not keys:
        print("  none")
        return
    for key in keys:
        marker = "OK" if counter.get(key, 0) > 0 else "MISS"
        print(f"  {marker:4s} {key}: {counter.get(key, 0)}")


def print_suite_matrix(traces: list[dict]) -> None:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for trace in traces:
        domain = normalize_domain(str(trace.get("domain", "unknown")))
        scenario = str(trace.get("scenario", "benign"))
        if int(trace.get("label", 0)) == 0:
            scenario = "benign"
        matrix[domain][scenario] += 1

    print("\nDomain x scenario coverage")
    scenarios = ["benign", *sorted(REQUIRED_MALICIOUS_SCENARIOS)]
    header = "domain".ljust(20) + "".join(scenario.replace("_", " ")[:10].rjust(12) for scenario in scenarios)
    print(header)
    for domain in sorted(REQUIRED_DOMAINS | set(matrix)):
        row = domain.ljust(20)
        for scenario in scenarios:
            row += str(matrix[domain].get(scenario, 0)).rjust(12)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report incremental AgentDojo/Groq collection coverage for paper seeds."
    )
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "real_traces.jsonl")
    parser.add_argument("--seed", type=Path, default=ROOT / "data" / "real_seed_traces.jsonl")
    args = parser.parse_args()

    traces = load_jsonl(args.input)
    seed_traces = load_jsonl(args.seed)
    print(f"Trace file: {args.input}")
    print(f"Total parsed traces: {len(traces)}")
    print(f"Seed file: {args.seed}")
    print(f"Current seed traces: {len(seed_traces)}")

    labels = count_labels(traces)
    domains = Counter(normalize_domain(str(trace.get("domain", ""))) for trace in traces)
    scenarios = Counter(
        str(trace.get("scenario", "unknown"))
        for trace in traces
        if int(trace.get("label", 0)) == 1
    )

    print_counter("Label counts", labels)
    print_counter("Domain coverage", domains, REQUIRED_DOMAINS)
    print_counter("Malicious scenario coverage", scenarios, REQUIRED_MALICIOUS_SCENARIOS)
    print_suite_matrix(traces)

    ok, messages = coverage_status(traces)
    print("\nPaper seed readiness")
    if ok:
        print("  OK: enough parsed traces exist to export structurally valid 48 balanced paper seeds.")
        print("  Next: run data/collect_real_traces.py with --skip-benchmark to refresh real_seed_traces.jsonl.")
    else:
        print("  NOT READY")
        for message in messages:
            print(f"  - {message}")
        print("  Keep collecting missing suites/scenarios; real_traces.jsonl is append/deduplicated.")


if __name__ == "__main__":
    main()
