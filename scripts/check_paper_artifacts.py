import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "seed": ROOT / "data" / "real_seed_traces.jsonl",
    "train": ROOT / "data" / "train_traces.jsonl",
    "val": ROOT / "data" / "val_traces.jsonl",
    "test": ROOT / "data" / "test_traces.jsonl",
    "model": ROOT / "models_saved" / "dars_model_real.pt",
}
EXPECTED_COUNTS = {
    "seed": {0: 24, 1: 24},
    "train": {0: 2450, 1: 1260},
    "val": {0: 525, 1: 270},
    "test": {0: 525, 1: 270},
}
REQUIRED_DOMAINS = {"workspace", "slack", "data_manipulation", "external_api"}
REQUIRED_MALICIOUS_SCENARIOS = {
    "S1_prompt_injection",
    "S2_privilege_escalation",
    "S3_malicious_api_usage",
    "S4_data_exfiltration",
    "S5_denial_of_wallet",
    "S6_stealth_mimicry",
}
MALICIOUS_SEEDS_PER_SCENARIO = 4
BENIGN_SEEDS_PER_DOMAIN = 6
REQUIRED_FEATURES = {
    "bdi_deviation",
    "privilege_level",
    "privilege_escalation",
    "transition_anomaly",
    "sequence_anomaly",
    "token_burst",
}
DISALLOWED_SEED_SOURCES = {"agentdojo_seed_synthetic", "template", "synthetic_template"}
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


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_counts(traces: list[dict]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for trace in traces:
        counts[int(trace.get("label", 0))] = counts.get(int(trace.get("label", 0)), 0) + 1
    return counts


def normalize_domain(domain: str) -> str:
    value = (domain or "unknown").strip().lower().replace("-", "_")
    return DOMAIN_ALIASES.get(value, value)


def check_trace_file(name: str, path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        print(f"FAIL {name}: missing or empty: {path}")
        return False

    traces = read_jsonl(path)
    counts = label_counts(traces)
    expected = EXPECTED_COUNTS[name]
    ok = counts == expected
    status = "OK" if ok else "FAIL"
    print(f"{status} {name}: labels={counts}, expected={expected}, rows={len(traces)}")

    empty_calls = sum(1 for trace in traces if not trace.get("tool_calls"))
    if empty_calls:
        print(f"FAIL {name}: {empty_calls} traces have empty tool_calls.")
        ok = False

    if name == "seed":
        domains = {normalize_domain(str(trace.get("domain", ""))) for trace in traces}
        missing_domains = REQUIRED_DOMAINS - domains
        if missing_domains:
            print(f"FAIL seed: missing required domains {sorted(missing_domains)}.")
            ok = False
        benign_domain_counts = {
            domain: sum(
                1
                for trace in traces
                if int(trace.get("label", 0)) == 0
                and normalize_domain(str(trace.get("domain", ""))) == domain
            )
            for domain in REQUIRED_DOMAINS
        }
        low_benign_domains = {
            domain: count
            for domain, count in benign_domain_counts.items()
            if count < BENIGN_SEEDS_PER_DOMAIN
        }
        if low_benign_domains:
            print(
                "FAIL seed: expected at least "
                f"{BENIGN_SEEDS_PER_DOMAIN} benign traces per domain, found {low_benign_domains}."
            )
            ok = False

        malicious_scenarios = {
            str(trace.get("scenario", ""))
            for trace in traces
            if int(trace.get("label", 0)) == 1
        }
        missing_scenarios = REQUIRED_MALICIOUS_SCENARIOS - malicious_scenarios
        if missing_scenarios:
            print(f"FAIL seed: missing required malicious scenarios {sorted(missing_scenarios)}.")
            ok = False
        scenario_counts = {
            scenario: sum(
                1
                for trace in traces
                if int(trace.get("label", 0)) == 1 and trace.get("scenario") == scenario
            )
            for scenario in REQUIRED_MALICIOUS_SCENARIOS
        }
        too_few = {
            scenario: count
            for scenario, count in scenario_counts.items()
            if count < MALICIOUS_SEEDS_PER_SCENARIO
        }
        if too_few:
            print(
                "FAIL seed: expected at least "
                f"{MALICIOUS_SEEDS_PER_SCENARIO} malicious traces per scenario, found {too_few}."
            )
            ok = False

        bad_sources = sorted(
            {
                str(trace.get("source", "")).lower()
                for trace in traces
                if str(trace.get("source", "")).lower() in DISALLOWED_SEED_SOURCES
            }
        )
        if bad_sources:
            print(f"FAIL seed: synthetic/template sources found {bad_sources}.")
            ok = False

    missing_severity = sum(1 for trace in traces if "severity" not in trace)
    if missing_severity and name != "seed":
        print(f"FAIL {name}: {missing_severity} traces are missing severity values.")
        ok = False

    return ok


def check_model(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        print(f"FAIL model: missing or empty: {path}")
        return False

    try:
        import torch
    except ModuleNotFoundError:
        print("FAIL model: torch is not installed in this Python environment; cannot inspect checkpoint.")
        return False

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"model_state", "standard_lstm_state", "extractor_state", "shap_background", "config"}
    missing = required_keys - set(checkpoint)
    if missing:
        print(f"FAIL model: checkpoint missing keys {sorted(missing)}.")
        return False

    config = checkpoint.get("config", {})
    feature_names = config.get("feature_names")
    missing_features = REQUIRED_FEATURES - set(feature_names or [])
    if missing_features:
        print(f"FAIL model: checkpoint feature_names missing {sorted(missing_features)}.")
        return False
    print(f"OK model: feature_names={feature_names}, shap_background={tuple(checkpoint['shap_background'].shape)}")
    return True


def check_code_contracts() -> bool:
    ok = True
    extractor_tree = ast.parse((ROOT / "src" / "feature_extraction" / "extractor.py").read_text(encoding="utf-8"))
    risk_tree = ast.parse((ROOT / "src" / "scoring" / "risk.py").read_text(encoding="utf-8"))
    feature_names = []
    static_rules = []
    for node in extractor_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FEATURE_NAMES":
                    feature_names = ast.literal_eval(node.value)
    for node in risk_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RULE_DEFINITIONS":
                    static_rules = ast.literal_eval(node.value)

    missing_features = REQUIRED_FEATURES - set(feature_names)
    if missing_features:
        print(f"FAIL code: FEATURE_NAMES missing {sorted(missing_features)}.")
        ok = False
    if len(static_rules) != 42:
        print(f"FAIL code: expected 42 static rules, found {len(static_rules)}.")
        ok = False
    if ok:
        print(f"OK code: features={feature_names}, static_rules={len(static_rules)}")
    return ok


def main() -> None:
    ok = True
    ok = check_code_contracts() and ok
    for name in ["seed", "train", "val", "test"]:
        ok = check_trace_file(name, REQUIRED_FILES[name]) and ok
    ok = check_model(REQUIRED_FILES["model"]) and ok

    if not ok:
        raise SystemExit(
            "Paper artifacts are incomplete. Do not report ICCCNet metrics until all checks pass."
        )
    print("All paper artifacts are present and structurally valid.")


if __name__ == "__main__":
    main()
