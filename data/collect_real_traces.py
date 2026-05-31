import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DECOMMISSIONED_MODELS = {"llama3-70b-8192"}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def preflight_groq(api_key: str, timeout: int = 20) -> None:
    """Simple connectivity check against Groq's OpenAI-compatible /models endpoint."""
    req = Request(f"{GROQ_BASE_URL}/models", method="GET", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) >= 400:
                raise SystemExit(f"Groq preflight returned HTTP {getattr(resp, 'status', 'unknown')}")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Cloudflare edge policy may return 403 + code 1010 before auth is evaluated.
        # Allow the benchmark to continue and let the actual model call provide a definitive error.
        if exc.code == 403 and "1010" in body:
            print(
                "Warning: Groq preflight hit HTTP 403 (code 1010). "
                "This is usually a network/edge access policy. Continuing anyway.",
                file=sys.stderr,
            )
            return
        raise SystemExit(f"Groq preflight returned HTTP {exc.code}. Response body: {body[:500]}") from exc
    except Exception as exc:
        raise SystemExit(f"Cannot reach Groq API at {GROQ_BASE_URL}: {exc}") from exc


def _is_retryable_rate_limit_output(text: str) -> bool:
    lowered = text.lower()
    return "429" in lowered or "rate limit" in lowered or "too many requests" in lowered


def _is_unsupported_suite_output(text: str) -> bool:
    lowered = text.lower()
    return "keyerror" in lowered and ("get_suite" in lowered or "load_suites" in lowered)


def _is_insufficient_seed_output(text: str) -> bool:
    lowered = text.lower()
    return "need 15 label=0" in lowered or "need 10 label=1" in lowered


def _is_invalid_api_key_output(text: str) -> bool:
    lowered = text.lower()
    return (
        "invalid_api_key" in lowered
        or "invalid api key" in lowered
        or "401 unauthorized" in lowered
        or "authenticationerror" in lowered
    )


def show_collection_coverage(output: str, seed_output: str | None) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "check_collection_coverage.py"),
        "--input",
        str(ROOT / output),
    ]
    if seed_output:
        command.extend(["--seed", str(ROOT / seed_output)])
    subprocess.run(command, cwd=ROOT, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AgentDojo/Groq collection and convert logs into DARS JSONL."
    )
    parser.add_argument(
        "--suite",
        default="workspace,slack,travel,banking",
        help="AgentDojo suite name or comma-separated suite names.",
    )
    parser.add_argument(
        "--attack",
        default="none",
        help=(
            "AgentDojo attack name or comma-separated attack names. Use 'none' for benign. "
            "Examples: direct, tool_knowledge, dos, injecagent."
        ),
    )
    parser.add_argument(
        "--injection-task",
        default="",
        help="Optional AgentDojo injection task id, e.g. injection_task_5.",
    )
    parser.add_argument(
        "--user-task",
        default="",
        help="Optional AgentDojo user task id, e.g. user_task_0.",
    )
    parser.add_argument("--model-id", default="llama-3.3-70b-versatile", help="Groq model id.")
    parser.add_argument("--output", default="data/real_traces.jsonl", help="Converted JSONL output.")
    parser.add_argument(
        "--seed-output",
        default="data/real_seed_traces.jsonl",
        help="Validated 48-trace seed file used by data/generate_data.py.",
    )
    parser.add_argument(
        "--force-seed-refresh",
        action="store_true",
        help="Rewrite the 48-trace seed file from all accumulated parsed traces.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only parse existing runs/ logs; do not launch AgentDojo.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Do not perform the Groq /models connectivity check before benchmarking.",
    )
    args = parser.parse_args()

    hit_rate_limit = False

    if not args.skip_benchmark:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("Set GROQ_API_KEY before running live AgentDojo/Groq collection.")
        api_key = api_key.strip()
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = GROQ_BASE_URL
        if not args.skip_preflight:
            preflight_groq(api_key)
        suites = [suite.strip() for suite in args.suite.split(",") if suite.strip()]
        attacks = [attack.strip() for attack in args.attack.split(",") if attack.strip()]
        if not attacks:
            attacks = ["none"]
        for suite in suites:
            for attack in attacks:
                command = [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(ROOT / "scripts" / "run_groq_benchmark.ps1"),
                    "-Suite",
                    suite,
                    "-ModelId",
                    args.model_id,
                    "-Attack",
                    attack,
                ]
                if args.injection_task:
                    command.extend(["-InjectionTask", args.injection_task])
                if args.user_task:
                    command.extend(["-UserTask", args.user_task])
                result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
                if result.returncode == 0:
                    continue
                combined_output = "\n".join(part for part in [result.stdout, result.stderr] if part)
                if _is_retryable_rate_limit_output(combined_output):
                    hit_rate_limit = True
                    print(
                        f"Suite '{suite}' attack '{attack}' hit Groq rate limits and failed after retries. "
                        "Skipping this run and continuing.",
                        file=sys.stderr,
                    )
                    continue
                if _is_invalid_api_key_output(combined_output):
                    raise SystemExit(
                        "Groq rejected GROQ_API_KEY with 401 invalid_api_key. "
                        "Set a valid key in the same PowerShell session and rerun."
                    )
                if _is_unsupported_suite_output(combined_output):
                    print(
                        f"AgentDojo suite '{suite}' is not available in this benchmark version. "
                        "Skipping it and continuing.",
                        file=sys.stderr,
                    )
                    continue
                raise subprocess.CalledProcessError(
                    result.returncode,
                    command,
                    output=result.stdout,
                    stderr=result.stderr,
                )

    parse_command = [
        sys.executable,
        str(ROOT / "scripts" / "parse_agentdojo_logs.py"),
        "--output",
        str(ROOT / args.output),
    ]
    if args.seed_output:
        parse_with_seed = [*parse_command, "--seed-output", str(ROOT / args.seed_output)]
        if args.force_seed_refresh:
            parse_with_seed.append("--force-seed-refresh")
        result = subprocess.run(parse_with_seed, cwd=ROOT, check=False, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            combined_output = "\n".join(part for part in [result.stdout, result.stderr] if part)
            if _is_insufficient_seed_output(combined_output):
                if hit_rate_limit:
                    print(
                        "Warning: Partial run due to Groq rate limits; parsed traces were written, "
                        "but seed export was skipped because there are not enough benign/malicious traces yet.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Warning: Parsed traces were written, but seed export was skipped because "
                        "there are not enough benign/malicious traces yet.",
                        file=sys.stderr,
                    )
                show_collection_coverage(args.output, args.seed_output)
                return
            raise subprocess.CalledProcessError(
                result.returncode,
                parse_with_seed,
                output=result.stdout,
                stderr=result.stderr,
            )
        show_collection_coverage(args.output, args.seed_output)
    else:
        subprocess.run(parse_command, cwd=ROOT, check=True)
        show_collection_coverage(args.output, args.seed_output)


if __name__ == "__main__":
    main()
