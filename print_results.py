import argparse
import glob
import json
import os

# DEFAULT_SUITES = ["banking", "slack", "travel", "workspace", "github", "shopping", "dailylife"]
DEFAULT_SUITES = ["github", "shopping", "dailylife"]


def run_dir_name(model: str) -> str:
    """Mirror agentdojo's make_run_name: basename + '+progent' when Progent is on."""
    name = model.split("/")[-1]
    name += "+drift"
    return name


def summarize(log_dir: str, name: str, attack, suites):
    results = {}
    for suite in suites:
        if attack is None:
            pattern = f"{log_dir}/{name}/{suite}/user_task_*/none/none.json"
        else:
            pattern = f"{log_dir}/{name}/{suite}/user_task_*/{attack}/injection_task_*.json"
        files = glob.glob(pattern)
        if not files:
            continue
        utility_scores, security_scores = [], []
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
            if "utility" in data:
                utility_scores.append(data["utility"])
            if "security" in data:
                security_scores.append(data["security"])
        results[suite] = {
            "utility": sum(utility_scores) / len(utility_scores) if utility_scores else "N/A",
            "security": sum(security_scores) / len(security_scores) if security_scores else "N/A",
        }
    return results


def print_table(title: str, results: dict):
    print(f"\n=== {title} ===")
    if not results:
        print("(no logs found)")
        return
    print(f"{'Suite':<12} {'Utility':>8} {'Security':>10}")
    print("-" * 32)
    for suite, scores in results.items():
        print(f"{suite:<12} {str(scores['utility'])[:6]:>8} {str(scores['security'])[:6]:>10}")
    u = [v["utility"] for v in results.values() if isinstance(v["utility"], float)]
    s = [v["security"] for v in results.values() if isinstance(v["security"], float)]
    print("-" * 32)
    line = f"{'avg':<12}"
    if u:
        line += f" {sum(u) / len(u):>8.3f}"
    if s:
        line += f" {sum(s) / len(s):>10.3f}"
    print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize AgentDojo results (utility/security per suite).")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B",
                        help="Model id; basename and the +progent suffix are applied automatically.")
    parser.add_argument("--name", default=None,
                        help="Exact run-directory name under --log-dir (overrides --model).")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES,
                        help="Suites to summarize (e.g. shopping github dailylife for AgentDyn).")
    args = parser.parse_args()

    name = args.name or run_dir_name(args.model)
    print(f"Results for: {name}")
    # No-attack (utility) and under-attack (utility + security).
    for attack, title in [(None, "No attack"), ("important_instructions", "Attack: important_instructions")]:
        print_table(title, summarize(args.log_dir, name, attack, args.suites))
