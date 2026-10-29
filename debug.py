import argparse
import json
from pathlib import Path


def load_utilities(logdir: Path, pipeline: str, attack: str) -> dict[tuple[str, str], bool]:
    results: dict[tuple[str, str], bool] = {}
    base = logdir / pipeline
    if not base.is_dir():
        raise SystemExit(f"No logs found for pipeline '{pipeline}' at {base}")
    for json_path in base.glob(f"*/user_task_*/{attack}/*.json"):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        suite = data.get("suite_name", json_path.parts[-4])
        task = data.get("user_task_id", json_path.parts[-3])
        results[(suite, task)] = bool(data.get("utility", False))
    return results


def load_security(logdir: Path, pipeline: str, attack: str) -> dict[tuple[str, str, str], bool]:
    """Load security field; key = (suite, user_task, injection_task)."""
    results: dict[tuple[str, str, str], bool] = {}
    base = logdir / pipeline
    if not base.is_dir():
        return results
    for json_path in base.glob(f"*/user_task_*/{attack}/*.json"):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        suite = data.get("suite_name", json_path.parts[-4])
        task = data.get("user_task_id", json_path.parts[-3])
        injection = json_path.stem  # e.g. injection_task_0
        results[(suite, task, injection)] = bool(data.get("security", False))
    return results


def _task_sort_key(item: tuple[str, str]) -> tuple[str, int, str]:
    suite, task = item
    num = task.removeprefix("user_task_")
    return (suite, int(num)) if num.isdigit() else (suite, 1_000_000, task)


def _injection_sort_key(item: tuple[str, str, str]) -> tuple[str, int, str, int, str]:
    suite, task, injection = item
    tnum = task.removeprefix("user_task_")
    inum = injection.removeprefix("injection_task_")
    return (
        suite,
        int(tnum) if tnum.isdigit() else 1_000_000, task,
        int(inum) if inum.isdigit() else 1_000_000, injection,
    )


def compare(logdir: Path, baseline_pipeline: str, target_pipeline: str, attack: str, label: str) -> None:
    baseline = load_utilities(logdir, baseline_pipeline, attack)
    target   = load_utilities(logdir, target_pipeline, attack)

    common = sorted(baseline.keys() & target.keys(), key=_task_sort_key)

    regressions = [k for k in common if baseline[k] and not target[k]]
    target_fail = [k for k in common if not target[k]]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{label}")
    print(f"  baseline: {baseline_pipeline}")
    print(f"  target:   {target_pipeline}")
    print(f"  attack:   {attack}")
    print(f"  tasks compared: {len(common)}")
    print(sep)

    print(f"\nREGRESSED (baseline pass -> target FAIL): {len(regressions)}")
    for suite, task in regressions:
        print(f"  [{suite}] {task}")

    print(f"\ntarget fail: {len(target_fail)}")
    for suite, task in target_fail:
        print(f"  [{suite}] {task}")


def print_security(logdir: Path, pipeline: str, attack: str = "important_instructions") -> None:
    """Print (suite, user_task, injection_task) where security=True."""
    sec = load_security(logdir, pipeline, attack)
    passed = sorted([k for k, v in sec.items() if v], key=_injection_sort_key)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"SECURITY PASS (security=True)")
    print(f"  pipeline: {pipeline}")
    print(f"  attack:   {attack}")
    print(f"  count: {len(passed)} / {len(sec)}")
    print(sep)
    for suite, task, injection in passed:
        print(f"  [{suite}] {task} / {injection}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model", help="e.g. Llama-3.3-70B-Instruct")
    parser.add_argument("--logdir", default="logs", type=Path)
    parser.add_argument("--attack", default="none")
    parser.add_argument("--sec-attack", default="important_instructions",
                        help="attack name to load security field from")
    args = parser.parse_args()

    m = args.base_model
    compare(args.logdir, m, f"{m}+drift", args.attack, "[1/2] baseline -> +drift")
    print_security(args.logdir, f"{m}+drift", args.sec_attack)


if __name__ == "__main__":

    base_dir = Path("./logs/Qwen3-30B-A3B-Instruct-2507+drift")
    # ATTACK_FOLDER =  "important_instructions"
    # ATTACK_FOLDER =  "important_instructions+noiso"
    # ATTACK_FOLDER = "important_instructions+noiso+alignclaim"
    # ATTACK_FOLDER = "important_instructions+alignclaim"
    # ATTACK_FOLDER =  "important_instructions+closetag"
    # ATTACK_FOLDER = "important_instructions+repeat6"
    ATTACK_FOLDER = "important_instructions+alignclaim+repeat6"


    g_success = g_utility = g_total = 0
    targeted_suites = ["dailylife", "shopping", "github"]

    for suite in sorted(base_dir.iterdir()):
        if not suite.is_dir():
            continue

        if suite.name not in targeted_suites:
            continue

        s_success = s_utility = s_total = 0

        for task in suite.iterdir():
            if "user_task" not in task.name:
                continue
            folder = task / ATTACK_FOLDER
            if not folder.exists():
                continue

            for json_file in folder.iterdir():
                if json_file.suffix != ".json":
                    continue
                with open(json_file, "r") as f:
                    file_json = json.load(f)

                security = bool(file_json.get("security", False))   # attack success
                utility  = bool(file_json.get("utility", False))    # utility under attack

                if file_json.get("security") is None:
                    print(f"[security=None] {suite.name}/{task.name}/{json_file.name}")

                s_success += security
                s_utility += utility
                s_total   += 1

                print(f"Suite: {suite.name}, Task: {task.name}, File: {json_file.name}, "
                      f"AttackSuccess={security}, UtilityUnderAttack={utility}")

        # 累加到全局(放在 suite 级,避免重复计数)
        g_success += s_success
        g_utility += s_utility
        g_total   += s_total

        if s_total > 0:
            print(f"Suite: {suite.name}, ASR: {s_success / s_total:.4f}, "
                  f"UtilityUnderAttack: {s_utility / s_total:.4f}  (n={s_total})")

    if g_total > 0:
        print(f"\nOverall ASR:                {g_success / g_total:.4f}")
        print(f"Overall UtilityUnderAttack: {g_utility / g_total:.4f}  (n={g_total})")