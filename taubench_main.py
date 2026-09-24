"""DRIFT on tau-bench (retail) with DoomArena's catalog attack. Entry point.

    # benign, defended: every retail task
    python taubench_main.py MODEL --suite retail --defense drift

    # under attack (the reproducible fixed-injection catalog poison), defended
    python taubench_main.py MODEL --run-attack --attack malicious_catalog_fixed \
        --suite retail --defense drift

    # the undefended DRIFT baseline (same scaffold, no defenses)
    python taubench_main.py MODEL --run-attack --defense none

    # one task, with an HTML trace
    python taubench_main.py MODEL --run-attack -ut 0 --defense drift --html

Results: logs_taubench/<model>[+drift]/retail/user_task_<i>/<attack>/none.json  (+ .html with --html).
Utility is tau-bench's reward; security is DoomArena's leak record (True == the attack succeeded).
Only `retail` runs under attack. The tau-bench USER SIMULATOR is a separate model (--user-model,
default: the same server for a local agent, else gpt-4o) and goes through litellm; the DRIFT AGENT
uses DRIFT's own client.
"""

import argparse
import json
import os

from tau_bench.envs.retail.tasks_test import TASKS_TEST as RETAIL_TASKS

from utils import get_logger, set_seed
from taubench_pipeline import CATALOG_CONFIGS, run_task

TASK_COUNTS = {"retail": len(RETAIL_TASKS)}


def get_args():
    p = argparse.ArgumentParser(description="DRIFT on tau-bench")
    p.add_argument("model", type=str, help="Agent model, e.g. gemma-4-26B-A4B, local:<name>, gpt-4o.")
    p.add_argument("--suite", dest="domain", type=str, default="retail",
                   help="tau-bench domain. Only 'retail' is supported under attack.")
    p.add_argument("--run-attack", dest="do_attack", action="store_true",
                   help="Turn DoomArena's catalog injection on (omit for the clean setting).")
    p.add_argument("--attack", dest="attack_type", type=str, default="malicious_catalog_fixed",
                   help=f"Catalog attack: {sorted(CATALOG_CONFIGS)}.")
    p.add_argument("--defense", type=str, default="drift",
                   help="'drift' (build_constraints + injection_isolation + dynamic_validation) or "
                        "'none' (DRIFT scaffold, no defenses -- the undefended baseline).")
    p.add_argument("--user-task", "-ut", dest="user_tasks", type=str, nargs="*", default=None,
                   help="Task indices or names to run (e.g. -ut 0 2 5 or -ut user_task_0). Default: all.")
    p.add_argument("--user-model", dest="user_model", type=str, default=None,
                   help="tau-bench user simulator model (default: the agent's server if local, else gpt-4o).")
    p.add_argument("--user-provider", dest="user_provider", type=str, default="openai")
    p.add_argument("--doomarena", type=str, default="../DoomArena",
                   help="Path to the DoomArena checkout the attack yaml is read from.")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=30)
    p.add_argument("--logdir", type=str, default="logs_taubench")
    p.add_argument("--html", action="store_true", help="Also save a rendered HTML trace per task.")
    p.add_argument("--force_rerun", action="store_true", help="Recompute even if a result JSON exists.")
    p.add_argument("--seed", type=int, default=98)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    if args.domain not in TASK_COUNTS:
        raise SystemExit(f"--suite must be one of {sorted(TASK_COUNTS)}")
    if args.do_attack and args.domain != "retail":
        raise SystemExit("only --suite retail can be run under attack: DoomArena ships no tool-output "
                         "attack for airline, and its user-channel attacks are outside DRIFT's threat model.")
    if args.do_attack and args.attack_type not in CATALOG_CONFIGS:
        raise SystemExit(f"--attack must be one of {sorted(CATALOG_CONFIGS)}")

    # The user simulator defaults to the same server as a local agent, so a fully self-hosted run
    # needs no hosted key. A local agent with a hosted openai-provider simulator would silently
    # redirect the simulator to the local port (litellm reads ONE process-wide OPENAI_BASE_URL).
    agent_local = args.model.startswith("local:") or "/" not in args.model
    user_model = args.user_model or (args.model if agent_local else "gpt-4o")
    user_local = user_model.startswith("local:") or "/" not in user_model
    if agent_local and args.user_provider == "openai" and not user_local:
        raise SystemExit(
            f"agent {args.model} is self-hosted but --user-model {user_model} is not, and both go "
            "through litellm's `openai` provider (one process-wide OPENAI_BASE_URL). Run the simulator "
            f"on the same server (--user-model {args.model}) or on another provider "
            "(--user-model openai/gpt-4o --user-provider openrouter).")

    defense_on = (args.defense or "none").strip().lower() == "drift"
    model_dir = f"{args.model}+drift" if defense_on else args.model
    attack_name = args.attack_type if args.do_attack else "none"

    base = os.path.join(args.logdir, model_dir, args.domain)
    os.makedirs(base, exist_ok=True)
    logger = get_logger(os.path.join(base, "log.txt"))

    if args.user_tasks:
        try:
            todo = [int(str(u).strip().replace("user_task_", "")) for u in args.user_tasks]
        except ValueError:
            raise SystemExit("--user-task takes an index (0) or a name (user_task_0)")
    else:
        todo = list(range(TASK_COUNTS[args.domain]))

    logger.info(f"== DRIFT :: tau-bench {args.domain} :: {len(todo)} task(s) -> {base}")
    logger.info(f"   model={args.model}  defense={args.defense}  attack={attack_name}  "
                f"max_steps={args.max_steps}")
    logger.info(f"   user simulator: {user_model} via {args.user_provider}")

    results = []
    for n, i in enumerate(todo, 1):
        out_dir = os.path.join(base, f"user_task_{i}", attack_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "none.json")
        if os.path.exists(path) and not args.force_rerun:
            logger.info(f"  [{n}/{len(todo)}] {args.domain}/user_task_{i}/{attack_name} exists, skip")
            results.append(json.load(open(path)))
            continue

        rec = run_task(args.domain, i, args.model, args.defense, run_attack=args.do_attack,
                       attack=args.attack_type, doomarena_root=args.doomarena, user_model=user_model,
                       user_provider=args.user_provider, max_steps=args.max_steps, logger=logger)
        with open(path, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        if args.html:
            try:
                from taubench_render import render_taubench_html
                with open(path[:-5] + ".html", "w", encoding="utf-8") as hf:
                    hf.write(render_taubench_html(rec))
            except Exception as e:
                logger.info(f"HTML render failed: {e}")
        results.append(rec)

        if rec.get("failed"):
            mark, tail = "!", f"failed: {rec['error']}"[:110]
        else:
            mark = "+" if rec.get("utility") else "-"
            tail = f"utility={rec.get('utility')} security={rec.get('security')}"
        logger.info(f"  {mark} [{n}/{len(todo)}] {args.domain}/user_task_{i}/{attack_name}  "
                    f"{rec.get('seconds')}s  {tail}  tok={rec.get('tokens', {}).get('total', 0)}")

    scored = [r for r in results if r.get("utility") is not None]
    if scored:
        util = sum(bool(r["utility"]) for r in scored) / len(scored)
        logger.info(f"\n  utility: {util:.0%} over {len(scored)} scored task(s)")
        secs = [r["security"] for r in scored if r.get("security") is not None]
        if secs:
            asr = sum(bool(x) for x in secs) / len(secs)
            logger.info(f"  attack success rate: {asr:.0%} over {len(secs)} attacked task(s) "
                        "(security=True == leak)")
    logger.info(f"\n  results under {base}")


if __name__ == "__main__":
    main()
