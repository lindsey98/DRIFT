import argparse
import random
import numpy as np
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    return seed

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def get_args(description='DRIFT'):
    parser = argparse.ArgumentParser(description=description)

    # Model is positional:  python pipeline_main.py MODEL [options]
    parser.add_argument('model', type=str,
                        help='Model name, e.g. gpt-4o-mini-2024-07-18, Qwen3.6-35B-A3B, anthropic:claude-sonnet-4-5-20250929.')

    # Suites are space-separated:  --suite banking slack travel workspace
    parser.add_argument('--suite', dest='suites', nargs='+',
                        default=['banking', 'slack', 'travel', 'workspace'],
                        help='Suites to run, space-separated. Available: banking slack travel workspace shopping github dailylife.')
    parser.add_argument('--benchmark_version', type=str, default='v1.2', help='agentdojo benchmark version.')

    # Attack:  --run-attack --attack <name>
    parser.add_argument('--run-attack', dest='do_attack', action='store_true',
                        help='Run under attack (omit for the benign/no-attack setting).')
    parser.add_argument('--attack', dest='attack_type', type=str, default='important_instructions',
                        help='Attack name (any registered agentdojo attack), e.g. important_instructions, tool_knowledge, data_only_syntactic, chat_inject_qwen3.')

    # Defense:  --defense none | drift
    parser.add_argument('--defense', type=str, default='none',
                        help="Defense: 'none' (undefended original model) or 'drift' "
                             "(= build_constraints + injection_isolation + dynamic_validation).")

    # Task selection (space-separated ids or numbers; default = all)
    parser.add_argument('--user-task', '-ut', dest='target_user_tasks', type=str, nargs='*', default=None,
                        help='User tasks to run (space-separated ids or numbers, e.g. -ut 1 4 7). Default: all.')
    parser.add_argument('--injection-task', '-it', dest='target_injection_tasks', type=str, nargs='*', default=None,
                        help='Injection tasks to run (space-separated ids or numbers, e.g. -it 0 1 2). Default: all.')

    # Optional
    parser.add_argument('--force_rerun', action='store_true', help='Recompute even if a result JSON already exists.')
    parser.add_argument('--html', action='store_true', help='Also save a rendered HTML trace next to each result JSON (<name>.json + <name>.html).')

    # Attack modifiers (compose with --run-attack)
    parser.add_argument("--adaptive_attack", action='store_true', help="Append the adaptive-attack claim to the injection.")
    parser.add_argument("--align_claim", action='store_true', help="Wrap the injected goal with an aligned/required claim (targets the injection detector).")
    parser.add_argument("--close_tag", action='store_true', help="Context-escape injection: close the tool-result tag so the payload renders outside the tool-data region.")
    parser.add_argument("--repeated_instruction", action='store_true', help="Inject the goal plus N diverse paraphrases (sandwiched), so any phrasing the detector misses survives.")
    parser.add_argument("--repeat_n", type=int, default=6, help="Number of paraphrases for --repeated_instruction (default 6 -> 3 before, goal, 3 after).")

    parser.add_argument('--seed', type=int, default=98, help='Random seed.')

    args = parser.parse_args()

    # Expand --defense into the three DRIFT component flags (all-on for 'drift', all-off otherwise).
    drift_on = (args.defense or 'none').strip().lower() == 'drift'
    args.build_constraints = drift_on
    args.injection_isolation = drift_on
    args.dynamic_validation = drift_on

    return args