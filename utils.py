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
    # Eval Setting
    parser.add_argument('--benchmark_version', type=str, default='v1.2', help='the version of agentdojo')
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18', help='gpt-4o-mini, gpt-4o')
    parser.add_argument("--suites", type=str, default="banking,slack,travel,workspace", help="which suites to use, separated by comma.")
    parser.add_argument('--force_rerun', action='store_true', help='Whether to force rerun.')
    parser.add_argument('--do_attack', action='store_true', help='Whether the setting is under attack.')
    parser.add_argument('--attack_type', type=str, default="important_instructions", help='The attack type, you can select from "direct, ignore_previous, system_message, injecagent, dos, swearwords_dos, captcha_dos, offensive_email_dos, felony_dos, important_instructions, important_instructions_no_user_name, important_instructions_no_model_name, important_instructions_no_names, important_instructions_wrong_model_name, important_instructions_wrong_user_name, tool_knowledge"')

    parser.add_argument('--target_user_tasks', type=str, default=None, help='User task number you want to evaluate, sperated by comma, such as "1,4,7".')
    parser.add_argument('--target_injection_tasks', type=str, default=None, help='Injection task number you want to specific evaluate, sperated by comma, such as "1,2,3".')

    # DRIFT Setting
    parser.add_argument("--build_constraints", action='store_true', help="Whether to build initial constraints.")
    parser.add_argument("--injection_isolation", action='store_true', help="Whether to detect injection instruction.")
    parser.add_argument("--dynamic_validation", action='store_true', help="Whether to validate dynamically.")
    parser.add_argument("--adaptive_attack", action='store_true', help="Whether to implement adaptive attack.")
    parser.add_argument("--align_claim", action='store_true', help="Append a task-specific claim to the injection asserting it is aligned with / required by the user's actual task (targets the injection detector).")
    parser.add_argument("--close_tag", action='store_true', help="Context-escape injection: prepend a closing tool-result tag (</tool_response>) plus a 'do this task first' prerequisite before the injected goal, so the payload renders outside the tool-data region of the model's chat template.")
    parser.add_argument("--repeated_instruction", action='store_true', help="Repeated-instruction attack: paraphrase the injected goal N ways (via the target client, cached to disk) and inject all paraphrases plus the original together, so any phrasing the injection detector misses survives removal.")
    parser.add_argument("--repeat_n", type=int, default=6, help="Number of paraphrases for --repeated_instruction; split evenly around the goal (default 6 -> 3 before, goal, 3 after).")

    # Environment
    parser.add_argument('--seed', type=int, default=98, help='Random Seed.')


    args = parser.parse_args()

    return args