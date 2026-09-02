#!/usr/bin/env python
"""Aggregate ASB OPI JSON traces: per-label ASR/utility/refuse + baseline-vs-defense case diff.

Reads the per-task AgentDojo-format traces written under ``logs/json/<model>/<label>/...`` and:
  1. Prints a per-label summary (n, ASR, utility, refuse-rate), overall and per attack_type.
  2. If a baseline and a defense label are both present, does a per-case diff (matched on
     agent / injection-task / user-task / attack_type): how many attacks the defense neutralised
     (baseline hit -> defense miss), how many still got through, and how many benign tasks the
     defense broke (utility 1 -> 0, i.e. over-defense).

Usage:
    python scripts/analyze.py                       # scan ./logs/json
    python scripts/analyze.py --logs-dir logs/json  # explicit
    python scripts/analyze.py --baseline react_baseline --defense camel --dump-diff diff.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def _label(d):
    if d.get("defense_type"):
        return d["defense_type"]
    kind = "clean" if d.get("clean") else "baseline"
    return f"{d.get('workflow_mode', 'unknown')}_{kind}"


def _case_key(d):
    # One attack instance: which agent, which injection task, which user task, which attack style.
    return (
        d.get("suite_name"),
        d.get("injection_task_id") or d.get("attacker_tool"),
        d.get("user_task_id"),
        d.get("attack_type"),
    )


def _load(logs_dir):
    records = []
    for path in glob.glob(os.path.join(logs_dir, "**", "*.json"), recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        if "attack_success" not in d and "security" not in d:
            continue  # not an ASB trace
        records.append(d)
    return records


def _rate(num, den):
    return f"{num/den:.3f}" if den else "  -  "


def summarize(records, by_attack_type=False):
    # group -> [n, asr, util, refuse]
    groups = defaultdict(lambda: [0, 0, 0, 0])
    for d in records:
        keys = [_label(d)]
        if by_attack_type:
            keys = [f"{_label(d)}  |  {d.get('attack_type')}"]
        asr = bool(d.get("attack_success", d.get("security")))
        util = bool(d.get("utility"))
        refused = str(d.get("refuse")) == "0"
        for k in keys:
            g = groups[k]
            g[0] += 1
            g[1] += asr
            g[2] += util
            g[3] += refused
    header = "label" if not by_attack_type else "label | attack_type"
    print(f"\n{header:48s} {'n':>6s} {'ASR':>8s} {'utility':>9s} {'refuse':>8s}")
    print("-" * 84)
    for k in sorted(groups):
        n, a, u, r = groups[k]
        print(f"{k:48s} {n:>6d} {_rate(a,n):>8s} {_rate(u,n):>9s} {_rate(r,n):>8s}")


def diff(records, baseline_label, defense_label, dump_csv=None):
    by_label_case = defaultdict(dict)
    for d in records:
        by_label_case[_label(d)][_case_key(d)] = d

    base = by_label_case.get(baseline_label)
    dfn = by_label_case.get(defense_label)
    if not base or not dfn:
        avail = ", ".join(sorted(by_label_case))
        print(f"\n[diff] need both '{baseline_label}' and '{defense_label}'. Available labels: {avail}")
        return

    shared = sorted(set(base) & set(dfn))
    print(f"\n=== {baseline_label}  vs  {defense_label}  ({len(shared)} matched cases) ===")
    neutralised, still_hit, newly_hit, broke_util, fixed_util = [], [], [], [], []
    for k in shared:
        b, v = base[k], dfn[k]
        ba = bool(b.get("attack_success", b.get("security")))
        va = bool(v.get("attack_success", v.get("security")))
        bu, vu = bool(b.get("utility")), bool(v.get("utility"))
        if ba and not va:
            neutralised.append(k)
        if ba and va:
            still_hit.append(k)
        if not ba and va:
            newly_hit.append(k)
        if bu and not vu:
            broke_util.append(k)
        if not bu and vu:
            fixed_util.append(k)

    n = len(shared)
    b_asr = sum(bool(base[k].get("attack_success", base[k].get("security"))) for k in shared)
    d_asr = sum(bool(dfn[k].get("attack_success", dfn[k].get("security"))) for k in shared)
    b_u = sum(bool(base[k].get("utility")) for k in shared)
    d_u = sum(bool(dfn[k].get("utility")) for k in shared)
    print(f"  ASR:      {baseline_label} {_rate(b_asr,n)}  ->  {defense_label} {_rate(d_asr,n)}")
    print(f"  utility:  {baseline_label} {_rate(b_u,n)}  ->  {defense_label} {_rate(d_u,n)}")
    print(f"  attacks neutralised by defense (hit -> miss): {len(neutralised)}")
    print(f"  attacks still succeeding (hit -> hit):        {len(still_hit)}")
    print(f"  attacks the defense INTRODUCED (miss -> hit): {len(newly_hit)}")
    print(f"  benign tasks broken by defense (util 1 -> 0): {len(broke_util)}  (over-defense)")
    print(f"  benign tasks fixed by defense (util 0 -> 1):  {len(fixed_util)}")

    if dump_csv:
        with open(dump_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["agent", "injection_task", "user_task", "attack_type",
                        "base_ASR", "defense_ASR", "base_util", "defense_util", "verdict"])
            for k in shared:
                b, v = base[k], dfn[k]
                ba = int(bool(b.get("attack_success", b.get("security"))))
                va = int(bool(v.get("attack_success", v.get("security"))))
                bu, vu = int(bool(b.get("utility"))), int(bool(v.get("utility")))
                verdict = ("neutralised" if ba and not va else
                           "still_hit" if ba and va else
                           "introduced" if not ba and va else "clean")
                w.writerow([*k, ba, va, bu, vu, verdict])
        print(f"  per-case diff written to {dump_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs/json")
    ap.add_argument("--baseline", default=None, help="baseline label (default: auto-detect a *_baseline)")
    ap.add_argument("--defense", default="camel", help="defense label to compare (default: camel)")
    ap.add_argument("--dump-diff", default=None, help="write the per-case diff to this CSV")
    args = ap.parse_args()

    records = _load(args.logs_dir)
    if not records:
        print(f"No traces found under {args.logs_dir}")
        return
    print(f"Loaded {len(records)} traces from {args.logs_dir}")

    summarize(records)
    summarize(records, by_attack_type=True)

    labels = {_label(d) for d in records}
    baseline = args.baseline or next((l for l in sorted(labels) if l.endswith("_baseline")), None)
    if baseline and args.defense in labels:
        diff(records, baseline, args.defense, args.dump_diff)
    elif baseline:
        print(f"\n[diff] defense label '{args.defense}' not found; labels present: {sorted(labels)}")


if __name__ == "__main__":
    main()
