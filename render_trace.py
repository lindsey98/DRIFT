"""Render a DRIFT result record (the dict dumped to <task>.json) as a standalone HTML
trace, saved next to the JSON.

Layout follows the run's actual timeline:
  - a Config card at the top with the INITIAL plan DRIFT built (trajectory + checklist)
    and the tool-permission table -- these are decided up front;
  - the conversation, with DRIFT's runtime decisions interleaved where they happen:
    after each assistant turn, the dynamic validator's verdicts (what it extended /
    auto-approved / rejected, and why); after each tool result, the injection isolator's
    action (what it detected and stripped).

Purely presentational and defensive: any missing/odd field is skipped, never raises, so a
render failure can't break the eval loop.
"""
import html
import json
import re

_CSS = """
  :root { --usr:#2563eb; --ast:#16a34a; --tool:#d97706; --cfg:#7c3aed; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:"Helvetica Neue", Arial, sans-serif; background:#fff; display:flex; justify-content:center; padding:24px; color:#1f2937; }
  .figure { width:940px; }
  .title { font-size:15px; font-weight:700; color:#111; margin-bottom:4px; }
  .title span { font-weight:400; color:#666; font-size:13px; }
  .subnote { font-size:11.5px; color:#94a3b8; margin-bottom:12px; }
  .msg { border:1.5px solid; border-radius:8px; padding:10px 14px; margin-bottom:10px; font-size:12.5px; line-height:1.45; position:relative; }
  .role { display:inline-block; font-size:10.5px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:#fff; border-radius:4px; padding:2px 8px; margin-bottom:6px; }
  .usr { border-color:var(--usr); background:#eff6ff; } .usr .role { background:var(--usr); }
  .ast { border-color:var(--ast); background:#f0fdf4; } .ast .role { background:var(--ast); }
  .tool { border-color:var(--tool); background:#fffbeb; margin-left:32px; } .tool .role { background:var(--tool); }
  .cfgcard { border-color:var(--cfg); background:#f5f3ff; } .cfgcard .role { background:var(--cfg); }
  .valcard { border-color:var(--cfg); background:#faf5ff; margin-left:32px; } .valcard .role { background:var(--cfg); }
  .isocard { border-color:#d8b4fe; background:#f3e8ff; margin-left:32px; } .isocard .role { background:#9333ea; }
  code, .mono { font-family:"SF Mono", Menlo, Consolas, monospace; font-size:11.5px; }
  .call { background:#dcfce7; border:1px solid #86efac; border-radius:6px; padding:6px 10px; margin-top:6px; display:block; white-space:pre-wrap; word-break:break-word; color:#14532d; }
  .thought { background:#f0fdf4; border:1px dashed #86efac; border-radius:6px; padding:6px 10px; margin-top:6px; display:block; color:#14532d; white-space:pre-wrap; word-break:break-word; }
  .result { background:#fef3c7; border:1px solid #fcd34d; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; white-space:pre-wrap; word-break:break-word; color:#78350f; }
  .final { background:#ecfdf5; border:1px solid #6ee7b7; border-radius:6px; padding:6px 10px; margin-top:6px; display:block; white-space:pre-wrap; word-break:break-word; color:#065f46; }
  .cfgbox { background:#ede9fe; border:1px solid #c4b5fd; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; white-space:pre-wrap; word-break:break-word; color:#4c1d95; }
  .isobox { background:#faf5ff; border:1px solid #e9d5ff; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; color:#581c87; white-space:pre-wrap; word-break:break-word; }
  .strike { text-decoration:line-through; color:#b91c1c; }
  .box { margin-top:8px; border-top:1px dashed #c4b5fd; padding-top:8px; }
  .box-label { font-size:10.5px; font-weight:700; color:var(--cfg); letter-spacing:0.06em; margin-bottom:4px; }
  .perm-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:2px 18px; margin-top:4px; }
  .perm-grid div { font-size:11px; color:#334155; }
  .tag { display:inline-block; font-size:9.5px; font-weight:700; color:#fff; border-radius:3px; padding:0 4px; margin-right:5px; min-width:20px; text-align:center; }
  .r { background:#94a3b8; } .w { background:#d97706; } .x { background:#b91c1c; }
  .ok { color:#15803d; font-weight:700; } .bad { color:#b91c1c; font-weight:700; } .warn { color:#b45309; font-weight:700; }
  .dec { font-size:11.5px; margin:2px 0; color:#4c1d95; }
  .reason { color:#6b21a8; font-style:italic; }
  .nocontent { color:#6b7280; font-style:italic; font-size:11.5px; }
  .foot { font-size:11px; color:#64748b; margin-top:6px; }
"""

# validator outcomes that let the call through vs block it
_PASS = {"on_plan", "extend_read_auto", "extend_aligned", "checklist_user_approved"}


def _esc(s):
    return html.escape("" if s is None else str(s))


def _content_text(content):
    if isinstance(content, list):
        return "\n".join(str(b.get("content", "")) for b in content if isinstance(b, dict))
    return "" if content is None else str(content)


def _split_assistant(text):
    def grab(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
        return m.group(1).strip() if m else ""
    return grab("function_thought"), grab("function_call"), grab("final_answer")


def _call_names(msg, call_text):
    """Function names invoked in an assistant turn, from tool_calls or the call text."""
    names = []
    for c in (msg.get("tool_calls") or []):
        fn = c.get("function")
        if isinstance(fn, dict):
            fn = fn.get("name")
        if fn:
            names.append(fn)
    if not names and call_text:
        names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", call_text)
    return names


def _perm_tag(perm):
    cls = {"Read": "r", "Write": "w", "Execute": "x"}.get(perm, "r")
    ab = {"Read": "R", "Write": "W", "Execute": "X"}.get(perm, "?")
    return f'<span class="tag {cls}">{ab}</span>'


def _pretty_checklist(raw):
    if not raw or raw == "None":
        return ""
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except Exception:
        return str(raw).strip()


def _render_decision(ev):
    """One runtime validator event (trajectory / checklist / format_retry / early_exit)."""
    t = ev.get("type")
    if t == "trajectory_decision":
        outcome = ev.get("outcome", "")
        cls = "ok" if outcome in _PASS else "bad"
        fn = _esc(ev.get("function"))
        args = ev.get("args")
        argstr = f"({_esc(args)})" if args else ""
        reason = f"<div class='reason'>{_esc(ev.get('reason'))}</div>" if ev.get("reason") else ""
        return (f"<div class='msg valcard'><span class='role'>Validator · trajectory</span>"
                f"<div class='dec'><span class='{cls}'>{_esc(outcome)}</span> · {fn}{argstr} "
                f"· perm={_esc(ev.get('permission'))}</div>{reason}</div>")
    if t == "checklist_decision":
        outcome = ev.get("outcome", "")
        cls = "ok" if outcome in _PASS else "bad"
        calls = ev.get("calls") or []
        cs = ", ".join(_esc(c.get("function")) for c in calls) if calls else _esc(ev.get("function"))
        reason = f"<div class='reason'>{_esc(ev.get('reason'))}</div>" if ev.get("reason") else ""
        return (f"<div class='msg valcard'><span class='role'>Validator · checklist</span>"
                f"<div class='dec'><span class='{cls}'>{_esc(outcome)}</span> · {cs}</div>{reason}</div>")
    if t == "format_retry":
        return (f"<div class='msg valcard'><span class='role'>Format retry</span>"
                f"<div class='dec'><span class='warn'>parse failed</span> · {_esc(ev.get('error'))}</div></div>")
    # early_exit / anything else
    return (f"<div class='msg valcard'><span class='role'>{_esc(t)}</span>"
            f"<div class='dec'>{_esc(ev.get('reason') or ev.get('error') or '')}</div></div>")


def _render_isolation(ev):
    """One injection-isolator pass: what it detected and whether it stripped it."""
    ok = ev.get("success")
    det = ev.get("detected") or []
    det_html = "\n".join(f'<span class="strike">{_esc(x)}</span>' for x in det)
    status = f"<span class='ok'>stripped ✓</span>" if ok else "<span class='bad'>detected, not removed ✗</span>"
    return (f"<div class='msg isocard'><span class='role'>Isolator</span>"
            f"<code class='isobox'>{det_html}\n{ev.get('before_len')} → {ev.get('after_len')} chars · {status}</code></div>")


def render_trace_html(d: dict) -> str:
    P = []
    suite = d.get("suite_name", "?"); utask = d.get("user_task_id", "?")
    itask = d.get("injection_task_id"); attack = d.get("attack_type") or "none"
    model = d.get("pipeline_name", "?")
    util_s = "✓" if d.get("utility") else "✗"
    sec_s = "✓" if d.get("security") else "✗"

    P.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
    P.append(f"<title>Trace — {_esc(suite)} / {_esc(utask)}{'' if itask is None else ' / ' + _esc(itask)}</title>")
    P.append(f"<style>{_CSS}</style></head><body><div class='figure'>")
    P.append(f"<div class='title'>Execution Trace <span>— suite: {_esc(suite)} · {_esc(utask)}"
             f" · attack: {_esc(attack)} · {_esc(model)} · utility {util_s} · security {sec_s}</span></div>")
    sub = []
    if itask is not None:
        sub.append(f"injection: {_esc(itask)}")
    if d.get("duration") is not None:
        sub.append(f"{d['duration']:.1f} s")
    if d.get("total_tokens") is not None:
        sub.append(f"{d['total_tokens']} tokens")
    P.append(f"<div class='subnote'>{' · '.join(sub)}</div>")

    # --- Config card: flags + INITIAL plan + permissions (decided up front) ---
    flags = [f"{lab} {'✓' if d.get(k) else '✗'}"
             for k, lab in [("build_constraints", "build_constraints"),
                            ("injection_isolation", "injection_isolation"),
                            ("dynamic_validation", "dynamic_validation"),
                            ("adaptive_attack", "adaptive_attack"), ("align_claim", "align_claim"),
                            ("close_tag", "close_tag"), ("repeated_instruction", "repeated_instruction")]
             if k in d]
    P.append("<div class='msg cfgcard'><span class='role'>Config</span>")
    P.append(f"<div>{' · '.join(flags)}</div>")
    if d.get("initial_trajectory") is not None:
        P.append("<div class='box'><div class='box-label'>INITIAL TRAJECTORY (DRIFT plan)</div>")
        P.append(f"<code class='cfgbox'>{_esc(d.get('initial_trajectory'))}</code></div>")
    init_cl = _pretty_checklist(d.get("initial_checklist"))
    if init_cl:
        P.append("<div class='box'><div class='box-label'>INITIAL CHECKLIST</div>")
        P.append(f"<code class='cfgbox'>{_esc(init_cl)}</code></div>")
    perms = d.get("tool_permission") or {}
    if perms:
        P.append("<div class='box'><div class='box-label'>TOOL PERMISSIONS</div><div class='perm-grid'>")
        for name, perm in perms.items():
            P.append(f"<div>{_perm_tag(perm)}{_esc(name)}</div>")
        P.append("</div></div>")
    P.append("</div>")

    # --- Interleaved timeline ---
    events = list(d.get("events") or [])
    isos = list(d.get("isolation_events") or [])
    ev_i = [0]
    iso_i = [0]

    def drain_validator(call_names):
        """Emit the validator events for one assistant turn: leading format_retry, then
        checklist decisions, then trajectory decisions for this turn's calls."""
        while ev_i[0] < len(events):
            ev = events[ev_i[0]]
            t = ev.get("type")
            if t == "format_retry" or t == "checklist_decision":
                P.append(_render_decision(ev)); ev_i[0] += 1; continue
            if t == "trajectory_decision" and (not call_names or ev.get("function") in call_names):
                P.append(_render_decision(ev)); ev_i[0] += 1; continue
            break

    def drain_isolation(tool_len):
        """Emit isolation passes for one tool result: drain up to the pass whose after_len
        matches the shown (post-isolation) content length; else emit a single pass."""
        emitted = 0
        while iso_i[0] < len(isos):
            ev = isos[iso_i[0]]
            P.append(_render_isolation(ev)); iso_i[0] += 1; emitted += 1
            if ev.get("after_len") == tool_len:
                break
            if emitted >= 1 and ev.get("after_len") != tool_len:
                # avoid over-draining when nothing matches this result's length
                if iso_i[0] < len(isos) and isos[iso_i[0]].get("before_len") != ev.get("after_len"):
                    break
        return emitted

    for m in (d.get("conversations") or []):
        role = m.get("role")
        text = _content_text(m.get("content"))
        if role in ("user", "human"):
            P.append(f"<div class='msg usr'><span class='role'>User</span><div>{_esc(text)}</div></div>")
        elif role in ("assistant", "gpt"):
            thought, call, final = _split_assistant(text)
            P.append("<div class='msg ast'><span class='role'>Assistant</span>")
            if thought:
                P.append(f"<code class='thought'>{_esc(thought)}</code>")
            if call:
                P.append(f"<code class='call'>{_esc(call)}</code>")
            if final:
                P.append(f"<code class='final'>{_esc(final)}</code>")
            if not (thought or call or final):
                P.append(f"<div>{_esc(text) or '<span class=nocontent>(no content)</span>'}</div>")
            P.append("</div>")
            # runtime validator decisions for the calls this turn proposed
            drain_validator(_call_names(m, call))
        elif role in ("tool", "observation"):
            P.append("<div class='msg tool'><span class='role'>Tool</span>"
                     f"<code class='result'>{_esc(text) or '(empty)'}</code></div>")
            # what the isolator did to this result
            drain_isolation(len(text))

    # any decisions / isolations not tied to a message (e.g. trailing early_exit)
    while ev_i[0] < len(events):
        P.append(_render_decision(events[ev_i[0]])); ev_i[0] += 1
    while iso_i[0] < len(isos):
        P.append(_render_isolation(isos[iso_i[0]])); iso_i[0] += 1

    if d.get("final_trajectory") is not None:
        P.append(f"<div class='foot'>final trajectory: {_esc(d.get('final_trajectory'))}</div>")
    P.append("</div></body></html>")
    return "".join(P)
