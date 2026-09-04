"""Render a DRIFT result record (the dict dumped to <task>.json) as a standalone HTML
trace, saved next to the JSON. Shows the run config, the DRIFT components (planned/final
trajectory + checklist, tool permissions, alignment decisions, isolation events, the
structured decision stream) and the full conversation (thoughts / calls / results).

Purely presentational and defensive: any missing/oddly-shaped field is skipped, never
raises, so a render failure can't break the eval loop.
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
  code, .mono { font-family:"SF Mono", Menlo, Consolas, monospace; font-size:11.5px; }
  .call { background:#dcfce7; border:1px solid #86efac; border-radius:6px; padding:6px 10px; margin-top:6px; display:block; white-space:pre-wrap; word-break:break-word; color:#14532d; }
  .thought { background:#f0fdf4; border:1px dashed #86efac; border-radius:6px; padding:6px 10px; margin-top:6px; display:block; color:#14532d; white-space:pre-wrap; word-break:break-word; }
  .result { background:#fef3c7; border:1px solid #fcd34d; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; white-space:pre-wrap; word-break:break-word; color:#78350f; }
  .final { background:#ecfdf5; border:1px solid #6ee7b7; border-radius:6px; padding:6px 10px; margin-top:6px; display:block; white-space:pre-wrap; word-break:break-word; color:#065f46; }
  .cfgbox { background:#ede9fe; border:1px solid #c4b5fd; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; white-space:pre-wrap; word-break:break-word; color:#4c1d95; }
  .isobox { background:#f3e8ff; border:1px solid #d8b4fe; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; color:#581c87; white-space:pre-wrap; word-break:break-word; }
  .strike { text-decoration:line-through; color:#b91c1c; }
  .box { margin-top:8px; border-top:1px dashed #c4b5fd; padding-top:8px; }
  .box-label { font-size:10.5px; font-weight:700; color:var(--cfg); letter-spacing:0.06em; margin-bottom:4px; }
  .perm-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:2px 18px; margin-top:4px; }
  .perm-grid div { font-size:11px; color:#334155; }
  .tag { display:inline-block; font-size:9.5px; font-weight:700; color:#fff; border-radius:3px; padding:0 4px; margin-right:5px; min-width:20px; text-align:center; }
  .r { background:#94a3b8; } .w { background:#d97706; } .x { background:#b91c1c; }
  .verdict-yes { color:#15803d; font-weight:700; } .verdict-no { color:#b91c1c; font-weight:700; }
  .ev { font-size:11px; margin:3px 0; color:#334155; }
  .ev b { color:#4c1d95; }
  .ok { color:#15803d; font-weight:700; } .bad { color:#b91c1c; font-weight:700; }
  .pill { display:inline-block; font-size:10px; font-weight:700; border-radius:10px; padding:1px 8px; }
  .pill-ok { background:#dcfce7; color:#15803d; } .pill-bad { background:#fee2e2; color:#b91c1c; }
  .nocontent { color:#6b7280; font-style:italic; font-size:11.5px; }
"""


def _esc(s):
    return html.escape("" if s is None else str(s))


def _content_text(content):
    """conversations content is a str (agentdojo DRIFT) or a list of {type,content} blocks."""
    if isinstance(content, list):
        return "\n".join(str(b.get("content", "")) for b in content if isinstance(b, dict))
    return "" if content is None else str(content)


def _split_assistant(text):
    def grab(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
        return m.group(1).strip() if m else ""
    return grab("function_thought"), grab("function_call"), grab("final_answer")


def _perm_tag(perm):
    cls = {"Read": "r", "Write": "w", "Execute": "x"}.get(perm, "r")
    ab = {"Read": "R", "Write": "W", "Execute": "X"}.get(perm, "?")
    return f'<span class="tag {cls}">{ab}</span>'


def _pretty_checklist(raw):
    """Pretty-print the checklist (a JSON string) if possible; else return it raw."""
    if not raw or raw == "None":
        return ""
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except Exception:
        return str(raw).strip()


def render_trace_html(d: dict) -> str:
    parts = []
    suite = d.get("suite_name", "?")
    utask = d.get("user_task_id", "?")
    itask = d.get("injection_task_id")
    attack = d.get("attack_type") or "none"
    model = d.get("pipeline_name", "?")
    util = d.get("utility")
    sec = d.get("security")
    util_s = "✓" if util else "✗"
    sec_s = "✓" if sec else "✗"

    parts.append(f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
    parts.append(f"<title>Trace — {_esc(suite)} / {_esc(utask)}{'' if itask is None else ' / ' + _esc(itask)}</title>")
    parts.append(f"<style>{_CSS}</style></head><body><div class='figure'>")

    parts.append(
        f"<div class='title'>Execution Trace <span>— suite: {_esc(suite)} · {_esc(utask)}"
        f" · attack: {_esc(attack)} · {_esc(model)} · utility {util_s} · security {sec_s}</span></div>")
    dur = d.get("duration")
    tok = d.get("total_tokens")
    sub = []
    if itask is not None:
        sub.append(f"injection: {_esc(itask)}")
    if dur is not None:
        sub.append(f"{dur:.1f} s")
    if tok is not None:
        sub.append(f"{tok} tokens")
    parts.append(f"<div class='subnote'>{' · '.join(sub)}</div>")

    # --- Config + plan card ---
    flags = []
    for k, lab in [("build_constraints", "build_constraints"), ("injection_isolation", "injection_isolation"),
                   ("dynamic_validation", "dynamic_validation"), ("adaptive_attack", "adaptive_attack"),
                   ("align_claim", "align_claim"), ("close_tag", "close_tag"),
                   ("repeated_instruction", "repeated_instruction")]:
        if k in d:
            flags.append(f"{lab} {'✓' if d.get(k) else '✗'}")
    parts.append("<div class='msg cfgcard'><span class='role'>Config</span>")
    parts.append(f"<div>{' · '.join(flags)}</div>")
    init_traj = d.get("initial_trajectory")
    final_traj = d.get("final_trajectory")
    if init_traj is not None or final_traj is not None:
        parts.append("<div class='box'><div class='box-label'>TRAJECTORY</div>")
        parts.append("<code class='cfgbox'>")
        parts.append(f"initial: {_esc(init_traj)}\nfinal:   {_esc(final_traj)}")
        parts.append("</code></div>")
    init_cl = _pretty_checklist(d.get("initial_checklist"))
    if init_cl:
        parts.append("<div class='box'><div class='box-label'>INITIAL CHECKLIST</div>")
        parts.append(f"<code class='cfgbox'>{_esc(init_cl)}</code></div>")
    perms = d.get("tool_permission") or {}
    if perms:
        parts.append("<div class='box'><div class='box-label'>TOOL PERMISSIONS</div><div class='perm-grid'>")
        for name, perm in perms.items():
            parts.append(f"<div>{_perm_tag(perm)}{_esc(name)}</div>")
        parts.append("</div></div>")
    parts.append("</div>")

    # --- Alignment decisions ---
    aligns = d.get("alignment_decisions") or []
    if aligns:
        parts.append("<div class='msg cfgcard'><span class='role'>Alignment</span>")
        for a in aligns:
            v = a.get("verdict", "")
            vc = "verdict-yes" if str(v).lower().startswith("y") else "verdict-no"
            ext = a.get("extended_trajectory") or a.get("current_trajectory")
            parts.append(
                f"<div class='box'><code class='cfgbox'>extend → {_esc(ext)}\n"
                f"verdict: <span class='{vc}'>{_esc(v)}</span>\nreason: {_esc(a.get('reason',''))}</code></div>")
        parts.append("</div>")

    # --- Isolation events ---
    isos = d.get("isolation_events") or []
    dets = d.get("detected_injections") or []
    if isos or dets:
        parts.append("<div class='msg cfgcard'><span class='role'>Isolation</span>")
        for ev in isos:
            ok = ev.get("success")
            det_list = ev.get("detected") or []
            det_html = "\n".join(f'<span class="strike">{_esc(x)}</span>' for x in det_list)
            parts.append(
                f"<div class='box'><code class='isobox'>{det_html}\n"
                f"strip: {ev.get('before_len')} → {ev.get('after_len')} chars "
                f"<span class='{'ok' if ok else 'bad'}'>{'success ✓' if ok else 'no-op ✗'}</span></code></div>")
        if not isos and dets:
            parts.append("<div class='box'><code class='isobox'>detected (not stripped):\n"
                         + "\n".join(_esc(x) for x in dets) + "</code></div>")
        parts.append("</div>")

    # --- Decision event stream ---
    events = d.get("events") or []
    if events:
        parts.append("<div class='msg cfgcard'><span class='role'>Decisions</span>")
        for ev in events:
            t = ev.get("type", "")
            if t == "trajectory_decision":
                parts.append(f"<div class='ev'><b>{_esc(ev.get('outcome'))}</b> · {_esc(ev.get('function'))}"
                             f"{('(' + _esc(ev.get('args')) + ')') if ev.get('args') else ''}"
                             f" · perm={_esc(ev.get('permission'))}"
                             f"{(' · ' + _esc(ev.get('reason'))) if ev.get('reason') else ''}</div>")
            elif t == "checklist_decision":
                calls = ev.get("calls") or []
                cs = ", ".join(_esc(c.get("function")) for c in calls) if calls else _esc(ev.get("function"))
                parts.append(f"<div class='ev'><b>{_esc(ev.get('outcome'))}</b> · {cs}"
                             f"{(' · ' + _esc(ev.get('reason'))) if ev.get('reason') else ''}</div>")
            else:
                parts.append(f"<div class='ev'><b>{_esc(t)}</b>"
                             f"{(' · ' + _esc(ev.get('reason') or ev.get('error'))) if (ev.get('reason') or ev.get('error')) else ''}</div>")
        parts.append("</div>")

    # --- Conversation ---
    for m in (d.get("conversations") or []):
        role = m.get("role")
        text = _content_text(m.get("content"))
        if role in ("user", "human"):
            parts.append(f"<div class='msg usr'><span class='role'>User</span><div>{_esc(text)}</div></div>")
        elif role in ("assistant", "gpt"):
            thought, call, final = _split_assistant(text)
            parts.append("<div class='msg ast'><span class='role'>Assistant</span>")
            if thought:
                parts.append(f"<code class='thought'>{_esc(thought)}</code>")
            if call:
                parts.append(f"<code class='call'>{_esc(call)}</code>")
            if final:
                parts.append(f"<code class='final'>{_esc(final)}</code>")
            if not (thought or call or final):
                parts.append(f"<div>{_esc(text) or '<span class=nocontent>(no content)</span>'}</div>")
            parts.append("</div>")
        elif role in ("tool", "observation"):
            parts.append("<div class='msg tool'><span class='role'>Tool</span>"
                         f"<code class='result'>{_esc(text) or '(empty)'}</code></div>")
    parts.append("</div></body></html>")
    return "".join(parts)
