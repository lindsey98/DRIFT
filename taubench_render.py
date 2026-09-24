"""Render a DRIFT tau-bench result record (the dict run_task returns / the saved JSON) as a
single self-contained HTML page: the config + scores, the DRIFT components (initial plan,
checklist, tool permissions, isolation and alignment decisions), and the full conversation."""

import html
import json


def _esc(x) -> str:
    return html.escape(str(x))


def _pre(x) -> str:
    if not isinstance(x, str):
        x = json.dumps(x, indent=2, ensure_ascii=False, default=str)
    return f"<pre>{_esc(x)}</pre>"


def _card(title: str, body: str, cls: str = "") -> str:
    return f'<div class="card {cls}"><h3>{_esc(title)}</h3>{body}</div>'


def render_taubench_html(rec: dict) -> str:
    d = rec.get("drift", {}) or {}
    util, sec = rec.get("utility"), rec.get("security")
    badge = lambda ok, t, f: f'<span class="pill {"ok" if ok else "bad"}">{t if ok else f}</span>'

    head = (
        f'<div class="pill">{_esc(rec.get("model"))}</div>'
        f'<div class="pill">defense: {_esc(rec.get("defense"))}</div>'
        f'<div class="pill">attack: {_esc(rec.get("attack"))}</div>'
        f'{badge(util, "utility ✓", "utility ✗") if util is not None else ""}'
        f'{badge(not sec, "secure ✓", "LEAKED ✗") if sec is not None else ""}'
    )

    cfg = ""
    if rec.get("query"):
        cfg += _card("User query (fixed t=0)", _pre(rec["query"]))
    if d.get("initial_trajectory"):
        cfg += _card("Initial function-trajectory plan", _pre(d["initial_trajectory"]))
    if d.get("initial_checklist") and d["initial_checklist"] != "None":
        cfg += _card("Initial parameter checklist", _pre(d["initial_checklist"]))
    if d.get("tool_permissions"):
        cfg += _card("Tool permissions", _pre(d["tool_permissions"]))
    if rec.get("security_results"):
        cfg += _card("Security (DoomArena leak record)", _pre(rec["security_results"]), "attack")
    if rec.get("error"):
        cfg += _card("Error", _pre(rec.get("traceback") or rec["error"]), "attack")

    decisions = ""
    for ev in d.get("isolation_events", []):
        ok = ev.get("success")
        decisions += _card(f'Injection isolation — {"stripped" if ok else "not stripped"}',
                           _pre(ev), "iso" + ("" if ok else " attack"))
    for ev in d.get("alignment_decisions", []):
        ok = ev.get("verdict") == "Yes"
        decisions += _card(f'Trajectory alignment — {ev.get("verdict")}',
                           _pre(ev), "align" + ("" if ok else " attack"))
    if d.get("detected_injections"):
        decisions += _card("Detected injected instructions", _pre(d["detected_injections"]), "attack")

    convo = ""
    for m in rec.get("messages", []):
        role = m.get("role", "?")
        body = _pre(m.get("content", ""))
        if m.get("tool_calls"):
            body += _pre(m["tool_calls"])
        name = f' · {_esc(m["name"])}' if m.get("name") else ""
        convo += f'<div class="msg {role}"><div class="role">{_esc(role)}{name}</div>{body}</div>'

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>DRIFT · tau-bench · {_esc(rec.get('user_task'))}</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}}
.wrap{{max-width:1000px;margin:0 auto;padding:24px 16px}}
h1{{font-size:20px}} h3{{font-size:13px;margin:0 0 8px;color:#9aa4b2;text-transform:uppercase;letter-spacing:.04em}}
.head{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 20px}}
.pill{{background:#1b2130;border:1px solid #2a3242;border-radius:999px;padding:4px 12px;font-size:13px}}
.pill.ok{{background:#10331f;border-color:#1c6b3a;color:#7ee2a8}}
.pill.bad{{background:#3a1417;border-color:#7a2029;color:#ff9aa2}}
.card{{background:#151a24;border:1px solid #232b3a;border-radius:10px;padding:14px 16px;margin:10px 0}}
.card.attack{{border-color:#7a2029}} .card.iso{{border-color:#3a5a8a}} .card.align{{border-color:#6a5a2a}}
pre{{white-space:pre-wrap;word-break:break-word;margin:0;font-size:12.5px;line-height:1.5;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#cdd6e4}}
.msg{{border-radius:10px;padding:10px 14px;margin:10px 0;border:1px solid #232b3a}}
.msg .role{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8b95a5;margin-bottom:6px}}
.msg.user{{background:#12202e;border-color:#1e3a52}}
.msg.assistant{{background:#161b26}}
.msg.tool{{background:#1a1622;border-color:#3a2a4a}}
</style></head><body><div class="wrap">
<h1>DRIFT · tau-bench · {_esc(rec.get('suite'))}/{_esc(rec.get('user_task'))}</h1>
<div class="head">{head}</div>
<h3 style="margin-top:24px">Config &amp; components</h3>{cfg}
{f'<h3 style="margin-top:24px">Runtime decisions</h3>{decisions}' if decisions else ''}
<h3 style="margin-top:24px">Conversation</h3>{convo}
</div></body></html>"""
