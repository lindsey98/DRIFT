"""Adapter that lets the **real** CaMeL pipeline run as an ASB defense.

Instead of re-implementing CaMeL, the adapter drives the actual ``PrivilegedLLM`` element from
``src/camel`` over ASB's tools, so the defense is faithful to the AgentDojo CaMeL (same retry loop,
quarantined LLM, system-prompt generator, and untrusted output tagging). The only ASB-specific
glue is:

- ``defense.py`` : ``CaMeLDefense`` — wraps ASB tools as AgentDojo ``Function``s (benign ones carry
  the OPI injection), builds the local LLM / q-LLM, runs ``PrivilegedLLM``, and converts the
  resulting messages into ASB's trajectory for ASR/RR scoring.
- ``policy.py``  : ``make_asb_policy_engine_cls`` — a default-deny allowlist of the agent's benign
  tools, so the attacker tool is denied.

The adapter imports the CaMeL kernel (``src.camel``) from the parent repo. The ASB run script sets
``PYTHONPATH`` to include the repo root; as a fallback we also add it to ``sys.path`` here so
``import src.camel`` works regardless of how ASB was launched.
"""

import sys
from pathlib import Path

# camel_adapter/ -> asb/ -> <repo root>. Ensure the repo root is importable for `src.camel`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .defense import CaMeLDefense

__all__ = ["CaMeLDefense"]
