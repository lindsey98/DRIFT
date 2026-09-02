"""``ASBSecurityPolicyEngine``: CaMeL security policy for ASB agents.

ASB tools are parameterless, so CaMeL's usual lever — blocking untrusted data from flowing into a
side-effecting tool *argument* — does not directly apply. Instead we use CaMeL's default-deny
behaviour as an allowlist: the agent's own benign (task) tools are permitted, and anything else —
including the injected attacker tool, whose only justification is the untrusted OPI text in an
observation — falls through to the base engine's "No security policy matched -> Denied".

The base ``SecurityPolicyEngine.check_policy`` (src/camel/security_policy.py) already implements:
allow ``no_side_effect_tools``; deny if any dependency is non-public; else match ``policies`` in
order; default deny. We supply the allowlist and no extra policies.

``PrivilegedLLM`` instantiates the policy engine as ``security_policy_engine(env)`` (env only), so
``make_asb_policy_engine_cls`` bakes the per-agent benign allowlist into a class it can construct.
"""

from collections.abc import Iterable

from src.camel.pipeline_elements.security_policies.agentdojo_security_policies import (
    AgentDojoSecurityPolicyEngine,
)


def make_asb_policy_engine_cls(benign_tool_names: Iterable[str]) -> type:
    """Return a ``SecurityPolicyEngine`` subclass (constructed with ``env`` by PrivilegedLLM) whose
    allowlist is the agent's benign tools. The attacker tool (not benign) hits the base default-deny.
    """
    allowlist = set(benign_tool_names)

    class ASBSecurityPolicyEngine(AgentDojoSecurityPolicyEngine):
        def __init__(self, env) -> None:
            super().__init__(env)
            self.no_side_effect_tools = set(allowlist)
            self.policies = []
            # check_policy is inherited from the base SecurityPolicyEngine (default-deny).

    return ASBSecurityPolicyEngine
