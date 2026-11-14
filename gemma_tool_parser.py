"""Translate Gemma text tool calls into DRIFT's existing scaffold."""
import ast
import re


def normalize_gemma_calls(text):
    if not isinstance(text, str) or "<function_call>" in text:
        return text
    pattern = re.compile(r"<\|tool_call>\s*call:(\w+)\s*(.*?)<tool_call\|>", re.S)
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    calls = []
    for match in matches:
        raw = match.group(2).strip()
        # Gemma quotes strings with this token in some chat templates.
        raw = raw.replace('<|"|>', '"')
        tree = ast.parse(raw, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                node.keys = [ast.Constant(k.id) if isinstance(k, ast.Name) else k for k in node.keys]
        class Literals(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in ("true", "false", "null"):
                    return ast.copy_location(ast.Constant({"true": True, "false": False, "null": None}[node.id]), node)
                return node
        args = ast.literal_eval(Literals().visit(tree))
        if not isinstance(args, dict) or any(not isinstance(k, str) or not k.isidentifier() for k in args):
            raise ValueError("Invalid Gemma tool arguments")
        calls.append(match.group(1) + "(" + ", ".join(k + "=" + repr(v) for k, v in args.items()) + ")")
    remaining = pattern.sub("", text).strip()
    if not any(tag in remaining for tag in ("<function_thought>", "<final_answer>")):
        remaining = "<function_thought>" + remaining + "</function_thought>"
    return remaining + "\n<function_call>[" + ", ".join(calls) + "]</function_call>"
