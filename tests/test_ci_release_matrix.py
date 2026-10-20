"""Pages may consume only the current successful main push."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _expression(expression, github):
    source = expression.strip().removeprefix("${{").removesuffix("}}").strip()
    source = source.replace("&&", " and ").replace("||", " or ")
    tree = ast.parse("(" + source + ")", mode="eval")
    allowed = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
               ast.Compare, ast.Eq, ast.NotEq, ast.Attribute, ast.Name, ast.Load, ast.Constant)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError("unsupported deploy condition expression")
    return eval(compile(tree, "<workflow policy>", "eval"), {"__builtins__": {}}, {"github": github})


@pytest.mark.parametrize("change,allowed", [
    ({}, True), ({"conclusion": "failure"}, False), ({"conclusion": "cancelled"}, False),
    ({"event": "pull_request"}, False), ({"head_branch": "release"}, False),
    ({"head_sha": "stale"}, False),
])
def test_pages_deploys_only_the_exact_green_release_ci_main_tip(change, allowed):
    workflow = yaml.load((ROOT / ".github/workflows/deploy-pages.yml").read_text(), Loader=yaml.BaseLoader)
    ci = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_run"}
    trigger = workflow["on"]["workflow_run"]
    assert trigger["workflows"] == [ci["name"]]
    assert trigger["types"] == ["completed"]
    build_id, build = next((key, job) for key, job in workflow["jobs"].items()
                          if any(step.get("uses", "").startswith("actions/upload-pages-artifact@")
                                 for step in job.get("steps", [])))
    run = SimpleNamespace(**dict({"conclusion": "success", "event": "push", "head_branch": "main", "head_sha": "current"}, **change))
    github = SimpleNamespace(event=SimpleNamespace(workflow_run=run), sha="current")
    assert _expression(build["if"], github) is allowed
    checkout = next(step for step in build["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert _expression(checkout["with"]["ref"], github) == run.head_sha
    deploy = next(job for job in workflow["jobs"].values()
                  if any(step.get("uses", "").startswith("actions/deploy-pages@") for step in job.get("steps", [])))
    needs = deploy["needs"]
    assert build_id in ([needs] if isinstance(needs, str) else needs)
