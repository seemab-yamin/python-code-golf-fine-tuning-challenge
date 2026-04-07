"""
Parse single-lambda submissions and run examples under a restricted namespace with timeout.
"""
from __future__ import annotations

import ast
import json
import multiprocessing as mp
from typing import Any, Optional

from golf_ft.constraints import build_grader_namespace, scan_disallowed_import_calls


class MiniGraderError(Exception):
    pass


def strict_equal(got: Any, expected: Any) -> bool:
    """Type-sensitive equality (True != 1; int != float)."""
    if type(got) is not type(expected):
        return False
    if isinstance(got, list):
        return len(got) == len(expected) and all(
            strict_equal(x, y) for x, y in zip(got, expected)
        )
    if isinstance(got, dict):
        if set(got) != set(expected):
            return False
        return all(strict_equal(got[k], expected[k]) for k in got)
    if isinstance(got, tuple):
        return len(got) == len(expected) and all(
            strict_equal(x, y) for x, y in zip(got, expected)
        )
    if isinstance(got, set):
        if len(got) != len(expected):
            return False
        # set members compared strictly — approximate by sorting a stable repr; sets of unhashable fail
        try:
            return got == expected
        except TypeError:
            return False
    return got == expected


def is_single_lambda_source(source: str) -> tuple[bool, Optional[str]]:
    """True if source parses as eval mode and top-level node is Lambda."""
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as e:
        return False, str(e)
    body = tree.body
    if not isinstance(body, ast.Lambda):
        return False, "top-level expression must be a lambda"
    return True, None


def _strip_code_fences(raw: str) -> str:
    s = raw.strip()
    for fence in ("```python", "```py", "```"):
        if fence in s:
            parts = s.split(fence)
            for p in parts:
                t = p.strip()
                if "lambda" in t.lower():
                    s = t
                    break
    return s.strip()


def _longest_lambda_via_ast(s: str) -> str | None:
    """Find leftmost 'lambda' and longest suffix that parses as a single Lambda."""
    lower = s.lower()
    i = 0
    while True:
        j = lower.find("lambda", i)
        if j < 0:
            return None
        before = s[j - 1] if j > 0 else " "
        if j > 0 and (before.isalnum() or before == "_"):
            i = j + 6
            continue
        for end in range(len(s), j + 7, -1):
            chunk = s[j:end].strip()
            if not chunk:
                continue
            try:
                tree = ast.parse(chunk, mode="eval")
            except SyntaxError:
                continue
            if isinstance(tree.body, ast.Lambda):
                return chunk
        i = j + 6


def extract_lambda_string(raw: str) -> str:
    """Strip fences; extract a single lambda via AST from leftmost lambda keyword."""
    s = _strip_code_fences(raw)
    got = _longest_lambda_via_ast(s)
    if got is not None:
        return got
    idx = s.find("lambda")
    if idx >= 0:
        return s[idx:].split("\n", 1)[0].strip()
    return s


def _eval_and_run(
    lambda_source: str,
    examples: list[dict[str, Any]],
) -> tuple[bool, Optional[str]]:
    ns = build_grader_namespace()
    try:
        fn = eval(compile(lambda_source.strip(), "<submission>", "eval"), ns, {})
    except Exception as e:
        return False, f"eval: {e}"
    for ex in examples:
        inp = ex["input"]
        exp_out = ex["output"]
        try:
            if not isinstance(inp, list):
                got = fn(inp)
            else:
                got = fn(*inp)
        except Exception as e:
            return False, f"runtime: {e}"
        if not strict_equal(got, exp_out):
            return False, f"want {exp_out!r} got {got!r}"
    return True, None


def _worker_run(
    lambda_source: str,
    examples_json: str,
    out_q: mp.Queue,
) -> None:
    examples = json.loads(examples_json)
    ok, err = _eval_and_run(lambda_source, examples)
    out_q.put((ok, err))


def run_examples_subprocess(
    lambda_source: str,
    examples: list[dict[str, Any]],
    timeout_s: float = 5.0,
) -> tuple[bool, Optional[str]]:
    """Run in child process to hard-cap wall time."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(
        target=_worker_run,
        args=(lambda_source, json.dumps(examples), q),
    )
    p.start()
    p.join(timeout=timeout_s + 0.5)
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        return False, "timeout"
    if q.empty():
        return False, "no result from worker"
    ok, err = q.get_nowait()
    return ok, err


def validate_submission_row(
    code: str,
    examples: list[dict[str, Any]],
    use_subprocess: bool = True,
    timeout_s: float = 5.0,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    ok, err = is_single_lambda_source(code)
    if not ok:
        issues.append(err or "not a single lambda")
        return False, issues
    issues.extend(scan_disallowed_import_calls(code))
    if issues:
        return False, issues
    if use_subprocess:
        run_ok, run_err = run_examples_subprocess(code, examples, timeout_s=timeout_s)
    else:
        run_ok, run_err = _eval_and_run(code, examples)
    if not run_ok:
        issues.append(run_err or "run failed")
    return run_ok, issues
