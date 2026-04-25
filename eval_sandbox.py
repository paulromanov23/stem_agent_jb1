import subprocess, tempfile, os, json, re, sys
from dataclasses import dataclass, field
from typing import Callable
from dataset import get_test_cases, get_func_name

# Data helpers

def extract_code(text: str) -> str:
    """Strip markdown fences and any example usage the model appended."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    code = "\n\n".join(blocks).strip() if blocks else text.strip()

    # Strip everything from common example/test usage markers onward
    cutoff_markers = [
        "# Example", "# Test", "# Driver", "# Main",
        "# Usage", "if __name__",
    ]
    for marker in cutoff_markers:
        idx = code.find(marker)
        if idx != -1:
            code = code[:idx].strip()

    return code


# Runner

# Choose the same Python interpreter as the environment running this code
PYTHON = sys.executable   

IMPORTS = (
    "import sys, math, heapq, itertools, functools\n"
    "from typing import List, Optional, Tuple, Dict, Set\n"
    "from collections import defaultdict, deque, Counter\n\n"
)


def _build_runner(code: str, test_case: dict, func_name: str) -> str:
    """
    Build a self-contained Python script that runs one test case and prints
    the result. Handles both LeetCode-style (functional) and stdin/stdout.
    """
    test_type = test_case.get("testtype", "functional")

    if test_type == "functional":
        # Multi-line inputs are newline-separated args — join into one line
        raw_input = test_case["input"].strip()
        inline_input = ", ".join(line.strip() for line in raw_input.splitlines())

        # Detect whether the code defines a Solution class or a bare function
        has_class = bool(re.search(r"^class Solution", code, re.MULTILINE))

        if has_class:
            call = f"Solution().{func_name}({inline_input})"
        else:
            # Bare function — try func_name first, fall back to first defined func
            if not re.search(rf"^def {func_name}\b", code, re.MULTILINE):
                all_funcs = re.findall(r"^def (\w+)\s*\(", code, re.MULTILINE)
                actual_name = all_funcs[0] if all_funcs else func_name
            else:
                actual_name = func_name
            call = f"{actual_name}({inline_input})"

        return f"{IMPORTS}{code}\n\nresult = {call}\nprint(result)\n"

    else:
        # stdin/stdout style — code reads input itself
        return f"{IMPORTS}{code}\n"


def run_test_case(code: str, test_case: dict, func_name: str, timeout: int = 10) -> dict:
    """Run a single test case. Returns a result dict."""
    runner = _build_runner(code, test_case, func_name)
    stdin = test_case.get("input", "") if test_case.get("testtype") != "functional" else ""
    expected = test_case["output"].strip()
    fname = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(runner)
            fname = f.name

        proc = subprocess.run(
            [PYTHON, fname],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        actual = proc.stdout.strip()
        passed = actual == expected

        return {
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "stderr": proc.stderr.strip(),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False, "expected": expected,
            "actual": "", "stderr": "TLE", "timed_out": True,
        }
    except Exception as e:
        return {
            "passed": False, "expected": expected,
            "actual": "", "stderr": str(e), "timed_out": False,
        }
    finally:
        if fname and os.path.exists(fname):
            os.unlink(fname)


# Evaluation sandbox

@dataclass
class EvalResult:
    problem_title: str
    passed: bool
    pass_rate: float
    n_tests: int
    errors: list[dict] = field(default_factory=list)
    generated_code: str = ""

    def __repr__(self):
        status = "✓" if self.passed else "✗"
        return f"{status} {self.problem_title} ({self.pass_rate:.0%} — {self.n_tests} tests)"


class EvaluationSandbox:
    """
    Takes an agent callable, runs it on a list of LiveCodeBench problems,
    and returns pass@1.

    Usage:
        sandbox = EvaluationSandbox(agent_fn=baseline_solve, verbose=True)
        results = sandbox.run(splits["validation"])
        print(f"pass@1: {sandbox.pass_at_1(results):.3f}")
    """

    def __init__(
        self,
        agent_fn: Callable[[dict], str],
        timeout: int = 10,
        verbose: bool = True,
    ):
        self.agent_fn = agent_fn
        self.timeout = timeout
        self.verbose = verbose

    def _evaluate_one(self, p: dict) -> EvalResult:
        title = p.get("question_title", "unknown")
        func_name = get_func_name(p)
        test_cases = get_test_cases(p)

        try:
            raw = self.agent_fn(p)
            code = extract_code(raw)
        except Exception as e:
            return EvalResult(
                problem_title=title, passed=False, pass_rate=0.0,
                n_tests=len(test_cases),
                errors=[{"error": f"Agent failed: {e}"}],
                generated_code="",
            )

        tc_results = [
            run_test_case(code, tc, func_name, timeout=self.timeout)
            for tc in test_cases
        ]

        n_passed = sum(r["passed"] for r in tc_results)
        errors = [r for r in tc_results if not r["passed"]]

        return EvalResult(
            problem_title=title,
            passed=(n_passed == len(test_cases)),
            pass_rate=n_passed / len(test_cases) if test_cases else 0.0,
            n_tests=len(test_cases),
            errors=errors,
            generated_code=code,
        )

    def run(self, problems: list[dict]) -> list[EvalResult]:
        results = []
        for i, p in enumerate(problems, 1):
            result = self._evaluate_one(p)
            results.append(result)
            if self.verbose:
                print(f"[{i}/{len(problems)}] {result}")
        return results

    @staticmethod
    def pass_at_1(results: list[EvalResult]) -> float:
        """Calculate pass@1 from a list of EvalResult."""
        if not results:
            return 0.0
        return sum(r.passed for r in results) / len(results)

    @staticmethod
    def summary(results: list[EvalResult]) -> dict:
        """Return a summary dict with pass@1, number of problems, average pass rate, and failure list."""
        return {
            "pass@1": EvaluationSandbox.pass_at_1(results),
            "n_problems": len(results),
            "n_passed": sum(r.passed for r in results),
            "avg_pass_rate": sum(r.pass_rate for r in results) / len(results) if results else 0,
            "failures": [r.problem_title for r in results if not r.passed],
        }
        