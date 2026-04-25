import json
from dataclasses import asdict
from models import SolverGenome
from llm import call, parse_json
from solver import genome_to_solver
from eval_sandbox import EvaluationSandbox
from config import MAX_GENERATIONS, IMPROVEMENT_THRESHOLD, PLATEAU_PATIENCE, DATA_DIR


def _get_failures(genome: SolverGenome,
                  problems: list[dict]) -> list[dict]:
    """Returns only the list of failed problem dicts. No score."""
    solver = genome_to_solver(genome)
    sandbox = EvaluationSandbox(agent_fn=solver, verbose=False)
    results = sandbox.run(problems)
    failed_titles = {r.problem_title for r in results if not r.passed}
    return [p for p in problems if p.get("question_title") in failed_titles]


def _analyze_failures(genome: SolverGenome,
                      failed_problems: list[dict]) -> dict:
    """Returns structured analysis of failure modes, missing rules, weak strategies, and root cause."""
    failures_text = "\n\n---\n\n".join([
        f"Problem: {p.get('question_title')}\n"
        f"Difficulty: {p.get('difficulty')}\n"
        f"Tags: {p.get('tags', [])}\n"
        f"Content: {p.get('question_content', '')[:800]}"
        for p in failed_problems[:6]
    ])

    text = call(messages=[{"role": "user", "content": f"""
A competitive programming solver with this genome failed on these problems:

CURRENT GENOME SYSTEM PROMPT:
{genome.system_prompt[:1000]}

CURRENT CLASSIFICATION RULES:
{json.dumps(genome.classification_rules, indent=2)}

FAILED PROBLEMS:
{failures_text}

Analyze WHY the genome failed. Be specific:
- Which classification rules are missing or wrong?
- Which strategy_map entries are incomplete?
- Are there algorithm families not covered?
- Are there constraint patterns the genome doesn't handle?

Reply as JSON:
{{
  "failure_reasons": ["specific reason 1", "specific reason 2"],
  "missing_rules": ["rule that would have caught failure X"],
  "weak_strategies": ["strategy name that needs improvement"],
  "root_cause": "single most important thing to fix"
}}"""}])
    return parse_json(text)


def _propose_genome_diff(genome: SolverGenome,
                         failure_analysis: dict) -> dict:
    """Returns a minimal diff to apply to the genome. Only fields that need changing."""

    text = call(messages=[{"role": "user", "content": f"""
You are improving a competitive programming solver genome.

FAILURE ANALYSIS:
{json.dumps(failure_analysis, indent=2)}

CURRENT GENOME:
{json.dumps(asdict(genome), indent=2)}

Propose a MINIMAL targeted diff. Rules:
1. Only change what the failure analysis says is wrong
2. Classification rules must use ALGORITHM CONCEPTS not problem-specific phrases
   BAD:  "IF 'delete at most k elements' THEN sliding window"
   GOOD: "IF asked to find longest subarray/subseqence with at most k modifications THEN sliding window"
   BAD:  "IF '?' present in string THEN combinatorial counting"  
   GOOD: "IF string has wildcard characters AND counting valid configurations THEN combinatorial DP"
3. Each new rule must apply to a FAMILY of problems, not one specific problem
4. Do not add more than 3 new classification rules per generation

Reply as JSON with only the fields you want to change:
{{
  "system_prompt": "improved prompt — keep format rules, only add strategy",
  "classification_rules": ["COMPLETE new list — keep good rules, fix/add bad ones"],
  "strategy_map": {{"only_keys_that_need_changing": "new strategy text"}},
  "rationale": "one sentence: what specifically changed and why"
}}"""
}])
    return parse_json(text)


def _apply_diff(genome: SolverGenome, diff: dict) -> SolverGenome:
    current = asdict(genome)
    if "strategy_map" in diff:
        diff["strategy_map"] = {**current["strategy_map"], **diff["strategy_map"]}
    updated = {**current, **diff}
    updated["generation"] = genome.generation + 1
    updated["pass_rate"] = 0.0
    updated.pop("rationale", None)
    return SolverGenome(**updated)


def run_committed_loop(genome: SolverGenome,
                       validation_problems: list[dict],
                       current_score: float) -> SolverGenome:
    """Runs the committed improvement loop until convergence or max generations. Returns the best genome found."""
    best_genome = genome
    best_score = current_score
    no_improvement_streak = 0
    history = []

    print(f"\n[committed] Starting. Gen {genome.generation} score: {best_score:.3f}")

    for i in range(MAX_GENERATIONS):
        print(f"\n[committed] === Generation {best_genome.generation + 1} ===")

        # 1. find failures using your sandbox
        # 1. find failures (silent)
        failed_problems = _get_failures(best_genome, validation_problems)

        if not failed_problems:
            print("[committed] Perfect score — stopping early")
            break

        # analyze failures
        analysis = _analyze_failures(best_genome, failed_problems)
        print(f"[committed] Root cause: {analysis.get('root_cause')}")

        # propose diff
        diff = _propose_genome_diff(best_genome, analysis)
        rationale = diff.get("rationale", "no rationale given")
        print(f"[committed] Proposed change: {rationale}")

        # apply diff to candidate
        candidate = _apply_diff(best_genome, diff)

        # score candidate using the sandbox

        solver = genome_to_solver(candidate)
        sandbox = EvaluationSandbox(agent_fn=solver, verbose=True)
        results = sandbox.run(validation_problems)
        candidate_score = sandbox.pass_at_1(results)
        candidate_scores = sandbox.summary(results)

        print(f"[committed] Candidate: {candidate_score:.3f} "
              f"(best: {best_score:.3f}, "
              f"delta: {candidate_score - best_score:+.3f})")

        # commit or apoptosis
        if candidate_score >= best_score + IMPROVEMENT_THRESHOLD:
            print(f"[committed] ✓ Committed gen {candidate.generation}")
            best_genome = candidate
            best_score = candidate_score
            no_improvement_streak = 0
            #log_generation(best_genome, candidate_scores)
            json.dump(asdict(best_genome),
                open(f"{DATA_DIR}/genome_v{best_genome.generation}.json", "w")
                , indent=2)
        else:
            print(f"[committed] ✗ Apoptosis — discarding")
            no_improvement_streak += 1

        history.append({
            "generation": best_genome.generation,
            "score": best_score,
            "rationale": rationale,
            "committed": candidate_score >= best_score + IMPROVEMENT_THRESHOLD,
        })

        # plateau check to avoid long runs with tiny improvements
        if no_improvement_streak >= PLATEAU_PATIENCE:
            print(f"[committed] Plateau — stopping after "
                  f"{no_improvement_streak} non-commits")
            break

    json.dump(history, open(f"{DATA_DIR}/evolution_history.json", "w"), indent=2)
    print(f"\n[committed] Done. "
          f"Final: {best_score:.3f} at gen {best_genome.generation}")
    return best_genome