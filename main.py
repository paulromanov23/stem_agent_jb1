import json
from dataclasses import asdict
import datetime
from pathlib import Path
from config import DATA_DIR, MAX_TOKENS_SOLVE
from llm import call
from eval_sandbox import EvaluationSandbox, extract_code
from dataset import load_splits
from progenitor import run_progenitor, profile_to_genome
from solver import genome_to_solver
from committed import run_committed_loop

# ── load ──────────────────────────────────────────────────────────────

splits = load_splits()  # Loads the profile/validation/test splits from disk (run dataset.py first if not found)

# ── baseline (gen 0) ──────────────────────────────────────────────────

def baseline_solve(p: dict) -> str:
    return call(
        messages=[{"role": "user", "content": f"""
Solve this competitive programming problem.
Output ONLY a Python Solution class or function.
Do NOT use stdin. Do NOT add test cases or print statements.

{p['question_content'][:3000]}
"""}],
        max_tokens=MAX_TOKENS_SOLVE,  # ← from config
    )

print("=== Scoring baseline ===")
sandbox = EvaluationSandbox(agent_fn=baseline_solve, verbose=True)
results_base = sandbox.run(splits["validation"])
baseline_pass = sandbox.pass_at_1(results_base)
print(f"\nBaseline pass@1: {baseline_pass:.3f}")
print(sandbox.summary(results_base))

# ── progenitor ────────────────────────────────────────────────────────

print("\n=== Running progenitor ===")
profile = run_progenitor(splits["profile"])
json.dump(asdict(profile), 
          open(f"{DATA_DIR}/profile.json", "w"), indent=2)  # ← DATA_DIR

# ── differentiation: genome v1 ────────────────────────────────────────

print("\n=== Differentiating into genome v1 ===")
genome_v1 = profile_to_genome(profile)
json.dump(asdict(genome_v1),
          open(f"{DATA_DIR}/genome_v1.json", "w"), indent=2)  # ← DATA_DIR

# ── score gen 1 ───────────────────────────────────────────────────────

print("\n=== Scoring generation 1 ===")
solver_v1 = genome_to_solver(genome_v1)
sandbox_v1 = EvaluationSandbox(agent_fn=solver_v1, verbose=True)
results_v1 = sandbox_v1.run(splits["validation"])
pass_1_score = sandbox_v1.pass_at_1(results_v1)
print(f"\nGen 1 pass@1: {pass_1_score:.3f}")
print(sandbox_v1.summary(results_v1))

# ── committed loop ────────────────────────────────────────────────────

print("\n=== Running committed loop ===")
mature_genome = run_committed_loop(
    genome=genome_v1,
    validation_problems=splits["validation"],
    current_score=pass_1_score,
)
json.dump(asdict(mature_genome),
          open(f"{DATA_DIR}/genome_mature.json", "w"), indent=2)  

# ── validation eval of mature genome ─────────────────────────────────

mature_solver = genome_to_solver(mature_genome)
sandbox_mature_val = EvaluationSandbox(agent_fn=mature_solver, verbose=False)
results_val = sandbox_mature_val.run(splits["validation"])
mature_pass_val = sandbox_mature_val.pass_at_1(results_val)

# ── final eval on test set ────────────────────────────────────────────

print("\n=== FINAL EVAL ON TEST SET ===")
sandbox_test = EvaluationSandbox(agent_fn=mature_solver, verbose=True)
results_test = sandbox_test.run(splits["test"])
mature_pass = sandbox_test.pass_at_1(results_test)

sandbox_baseline_test = EvaluationSandbox(agent_fn=baseline_solve, verbose=False)
results_baseline_test = sandbox_baseline_test.run(splits["test"])
baseline_test_pass = sandbox_baseline_test.pass_at_1(results_baseline_test)

print(f"\n{'='*40}")
print(f"Baseline (gen 0)  : {baseline_pass:.3f}")
print(f"Gen 1 (progenitor): {pass_1_score:.3f}")
print(f"Mature genome     : {mature_pass_val:.3f}")
print(f"Total delta       : {mature_pass_val - baseline_pass:+.3f}")
print("--------------------------------")
print("Final evaluation on the test data")
print(f"Baseline test     : {baseline_test_pass:.3f}")
print(f"Mature genome test: {mature_pass:.3f}")
print(f"Total delta       : {mature_pass - baseline_test_pass:+.3f}")
print(f"{'='*40}")

# ── Log the results ────────────────────────────────────────────

results_log = {
    "baseline_val":   baseline_pass,
    "gen1_val":       pass_1_score,
    "mature_val":     mature_pass_val,
    "baseline_test":  baseline_test_pass,
    "mature_test":    mature_pass,
    "delta_test":     mature_pass - baseline_test_pass,
    "generations":    mature_genome.generation,
}

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump(results_log,
          open(f"{DATA_DIR}/results_{ts}.json", "w"), indent=2)
print(f"\nResults saved to {DATA_DIR}/results_{ts}.json")