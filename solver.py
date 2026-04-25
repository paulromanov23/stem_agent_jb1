from models import SolverGenome
from eval_sandbox import extract_code
from llm import call
from config import MAX_TOKENS_SOLVE

def genome_to_solver(genome: SolverGenome):
    """Returns a callable that solves a problem using the genome."""
    def solver(problem: dict) -> str:
        prompt = genome.to_solver_prompt(problem.get("question_content", ""))
        text = call(
            system=genome.system_prompt, 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS_SOLVE,
        )
        return extract_code(text)
    return solver

