from dataclasses import dataclass, field
import json

@dataclass
class AlgoProfile:
    dominant_categories: list[str]
    # e.g. ["dynamic_programming", "graph_bfs", "two_pointers"]
    # derived from actual tags/patterns you observe, not generic

    difficulty_distribution: dict
    # {"easy": 0.3, "medium": 0.5, "hard": 0.2}

    constraint_signals: list[str]
    # concrete rules: "n<=1e5 -> need O(nlogn) or better"
    # "grid input with BFS -> likely shortest path"
    # "substring/subsequence -> likely DP"

    pattern_signals: list[str]
    # text patterns that reveal algorithm:
    # "'minimum number of operations' -> BFS or DP"
    # "'all possible subsets' + n<=20 -> bitmask"

    common_pitfalls: list[str]
    # what a naive solver would get wrong on these problems
    # e.g. "off-by-one in sliding window", "not handling disconnected graphs"

    key_insight: str
    # the single most important strategic insight
    # should be specific to THIS problem set, not generic advice

    confidence: float  # 0.0 - 1.0

@dataclass
class SolverGenome:
    system_prompt: str
    # encodes the strategic meta-knowledge from AlgoProfile
    # this is what changes meaningfully between generations

    classification_rules: list[str]
    # explicit if/then rules derived from pattern_signals
    # e.g. "IF 'shortest path' AND weighted edges THEN Dijkstra"
    # aim for 8-12 specific rules

    strategy_map: dict
    # {"dynamic_programming": "1. identify overlapping subproblems..."}
    # step-by-step approach per algorithm family

    solving_order: list[str]
    # the meta-process: how to approach ANY problem
    # e.g. ["read_constraints", "classify", "recall_strategy", "code", "verify"]

    generation: int = 1
    pass_rate: float = 0.0

    def to_solver_prompt(self, problem_statement: str) -> str:
        """Render genome into an actual prompt for solving"""
        return f"""
Follow this solving process: {' -> '.join(self.solving_order)}

Classification rules (apply in order):
{chr(10).join(f'- {r}' for r in self.classification_rules)}

Strategy library:
{json.dumps(self.strategy_map, indent=2)}

Problem:
{problem_statement[:3000]}

Output only valid Python code. No explanation, no markdown.
"""