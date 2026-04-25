import json, random
from dataclasses import asdict
from models import AlgoProfile, SolverGenome
from llm import call, parse_json


def _phase_a(sample: list[dict]) -> dict:
    n = len(sample)
    problems_text = "\n\n---\n\n".join([
        f"Problem {i+1}:\n"
        f"Title: {p.get('question_title', '')}\n"
        f"Difficulty: {p.get('difficulty', '')}\n"
        f"Tags: {p.get('tags', [])}\n\n"
        f"{p.get('question_content', '')[:1500]}"
        for i, p in enumerate(sample)
    ])

    text = call(
        thinking=False,
        messages=[{"role": "user", "content": f"""
You are a competitive programming analyst. DO NOT solve these problems.
Only observe structure, classify, and hypothesize.

Here are {n} problems from the set:

{problems_text}

Reply as JSON:
{{
  "observed_categories": ["algorithm families — be specific"],
  "constraint_patterns": ["constraint -> what it implies"],
  "text_signals": ["phrase in problem -> algorithm type"],
  "naive_failure_modes": ["how a generic solver would fail"],
  "hypotheses_to_verify": ["things you suspect but need more data for"]
}}"""
}]
    )
    return parse_json(text)


def _phase_b(remaining: list[dict], hypotheses: dict) -> AlgoProfile:
    n = len(remaining)
    problems_text = "\n\n---\n\n".join([
        f"Problem {i+1}:\n"
        f"Title: {p.get('question_title', '')}\n"
        f"Difficulty: {p.get('difficulty', '')}\n"
        f"Tags: {p.get('tags', [])}\n\n"
        f"{p.get('question_content', '')[:1200]}"
        for i, p in enumerate(remaining)
    ])

    text = call(messages=[{"role": "user", "content": f"""
Your hypotheses from the initial sample:
{json.dumps(hypotheses, indent=2)}

Verify against {n} more problems:
{problems_text}

Produce final AlgoProfile JSON:
{{
  "dominant_categories": ["top 3-4 algorithm families by frequency"],
  "difficulty_distribution": {{"easy": 0.0, "medium": 0.0, "hard": 0.0}},
  "constraint_signals": ["n<=1e5 -> O(nlogn) max", "n<=20 -> bitmask ok"],
  "pattern_signals": ["'minimum steps' -> BFS or DP", "'all subsets' -> bitmask"],
  "common_pitfalls": ["specific mistake on THIS problem set"],
  "key_insight": "one concrete insight specific to this problem set",
  "confidence": 0.0
}}"""
    }])
    return AlgoProfile(**parse_json(text))


def run_progenitor(profile_problems: list[dict]) -> AlgoProfile:
    n_sample = max(5, len(profile_problems) // 3)  # 1/3 for phase A
    
    print(f"[progenitor] Phase A: sampling {n_sample} problems...")
    sample = random.sample(profile_problems, n_sample)
    hypotheses = _phase_a(sample)

    remaining = [p for p in profile_problems if p not in sample]
    print(f"[progenitor] Phase B: verifying against {len(remaining)} problems...")
    profile = _phase_b(remaining, hypotheses)
    
    return profile



def profile_to_genome(profile: AlgoProfile) -> SolverGenome:
    text = call(messages=[{"role": "user", "content": f"""
Design a specialized competitive programming solver genome.

Profile of the problem set:
{json.dumps(asdict(profile), indent=2)}

Produce JSON:
{{
  "system_prompt": "200+ word system prompt encoding strategic knowledge.
                    Reference specific algorithm families, constraint rules,
                    and pitfalls from the profile. Sound like a specialist.",
  "classification_rules": [
    "IF 'shortest path' AND weighted graph THEN Dijkstra",
    "IF n<=20 AND 'subset'/'combinations' THEN bitmask DP",
    "... minimum 10 rules from pattern_signals and constraint_signals ..."
  ],
  "strategy_map": {{
    "dynamic_programming": "step by step: 1. identify state 2. ...",
    "... one entry per dominant_category ...": "..."
  }},
  "solving_order": ["read_constraints", "classify", "select_strategy",
                    "handle_edge_cases", "implement", "verify"]
}}"""  
    }])

    genome_dict = parse_json(text)
    genome_dict["generation"] = 1
    genome_dict["pass_rate"] = 0.0
    return SolverGenome(**genome_dict)