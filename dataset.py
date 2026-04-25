from datasets import load_dataset
import json, random
from pathlib import Path
from config import SAMPLE_SIZE, SPLIT_SEED, N_PROFILE, N_VALIDATION, N_TEST, DATA_DIR

DATASET_SIZE = N_PROFILE + N_VALIDATION + N_TEST


def build_splits() -> dict:
    """Download dataset, build splits, save to data/splits.json."""
    print(f"Loading {SAMPLE_SIZE} problems from LiveCodeBench...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        trust_remote_code=True,
        streaming=True,
    )
    problems = list(ds.take(SAMPLE_SIZE))

    random.seed(SPLIT_SEED)
    random.shuffle(problems)

    splits = {
        "profile":    problems[:N_PROFILE],
        "validation": problems[N_PROFILE : N_PROFILE + N_VALIDATION],
        "test":       problems[N_PROFILE + N_VALIDATION : N_PROFILE + N_VALIDATION + N_TEST],
    }

    Path(DATA_DIR).mkdir(exist_ok=True)
    json.dump(splits, open(f"{DATA_DIR}/splits.json", "w"), default=str)

    print(f"Splits saved: {N_PROFILE} profile / "
          f"{N_VALIDATION} validation / {N_TEST} test")
    return splits


def load_splits() -> dict:
    """Load splits from disk. Run dataset.py first if file doesn't exist."""
    path = Path(DATA_DIR) / "splits.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python dataset.py` first"
        )
    return json.load(open(path))


def get_test_cases(problem: dict) -> list[dict]:
    """Extract test cases from problem dict."""
    return json.loads(problem["public_test_cases"])


def get_func_name(problem: dict) -> str:
    """Extract expected function name from problem metadata, defaulting to 'solution'."""
    try:
        return json.loads(problem["metadata"]).get("func_name", "solution")
    except Exception:
        return "solution"


if __name__ == "__main__":
    splits = build_splits()
    p = splits["profile"][0]
    print(f"Sample problem: {p.get('question_title')}")
    print(f"Keys: {list(p.keys())}")