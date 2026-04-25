from dotenv import load_dotenv
import os

load_dotenv()

# API
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
MODEL_NAME        = os.getenv("MODEL_NAME", "gpt-4o")
REASONING_MODEL   = os.getenv("REASONING_MODEL", "o4-mini")
TEMPERATURE       = float(os.getenv("TEMPERATURE", 0.2))
TOP_P             = float(os.getenv("TOP_P", 0.1))
MAX_TOKENS_SOLVE  = int(os.getenv("MAX_TOKENS_SOLVE", 2000))
MAX_TOKENS_GENOME = int(os.getenv("MAX_TOKENS_GENOME", 4000))

# splits
SAMPLE_SIZE       = int(os.getenv("SAMPLE_SIZE", 300))
SPLIT_SEED        = int(os.getenv("SPLIT_SEED", 42))
N_PROFILE         = int(os.getenv("N_PROFILE", 30))
N_VALIDATION      = int(os.getenv("N_VALIDATION", 50))
N_TEST            = int(os.getenv("N_TEST", 50))

# committed loop
MAX_GENERATIONS        = int(os.getenv("MAX_GENERATIONS", 5))
IMPROVEMENT_THRESHOLD  = float(os.getenv("IMPROVEMENT_THRESHOLD", 0.02))
PLATEAU_PATIENCE       = int(os.getenv("PLATEAU_PATIENCE", 2))

# paths
DATA_DIR = os.getenv("DATA_DIR", "data")