"""Stage defaults; pipeline manifest overrides runtime settings."""

MODEL_NAMES = {
    'gpt5.4': 'gpt-5.4',
    'gpt4o': 'gpt-4o-2024-08-06',
    'gpt4omini': 'gpt-4o-mini-2024-07-18',
    'o3': 'o3-2025-04-16-medium',
    'o3mini': 'o3-mini-medium',
    'claude3.5': 'claude-3-5-sonnet-20241022',
    'claude4.0': 'claude-sonnet-4-20250514-thinking',
    'dsv3': 'deepseek-v3-250324',
    'dsr1': 'deepseek-r1-250528',
    'gemini2.0': 'gemini-2.0-pro-exp-02-05',
    'gemini2.5': 'gemini-2.5-pro-exp-03-25',
    'grok3': 'grok-3',
    'grok2': 'grok-2-1212'
}

BASE_CONFIG = {
    'version': "formation_and_navigation",  # Retained output naming label.
    'model_name': "gpt5.4",   # base model name (use abbreviation from MODEL_NAMES)
    'assistant_model': "gpt4omini",  # assistant model name (use abbreviation from MODEL_NAMES)
    'round': 26,            # number of rounds
    'n_per_cohort': 4,      # number of participants per cohort
    'api_key_set': 1,       # which set of API keys to use (1-5)
    'openai_base_url': 'https://api.openai.com/v1',
    # Weighted navigation prompt settings
    'weights_csv_path': 'gpt5.4_obs_learning_cf_bottom_crossentropy.csv',  # Path to optimized weights CSV with prompt_text column
    'use_weighted_prompts': True,  # Whether to use weighted navigation prompts (set False to use default prompt)
}

CONDITION_PARAMS = {
    "Q00":  {"p": 0.5,  "q": 0},
    "Q05":  {"p": 0.5,  "q": 0.05},
    "Q50":  {"p": 0.5,  "q": 0.5},
}

from simulation_counts import DEFAULT_GROUP_COUNTS

EXPERIMENT_CONFIG = { # debo_condition1.py
    **BASE_CONFIG,  # Include all settings from BASE_CONFIG
    'condition': "Q50",      # experiment condition (Q00, Q05, Q50, Q'00, Q'05, Q'50)
    'n_total_cohort': DEFAULT_GROUP_COUNTS['Q50'],     # total number of cohorts across all parallel sessions
    'n_parallel_sessions': 2,  # number of parallel simulation notebooks to run
    'round': 26,             # number of rounds (can override BASE_CONFIG)
}

import os
from env_loader import load_environment

# Load LLM_Simulation/.env before capturing keys, regardless of working directory.
load_environment()

api_key_sets = {i: [os.environ.get(f"OPENAI_API_KEY_{i}") or os.environ.get("OPENAI_API_KEY", "")] for i in range(1, 6)}

def get_api_keys(config):
    return api_key_sets[config.get("api_key_set", 1)]

from experiment_runtime import configure

configure(globals())
