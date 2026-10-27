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
    'version': "context_formation_and_navigation",  # Retained output naming label.
    'model_name': "gpt5.4",   # base model name (use abbreviation from MODEL_NAMES)
    'assistant_model': "gpt4omini",  # assistant model name (use abbreviation from MODEL_NAMES)
    'n_cohort': 4,          # total number of independent cohorts to simulate
    'round': 26,            # number of rounds
    'n_per_cohort': 4,      # number of participants per cohort
    'api_key_set': 1,       # which set of API keys to use (1-5)
    'openai_base_url': 'https://api.openai.com/v1'
}

REPLICATION_CONFIG_Q50 = {
    **BASE_CONFIG,
    'csv_file': 'kremer-debo_q50_uninformed.csv',
    'condition': 'Q50',
    'output_folder': 'replication_results',
    'experiment_name': 'q50_uninformed_replication',
    'checkpoint_interval': 50,
}

import os

api_key_sets = {i: [os.environ.get(f"OPENAI_API_KEY_{i}") or os.environ.get("OPENAI_API_KEY", "")] for i in range(1, 6)}

def get_api_keys(config):
    return api_key_sets[config.get("api_key_set", 1)]

from experiment_runtime import configure

configure(globals())
