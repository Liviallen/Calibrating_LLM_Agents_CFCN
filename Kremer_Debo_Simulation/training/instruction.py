"""Initial instruction shared by training and validation."""
from shared_prompts import get_instruction_prompt


def instruction_prompt(condition='Q50'):
    return get_instruction_prompt(condition)
