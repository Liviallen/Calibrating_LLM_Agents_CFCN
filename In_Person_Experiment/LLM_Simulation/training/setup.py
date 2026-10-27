"""Helpers used by the Q50 training runner."""
import pickle
from shared_prompts import get_period_prompt

def save_record(data, file_path):

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_record(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def compose_full_input_prompt(instruction_prompt, period_prompt):
    """
    Combine the instruction prompt and period prompt into the full input prompt.
    """
    instruction_text = "" if instruction_prompt is None else str(instruction_prompt).rstrip()
    period_text = "" if period_prompt is None else str(period_prompt).lstrip()
    if instruction_text and period_text:
        return f"{instruction_text}\n\n{period_text}"
    return instruction_text or period_text

def new_period_prompt(cohort_id, rank_id, record_sheet, navigation_prompt=None):
    """Render the shared decision prompt with this training candidate."""
    return get_period_prompt(
        (cohort_id, rank_id), record_sheet, navigation_prompt=navigation_prompt)
