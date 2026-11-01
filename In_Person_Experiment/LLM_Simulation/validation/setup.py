from file_utils import atomic_output
import random
import pickle
import numpy as np
import pandas as pd
from shared_prompts import get_period_prompt
from simulation_config import BASE_CONFIG

def create_record_sheet(n_cohort, n=4):
    '''
    The output is a dictionary of dictionary that records all details of one experiment.
    Each cohort represents one independent experiment.
    key: (cohort_id, rank)
    
    Args:
        n_cohort: Number of cohorts to simulate
        n: Number of participants per cohort
    '''
    
    basic_record = {'position_id': None, 'subject_id': None, 'cohort_id': None, 'session_id': None, 
                    'profit': None, 'treatment': None, 'subject_rank': None, 'subject_type': None, 
                    'product_value': None, 'wait_time': None, 'queuing_time': None, 'shipping_time': None, 'subject_choice': None, 'response': None, 
                    'product_type': None, 'product_name': None, 'instruction_prompt': None, 'period_prompt': None,
                    'navigation_prompt_idx': None, 'navigation_prompt': None}  # Added for weighted prompt sampling
    record_sheet = {}
    
    # Create list of possible product names (A-pod to Z-pod)
    product_names = [f"{chr(65+i)}-Pod" for i in range(26)]  # A=65 in ASCII
    
    for i in range(n_cohort):
        # Randomly select a product name for this cohort
        product_name = random.choice(product_names)
        for k in range(n):
            key = (i+1, k+1)
            record_sheet[key] = basic_record.copy()
            record_sheet[key]['product_name'] = product_name
    
    return record_sheet

def shuffle_people(n_cohort, n = 4):
    """
    This function randomly assigns the positions of n participants for each cohort
    Each cohort is independent
    """
    result_matrix = []
    people = list(range(1, n+1))
    
    for _ in range(n_cohort):
        current_people = people[:]
        random.shuffle(current_people)
        result_matrix.append(current_people)
    result = np.array(result_matrix)
    
    return result

def add_position_record(record_sheet, positions):
    size = max(record_sheet.keys())
    for i in range(size[0]):
        for k in range(size[1]):
            key = (i+1, k+1)
            record_sheet[key]['position_id'] = key
            record_sheet[key]['subject_rank'] = k+1
            record_sheet[key]['subject_id'] = positions[i][k]
            record_sheet[key]['cohort_id'] = i+1
            record_sheet[key]['session_id'] = (i+1, positions[i][k])
    
    return record_sheet

def random_treatment(q):
    """
    Randomly assigns 1 or 0 based on probability q.
    """
    return 1 if random.random() < q else 0

def add_treatment_record(record_sheet, q, treatment_label=None):
    """
    Assigns a random treatment (1 or 0) to every key (i, j, k) in the nested dictionary
    based on probability q. If treatment_label is provided (e.g. "Q'00"), it is stored
    as the condition name; otherwise it is derived as "Q{q_percent:02d}".
    """
    if not 0 <= q <= 1:
        raise ValueError("Probability q must be between 0 and 1.")
    q_percent = round(q * 100)
    treatment = treatment_label if treatment_label is not None else f"Q{q_percent:02d}"

    if treatment in ('Q00', 'Q50'):
        cohorts = sorted({key[0] for key in record_sheet})
        for cohort in cohorts:
            keys = sorted(key for key in record_sheet if key[0] == cohort)
            if len(keys) != 4:
                raise ValueError("Each group must contain exactly four consumers.")
            assignments = ([(1, 0), (0, 0), (1, 1), (0, 1)]
                           if treatment == 'Q50' else [(0, 0), (0, 0), (0, 1), (0, 1)])
            random.shuffle(assignments)
            for key, (subject_type, shipping_time) in zip(keys, assignments):
                record_sheet[key].update(subject_type=subject_type,
                                         shipping_time=shipping_time, treatment=treatment)
    else:
        for key in record_sheet:
            record_sheet[key]['subject_type'] = random_treatment(q)
            record_sheet[key]['treatment'] = treatment
    return record_sheet

def add_product_record(record_sheet, p=0.5):
    """
    Assigns a random quality (1 or 0) to every cohort
    based on probability p. 
    """
    size = max(record_sheet.keys())
    for i in range(size[0]):
        prod_type = random_treatment(p)
        prod_value = prod_type * 3.5
        for k in range(size[1]):
            key = (i+1, k+1)
            record_sheet[key]['product_type'] = prod_type
            record_sheet[key]['product_value'] = prod_value
    return record_sheet

def add_wait_record(record_sheet):
    size = max(record_sheet.keys())
    for i in range(size[0]):
        key = (i+1, 1)
        if record_sheet[key]['treatment'] in ('Q00', 'Q50'):
            record_sheet[key]['queuing_time'] = 0
            record_sheet[key]['wait_time'] = record_sheet[key]['shipping_time']
        else:
            record_sheet[key]['wait_time'] = 1
    return record_sheet


def add_navigation_prompt_record(record_sheet, weights_csv_path=None):
    """
    Assigns a navigation prompt to each participant based on optimized weights.
    This should be called during initialization along with other randomizations.
    
    Args:
        record_sheet: The record sheet to update
        weights_csv_path: Path to CSV with columns: prompt_variant, optimal_weight, prompt_text
                         If None, uses path from BASE_CONFIG
    
    Returns:
        record_sheet: Updated record sheet with navigation_prompt_idx and navigation_prompt
    """
    # Get weights CSV path from config if not provided
    if weights_csv_path is None:
        weights_csv_path = BASE_CONFIG.get('weights_csv_path')
    
    # Check if weighted prompts are enabled
    if not BASE_CONFIG.get('use_weighted_prompts', True):
        print("Weighted prompts disabled. Using default navigation prompt.")
        return record_sheet
    
    if weights_csv_path is None:
        print("Warning: No weights_csv_path specified. Navigation prompts not assigned.")
        return record_sheet
    
    # Load weights CSV
    try:
        weights_df = pd.read_csv(weights_csv_path)
        print(f"Loaded navigation prompt weights from: {weights_csv_path}")
    except FileNotFoundError:
        print(f"Warning: Weights CSV not found at {weights_csv_path}. Navigation prompts not assigned.")
        return record_sheet
    
    # Verify required columns exist
    required_cols = ['prompt_variant', 'optimal_weight', 'prompt_text']
    missing_cols = [col for col in required_cols if col not in weights_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in weights CSV: {missing_cols}. Navigation prompts not assigned.")
        return record_sheet
    
    # Sort by prompt_variant to ensure consistent ordering (nav1, nav2, ..., nav11)
    weights_df = weights_df.sort_values('prompt_variant')
    
    # Extract weights and prompts as arrays
    prompt_variants = weights_df['prompt_variant'].tolist()
    weights_array = weights_df['optimal_weight'].values
    prompt_texts = weights_df['prompt_text'].tolist()
    
    # Normalize weights to sum to 1
    weights_array = weights_array / weights_array.sum()
    
    # Print weight distribution
    print("Navigation prompt weights:")
    for i, (variant, weight) in enumerate(zip(prompt_variants, weights_array)):
        if weight > 0.001:  # Only print non-zero weights
            print(f"  {variant}: {weight*100:.2f}%")
    
    # Assign navigation prompts to all participants based on weights
    n_participants = len(record_sheet)
    sampled_indices = np.random.choice(len(prompt_variants), size=n_participants, p=weights_array)
    
    # Update record sheet
    for i, key in enumerate(sorted(record_sheet.keys())):
        idx = sampled_indices[i]
        record_sheet[key]['navigation_prompt_idx'] = prompt_variants[idx]  # e.g., 'nav1', 'nav11'
        record_sheet[key]['navigation_prompt'] = prompt_texts[idx]
    
    # Print distribution of assigned prompts
    assigned_counts = {}
    for key in record_sheet.keys():
        nav_idx = record_sheet[key]['navigation_prompt_idx']
        assigned_counts[nav_idx] = assigned_counts.get(nav_idx, 0) + 1
    
    print(f"\nAssigned {n_participants} navigation prompts:")
    for variant in sorted(assigned_counts.keys()):
        count = assigned_counts[variant]
        print(f"  {variant}: {count} ({count/n_participants*100:.2f}%)")
    
    return record_sheet


def save_record(data, file_path):

    with atomic_output(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_record(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# The following two functions extract details locations of each participant in one experiment
# for the ease of aggreatating each participants' performance.
def get_positions(position_records, cohort_id, subject_id):
    positions = []
    df = pd.DataFrame(position_records[cohort_id-1])
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if df.iat[row, col] == subject_id:
                positions.append((cohort_id, row+1, col+1))
    return positions

def get_all_positions(position_record, n_cohort, n = 4):
    '''
    Extract details locations of each participant in the experiment.
    
    Args:
        position_record: Array containing position assignments
        n_cohort: Number of cohorts
        n: Number of participants per cohort
    '''
    position_records = position_record.reshape(n_cohort, n)
    rank_record = {}
    
    for cohort_id in range(1, n_cohort + 1):
        for subject_id in range(1, n+1):
            positions = []
            for col in range(n):
                if position_records[cohort_id-1][col] == subject_id:
                    positions.append((cohort_id, col+1))
            rank_record[(cohort_id,subject_id)] = positions
    
    return rank_record

# The following instructions get all records of participant with id n.
def get_records(session_id, position, record_sheet):
    positions = position[session_id]
    records = [record_sheet[pos] for pos in positions]
    print("Record of subject {} retrieved.".format(session_id)) 
    return records

def new_period_prompt(cohort_id, rank_id, record_sheet):
    """Render the shared decision prompt with the sampled navigation."""
    key = (cohort_id, rank_id)
    return get_period_prompt(
        key, record_sheet, navigation_prompt=record_sheet[key].get('navigation_prompt'))
