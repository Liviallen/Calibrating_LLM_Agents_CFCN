from file_utils import atomic_to_csv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import llm
import setup
import update
import instruction
import pandas as pd
from simulation_config import (
    EXPERIMENT_CONFIG,
    CONDITION_PARAMS,
    MODEL_NAMES,
    api_key_sets
)

# Load configuration for this simulation notebook
model_name = EXPERIMENT_CONFIG['model_name']
assistant_model_name = EXPERIMENT_CONFIG['assistant_model'] 
condition = EXPERIMENT_CONFIG['condition']
version = EXPERIMENT_CONFIG['version']
n = EXPERIMENT_CONFIG['n_per_cohort']  # number of participants per cohort
n_cohort = round(EXPERIMENT_CONFIG['n_total_cohort'] * EXPERIMENT_CONFIG['round'] / EXPERIMENT_CONFIG['n_parallel_sessions'])  # total cohorts in this notebook

print(f"model_name: {model_name}")
print(f"assistant_model_name: {assistant_model_name}")
print(f"condition: {condition}")
print(f"version: {version}")
print(f"n (participants per cohort): {n}")
print(f"n_cohort (cohorts in this notebook): {n_cohort}")
print(f"n_parallel_sessions: {EXPERIMENT_CONFIG['n_parallel_sessions']}")
print(f"n_total_cohort: {EXPERIMENT_CONFIG['n_total_cohort']}")
print(f"round: {EXPERIMENT_CONFIG['round']}")

# Get the appropriate API key set based on configuration
base_url = EXPERIMENT_CONFIG['openai_base_url']
api_key_set = EXPERIMENT_CONFIG['api_key_set']
api_keys = api_key_sets[api_key_set]
api_key = api_keys[0]  # Always use index 0 to avoid IndexError
model = llm.get_openai(model = MODEL_NAMES[model_name], base_url=base_url, api_key=api_key)
assistant_model = llm.get_openai(model = MODEL_NAMES[assistant_model_name], base_url=base_url, api_key=api_key)
decision_classifier = llm.create_decision_assistant(assistant_model)

print(MODEL_NAMES[model_name])
print(MODEL_NAMES[assistant_model_name])
print(base_url)
print(f"API key loaded: {'Yes' if api_key else 'No (please set OPENAI_API_KEY environment variable)'}")

# Generate folder name from weights_csv_path (without .csv extension)
weights_csv_path = EXPERIMENT_CONFIG.get('weights_csv_path', '')
if weights_csv_path:
    # Extract filename without extension to use as folder name
    folder_name = os.path.splitext(os.path.basename(weights_csv_path))[0]
    folder_name = folder_name.replace('_with_prompts', '')  # Shorten path to avoid length errors
    folder_name = folder_name.replace('_replication_', '_')  # Remove "replication" from folder/file names
else:
    folder_name = '{}_{}'.format(model_name, version)

file_name = '{}_{}_sim{}'.format(folder_name, condition, os.environ.get("TLV_SESSION", "1"))
# Create main folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# Create subfolder with file_name if it doesn't exist
subfolder_path = os.path.join(folder_name, file_name)
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)
print(f"folder_name: {folder_name}")
print(f"file_name: {file_name}")

# Get the full experiment instruction
human_msg1 = instruction.instruction_prompt(condition=condition)

store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    return store[session_id]

def create_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt | model
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history

chain_with_history = create_chain(model)

# Create and initialize experiment records
record = setup.create_record_sheet(n_cohort=n_cohort)
position_record = setup.shuffle_people(n_cohort=n_cohort)

record = setup.add_position_record(record_sheet=record, positions=position_record)
params = CONDITION_PARAMS[condition]  # KeyError if unknown condition
record = setup.add_treatment_record(record_sheet=record, q=params["q"], treatment_label=condition)
record = setup.add_product_record(record_sheet=record, p=params["p"])
record = setup.add_wait_record(record_sheet=record)
# Assign weighted navigation prompts (loaded from CSV specified in BASE_CONFIG)
record = setup.add_navigation_prompt_record(record_sheet=record)

# Save records
setup.save_record(record, os.path.join(subfolder_path, "{}.pkl".format(file_name)))
setup.save_record(position_record, os.path.join(subfolder_path, "{}_position.pkl".format(file_name)))
# Load records and initialize history
record = setup.load_record(os.path.join(subfolder_path, "{}.pkl".format(file_name)))
position_record = setup.load_record(os.path.join(subfolder_path, "{}_position.pkl".format(file_name)))
subject_positions = setup.get_all_positions(position_record, n_cohort=n_cohort)
store = llm.init_history(n_cohort=n_cohort, human_msg1=human_msg1)
for key in record.keys():
    record[key]['instruction_prompt'] = human_msg1

# Simulation functions
def get_response(cohort_id, rank_id, prompt, record_sheet):
    """Get response from the LLM for a participant."""
    id = record_sheet[(cohort_id, rank_id)]['session_id']
    response = chain_with_history.invoke(
        {"question": prompt},
        config={"configurable": {"session_id": f'{id[0]},{id[1]}'}}
    )
    return response

def filter_string(input_string):
    """Filter string to only keep '1' and '0'."""
    filtered = ''.join(char for char in input_string if char in ('1', '0'))
    return filtered

def simulate(cohort_id, rank_id, record_sheet=record, positions=subject_positions):
    """Simulate the decision of a participant.
    
    Args:
        cohort_id: The cohort identifier (1 to n_cohort)
        rank_id: The participant's rank in the cohort (1 to 4)
        record_sheet: The record sheet to update
        positions: The position records
    """
    # Get and store both prompts
    key = (cohort_id, rank_id)
    new_prompt = setup.new_period_prompt(cohort_id, rank_id, record_sheet)
    record_sheet[key]['period_prompt'] = new_prompt
    
    # Ensure instruction prompt is stored (in case it wasn't stored earlier)
    if record_sheet[key]['instruction_prompt'] is None:
        record_sheet[key]['instruction_prompt'] = instruction.instruction_prompt(condition=record_sheet[key]['treatment'])
    
    #print(new_prompt)
    response = get_response(cohort_id, rank_id, new_prompt, record_sheet)
    decision = decision_classifier.invoke(response.content)
    decision = filter_string(decision)
    # Fallback handling: if no valid decision extracted, default to '0' (NOT ORDER)
    if not decision:
        print(f"Warning: Could not extract decision from LLM response, defaulting to '0' (NOT ORDER)")
        decision = '0'
    nav_idx = record_sheet[key].get('navigation_prompt_idx', 'default')
    print(f"Simulating cohort {cohort_id}, rank {rank_id} (nav: {nav_idx})")
    print(f"The decision is '{decision}'")

    # Add responses to the record sheet
    record = update.add_response_record(cohort_id, rank_id, response=response, record_sheet=record_sheet)
    record = update.add_choice_record(cohort_id, rank_id, subject_choice=decision, record_sheet=record_sheet)
    record = update.add_payoff_record(cohort_id, rank_id, record_sheet=record_sheet)
    if rank_id != 4:
        record = update.update_wait_time(cohort_id, rank_id, record_sheet=record_sheet)
        return record
    elif rank_id == 4:
        return record
    
def simulate_all(n_cohort=n_cohort, n=4, file_name=file_name):
    """Simulate all cohorts from start to finish.
    
    Args:
        n_cohort: Total number of independent cohorts to simulate
        n: Number of participants per cohort
        file_name: Base name for saving files
    """
    try:
        for i in range(1, n_cohort+1):
            for k in range(1, n+1):
                record = simulate(i, k)
                # Save after each participant to enable recovery
                setup.save_record(record, os.path.join(subfolder_path, "{}.pkl".format(file_name)))
                setup.save_record(position_record, os.path.join(subfolder_path, "{}_position.pkl".format(file_name)))
                from progress import validation_progress
                validation_progress.update((i - 1) * n + k)
        return record
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise

def resume_simulation(n_cohort=n_cohort, n=4, cohort_id=1, rank_id=1, file_name=file_name):
    """Resume simulation from a specific point.
    
    Args:
        n_cohort: Total number of independent cohorts to simulate
        n: Number of participants per cohort
        cohort_id: Cohort to resume from
        rank_id: Rank to resume from
        file_name: Base name for saving files
    """
    try:
        # Complete current cohort
        for k in range(rank_id, n+1):
            record = simulate(cohort_id, k)
            setup.save_record(record, os.path.join(subfolder_path, "{}.pkl".format(file_name)))
            setup.save_record(position_record, os.path.join(subfolder_path, "{}_position.pkl".format(file_name)))
            
        # Do remaining cohorts
        for i in range(cohort_id+1, n_cohort+1):
            for k in range(1, n+1):
                record = simulate(i, k)
                setup.save_record(record, os.path.join(subfolder_path, "{}.pkl".format(file_name)))
                setup.save_record(position_record, os.path.join(subfolder_path, "{}_position.pkl".format(file_name)))
        return record
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise

pd.DataFrame(record).T

store

print(model_name)
print(version)
print(human_msg1)
print(setup.new_period_prompt(1,1,record))

record = simulate_all(n_cohort=n_cohort, n = n, file_name=file_name)

print(len(store['1,1'].messages),len(store['1,2'].messages),len(store['1,3'].messages),len(store['1,4'].messages))

# Removed notebook-only inspection (requires at least four cohorts).

setup.save_record(record, os.path.join(subfolder_path, "{}.pkl".format(file_name)))
setup.save_record(store, os.path.join(subfolder_path, "{}_history.pkl".format(file_name)))
setup.save_record(position_record, os.path.join(subfolder_path, "{}_position.pkl".format(file_name)))

record = setup.load_record(os.path.join(subfolder_path, "{}.pkl".format(file_name)))
store = setup.load_record(os.path.join(subfolder_path, "{}_history.pkl".format(file_name)))
position_record = setup.load_record(os.path.join(subfolder_path, "{}_position.pkl".format(file_name)))
subject_positions = setup.get_all_positions(position_record, n_cohort=n_cohort)

df_record = pd.DataFrame(record).T
df_record = df_record.reset_index(drop=True)
atomic_to_csv(df_record, os.path.join(subfolder_path, "{}.csv".format(file_name)), index=False)
