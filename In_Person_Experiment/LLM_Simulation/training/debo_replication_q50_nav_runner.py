"""
Shared Q50 human-experiment replication runner for a single navigation variant.

Output layout (compatible with weight_optimization.py):
  replication_results/Q50/{model}_{version}_replication_Q50_nav{k}/
    {model}_nav{k}_Q50_results.csv
    {model}_nav{k}_Q50_results.pkl
    {model}_nav{k}_Q50_store.pkl
"""

from __future__ import annotations

from file_utils import atomic_to_csv
import os
import random

from progress import Progress

import pandas as pd
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import instruction
import llm
import setup
from shared_prompts import NAVIGATION_PROMPTS
from simulation_config import MODEL_NAMES, REPLICATION_CONFIG_Q50, api_key_sets

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _filter_string(input_string: str) -> str:
    return "".join(char for char in input_string if char in ("1", "0"))


def _init_replication_store(n_instances: int, human_msg1: str) -> dict:
    store_dict = {}
    for i in range(n_instances):
        store_dict[str(i)] = InMemoryChatMessageHistory(
            messages=[
                HumanMessage(content=human_msg1),
            ]
        )
    return store_dict


def _create_chain(model, store: dict):
    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt | model
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def _create_period_prompt_from_instance(row, navigation_prompt: str) -> str:
    mock_record = {
        (1, 1): {
            "wait_time": int(row["wait_time"]),
            "subject_type": int(row["subject_type"]),
            "product_value": float(row["product_value"]),
            "product_name": f"{chr(65 + random.randint(0, 25))}-pod",
        }
    }
    return setup.new_period_prompt(
        1, 1, mock_record, navigation_prompt=navigation_prompt
    )


def _save_results(
    results_list,
    *,
    subfolder_path: str,
    model_name: str,
    condition: str,
    nav_idx: int,
) -> pd.DataFrame:
    df_results = pd.DataFrame(results_list)
    csv_filename = f"{model_name}_nav{nav_idx}_{condition}_results.csv"
    pkl_filename = f"{model_name}_nav{nav_idx}_{condition}_results.pkl"

    csv_path = os.path.join(subfolder_path, csv_filename)
    atomic_to_csv(df_results, csv_path, index=False)
    setup.save_record(results_list, os.path.join(subfolder_path, pkl_filename))
    print(f"Results saved to {csv_path}")
    return df_results


def _simulate_all_instances(
    df_human: pd.DataFrame,
    *,
    nav_idx: int,
    navigation_prompt: str,
    human_msg1: str,
    subfolder_path: str,
    model_name: str,
    condition: str,
    checkpoint_interval: int,
    chain_with_history,
    decision_classifier,
    store: dict,
    start_idx: int = 0,
) -> tuple[list, pd.DataFrame]:
    results_list: list = []
    results_pkl_filename = f"{model_name}_nav{nav_idx}_{condition}_results.pkl"
    store_filename = f"{model_name}_nav{nav_idx}_{condition}_store.pkl"
    results_pkl_path = os.path.join(subfolder_path, results_pkl_filename)
    n_instances = len(df_human)
    progress = Progress(f"train_nav{nav_idx}", n_instances, completed=start_idx)

    if start_idx > 0 and os.path.exists(results_pkl_path):
        results_list = setup.load_record(results_pkl_path)
        print(f"Loaded {len(results_list)} existing results, resuming from index {start_idx}")

    try:
        for idx in range(start_idx, n_instances):
            row = df_human.iloc[idx]
            instance_instruction = human_msg1
            period_prompt = _create_period_prompt_from_instance(row, navigation_prompt)
            response = chain_with_history.invoke(
                {"question": period_prompt},
                config={"configurable": {"session_id": str(idx)}},
            )
            decision = _filter_string(decision_classifier.invoke(response.content))
            results_list.append({
                "instance_idx": idx,
                "subject_id": row["subject_id"],
                "cohort_id": row["cohort_id"],
                "round": row["round"],
                "treatment": row["treatment"],
                "subject_rank": row["subject_rank"],
                "subject_type": row["subject_type"],
                "product_value": row["product_value"],
                "wait_time": row["wait_time"],
                "human_choice": row["subject_choice"],
                "llm_choice": decision,
                "llm_response": response.content,
                "period_prompt": period_prompt,
                "instruction_prompt": instance_instruction,
                "full_input_prompt": setup.compose_full_input_prompt(instance_instruction, period_prompt),
            })
            print(
                f"Instance {idx + 1}/{n_instances}: "
                f"Human={row['subject_choice']}, LLM={decision}"
            )

            progress.update(idx + 1)

            if (idx + 1) % checkpoint_interval == 0:
                _save_results(
                    results_list,
                    subfolder_path=subfolder_path,
                    model_name=model_name,
                    condition=condition,
                    nav_idx=nav_idx,
                )
                print(f"\n--- Checkpoint saved at instance {idx + 1} ---\n")

        df_results = _save_results(
            results_list,
            subfolder_path=subfolder_path,
            model_name=model_name,
            condition=condition,
            nav_idx=nav_idx,
        )
        store_path = os.path.join(subfolder_path, store_filename)
        setup.save_record(store, store_path)
        print(f"Conversation histories saved to {store_path}")
        progress.finish()
        return results_list, df_results

    except BaseException as exc:
        progress.fail(exc)
        print(f"\nError at instance {idx}: {exc}")
        print("Saving progress...")
        _save_results(
            results_list,
            subfolder_path=subfolder_path,
            model_name=model_name,
            condition=condition,
            nav_idx=nav_idx,
        )
        raise


def run_nav_variant(
    nav_idx: int,
    *,
    resume_from_instance: int | None = None,
    api_key_set: int | None = None,
) -> None:
    """
    Run Q50 replication for one navigation prompt variant.

    Args:
        nav_idx: Navigation index 1..len(NAVIGATION_PROMPTS).
        resume_from_instance: Optional 0-based instance index to resume from.
        api_key_set: Optional API key set (1-5). Defaults to nav_idx mapped onto 1-5.
    """
    if nav_idx < 1 or nav_idx > len(NAVIGATION_PROMPTS):
        raise ValueError(
            f"nav_idx must be between 1 and {len(NAVIGATION_PROMPTS)}, got {nav_idx}"
        )

    os.chdir(os.environ["TLV_STAGE_DIR"])

    config = REPLICATION_CONFIG_Q50
    model_name = config["model_name"]
    assistant_model_name = config["assistant_model"]
    condition = config["condition"]
    version = config["version"]
    csv_file = config["csv_file"]
    output_folder = os.path.join(config["output_folder"], condition)
    checkpoint_interval = config["checkpoint_interval"]

    if api_key_set is None:
        api_key_set = ((nav_idx - 1) % 5) + 1

    navigation_prompt = NAVIGATION_PROMPTS[nav_idx - 1]
    folder_name = f"{model_name}_{version}_replication"
    file_name = f"{folder_name}_{condition}"
    subfolder_path = os.path.join(output_folder, f"{file_name}_nav{nav_idx}")
    os.makedirs(subfolder_path, exist_ok=True)

    print(f"=== {condition} navigation variant {nav_idx}/{len(NAVIGATION_PROMPTS)} ===")
    print(f"Model: {model_name}")
    print(f"Condition: {condition}")
    print(f"API key set: {api_key_set}")
    print(f"Output folder: {subfolder_path}")

    df_human = pd.read_csv(csv_file)
    n_instances = len(df_human)
    print(f"Total instances to replicate: {n_instances}")

    results_pkl_path = os.path.join(
        subfolder_path,
        f"{model_name}_nav{nav_idx}_{condition}_results.pkl",
    )
    if os.path.exists(results_pkl_path):
        existing_results = setup.load_record(results_pkl_path)
        if len(existing_results) == n_instances:
            print(
                f"Navigation variant {nav_idx} already complete "
                f"with {len(existing_results)} instances. Skipping."
            )
            progress = Progress(f"train_nav{nav_idx}", n_instances, completed=n_instances)
            progress.finish()
            return

    start_idx = 0
    if resume_from_instance is not None:
        start_idx = resume_from_instance
    elif os.path.exists(results_pkl_path):
        existing_results = setup.load_record(results_pkl_path)
        start_idx = len(existing_results)
        if start_idx > 0:
            print(f"Auto-detected resume point: instance {start_idx}")

    human_msg1 = instruction.instruction_prompt(condition=condition)
    # Every Q50 training instance starts with independent message history.
    store = _init_replication_store(n_instances, human_msg1)

    base_url = config["openai_base_url"]
    api_key = api_key_sets[api_key_set][0]
    model = llm.get_openai(
        model=MODEL_NAMES[model_name], base_url=base_url, api_key=api_key
    )
    assistant_model = llm.get_openai(
        model=MODEL_NAMES[assistant_model_name], base_url=base_url, api_key=api_key
    )
    decision_classifier = llm.create_decision_assistant(assistant_model)
    chain_with_history = _create_chain(model, store)

    _simulate_all_instances(
        df_human,
        nav_idx=nav_idx,
        navigation_prompt=navigation_prompt,
        human_msg1=human_msg1,
        subfolder_path=subfolder_path,
        model_name=model_name,
        condition=condition,
        checkpoint_interval=checkpoint_interval,
        chain_with_history=chain_with_history,
        decision_classifier=decision_classifier,
        store=store,
        start_idx=start_idx,
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one Q50 navigation replication variant.")
    parser.add_argument(
        "--nav-idx",
        type=int,
        required=True,
        help="Candidate index within the selected navigation set.",
    )
    parser.add_argument(
        "--resume-from-instance",
        type=int,
        default=None,
        help="0-based instance index to resume from (default: auto-detect from checkpoint).",
    )
    parser.add_argument(
        "--api-key-set",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="API key set to use (default: nav_idx mapped to 1-5).",
    )
    args = parser.parse_args(argv)
    run_nav_variant(
        args.nav_idx,
        resume_from_instance=args.resume_from_instance,
        api_key_set=args.api_key_set,
    )


if __name__ == "__main__":
    main()
