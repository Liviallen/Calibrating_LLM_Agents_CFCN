"""Adapters around local copies only; no imports from the original source folders."""
from file_utils import atomic_to_csv, atomic_write, atomic_output, validate_session_csv, protect_stage
from simulation_counts import SIMULATION_CONDITIONS, groups_for_session
import argparse
import importlib
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys

from experiment_runtime import ROOT, manifest


def stage_path(stage):
    sys.path.insert(0, str(ROOT / stage))


def run_dir():
    return Path(os.environ['TLV_RUN_DIR'])


def fit():
    protect_stage(run_dir(), 'fit')
    import numpy as np
    import pandas as pd
    stage_path('training')
    import weight_optimization as wo
    m = manifest()
    prompts = m['prompts']['navigations']
    root = run_dir() / 'training'
    scope = 'Q50'
    for idx in range(1, len(prompts) + 1):
        path = root / f'replication_results/{scope}' / f'{m["model"]}_context_formation_and_navigation_replication_{scope}_nav{idx}' / f'{m["model"]}_nav{idx}_{scope}_results.csv'
        raw = pd.read_csv(path)
        choices = pd.to_numeric(raw['llm_choice'], errors='coerce')
        if choices.isna().any() or not choices.isin([0, 1]).all():
            raise ValueError(f'Invalid classifier decisions in {path}; correct/re-run them before fitting.')
    predictions = wo.load_llm_predictions_for_condition(
        str(root / 'replication_results'), m['model'], scope, len(prompts))
    # Reject invalid classifier output rather than silently turning it into a zero.
    Y, X, _ = wo.prepare_data_for_optimization(
        str(ROOT / 'training/kremer-debo_q50_uninformed.csv'), predictions)
    if len(prompts) == 1:
        weights = np.ones(1)
        result = dict(weights=weights, loss=wo.cross_entropy_loss(weights, Y, X),
                      method='fixed_single_prompt', loss_function='cross_entropy')
        success, message = True, 'No navigation weights to fit.'
    else:
        result = wo.optimize_prompt_weights(Y, X, method='softmax', loss_function='cross_entropy')
        success = bool(result['optimization_result'].success)
        message = str(result['optimization_result'].message)
    wo.save_weights_to_csv(result, str(root / 'weights.csv'), navigation_prompts=prompts)
    p = np.clip(X @ (result['weights'] / (result['weights'].sum() + 1e-10)), 1e-10, 1 - 1e-10)
    atomic_to_csv(pd.DataFrame([dict(formation=m['formation_id'], navigation=m['navigation_id'], mode=m['mode'],
                      model=m['model'], condition=scope, n_instances=len(Y), n_candidates=len(prompts),
                      loss_function='cross_entropy', total_training_loss=result['loss'],
                      mean_training_loss=result['loss'] / len(Y),
                      training_brier_mean=float(np.mean((p - Y)**2)),
                      optimizer_success=success, optimizer_message=message)]), root / 'training_loss.csv', index=False)
    print(pd.read_csv(root / 'training_loss.csv').to_string(index=False))
    if not success:
        print('WARNING: optimizer did not report convergence; inspect training_loss.csv before interpreting results.')


def train(nav_idx):
    protect_stage(run_dir(), 'train')
    stage_path('training')
    from debo_replication_q50_nav_runner import run_nav_variant
    count = len(manifest()['prompts']['navigations'])
    indices = [nav_idx] if nav_idx is not None else range(1, count + 1)
    if any(i < 1 or i > count for i in indices):
        raise ValueError(f'--nav-idx must be between 1 and {count}.')
    for i in indices:
        random.seed(manifest()['seed'])
        run_nav_variant(i, api_key_set=manifest()['api_key_set'])
    if nav_idx is None:
        fit()


def validate(condition, session):
    m = manifest()
    weights = run_dir() / 'training/weights.csv'
    if m['mode'] in ('cn', 'cfcn') and not weights.exists():
        raise ValueError('Training weights are missing. Complete train/fit first.')
    (run_dir() / 'validation/weights').mkdir(parents=True, exist_ok=True)
    conditions = [condition] if condition else SIMULATION_CONDITIONS
    sessions = [session] if session else range(1, m['sessions'] + 1)
    for cond in conditions:
        for sess in sessions:
            if groups_for_session(m['cohorts'][cond], m['sessions'], sess) == 0:
                continue
            output = run_dir() / 'validation/weights' / f'weights_{cond}_sim{sess}' / f'weights_{cond}_sim{sess}.csv'
            if output.exists():
                validate_session_csv(output, groups_for_session(m['cohorts'][cond], m['sessions'], sess) * 4)
                print(f'Already complete: {output}')
                continue
            if output.parent.exists() and any(output.parent.iterdir()):
                raise ValueError(f'Incomplete session artifacts at {output.parent}; preserve them and use a new --run-id.')
            env = os.environ.copy()
            env.update(TLV_CONDITION=cond, TLV_SESSION=str(sess))
            subprocess.run([sys.executable, str(ROOT / 'stage_worker.py'), '_simulate'], env=env, check=True)
    if condition is None and session is None:
        if any((run_dir() / 'validation').glob('*_Q00.csv')):
            print('Existing aggregate outputs preserved; use a new --run-id for new work.')
            return
        aggregate()
        test()


def simulate():
    import numpy as np
    m = manifest()
    session_dir = run_dir() / 'validation/weights' / f"weights_{os.environ['TLV_CONDITION']}_sim{os.environ['TLV_SESSION']}"
    if session_dir.exists() and any(session_dir.iterdir()):
        raise ValueError('Session outputs already exist; use a new --run-id.')
    seed = m['seed'] + int(os.environ['TLV_SESSION']) + 100 * ['Q00', 'Q05', 'Q50'].index(os.environ['TLV_CONDITION'])
    random.seed(seed)
    np.random.seed(seed)
    stage_path('validation')
    import progress
    condition = os.environ['TLV_CONDITION']
    session = os.environ['TLV_SESSION']
    total = groups_for_session(m['cohorts'][condition], m['sessions'], int(session)) * 4
    if total == 0:
        return
    with progress.Progress(f'validate_{condition}_sim{session}', total) as tracker:
        progress.validation_progress = tracker
        runpy.run_path(str(ROOT / 'validation/debo_condition1.py'), run_name='__main__')


def aggregate():
    protect_stage(run_dir(), 'aggregate')
    import pandas as pd
    stage_path('validation')
    from aggregate_data import make_clean_df
    m = manifest()
    dfs = {}
    for cond in SIMULATION_CONDITIONS:
        frames, offset = [], 0
        for sess in range(1, m['sessions'] + 1):
            expected_rows = groups_for_session(m['cohorts'][cond], m['sessions'], sess) * 4
            if expected_rows == 0:
                continue
            name = f'weights_{cond}_sim{sess}'
            path = run_dir() / 'validation/weights' / name / f'{name}.csv'
            if not path.exists():
                raise ValueError(f'Missing validation session: {path}')
            df = pd.read_csv(path)
            if len(df) != expected_rows or df.subject_choice.isna().any() or not df.subject_choice.isin([0, 1]).all():
                raise ValueError(f'Incomplete/invalid validation data: {path}')
            df['cohort_id'] += offset
            offset = int(df.cohort_id.max())
            frames.append(df)
        dfs[cond] = pd.concat(frames, ignore_index=True)
    for cond in ['Q50']:
        a, b = dfs[cond].copy(), dfs['Q00'].copy()
        b['cohort_id'] += int(a.cohort_id.max())
        combined = pd.concat([a, b], ignore_index=True)
        # Keep raw identifiers aligned with reassigned cohorts, as in the reference aggregator.
        combined['session_id'] = [
            str((int(cohort), int(subject)))
            for cohort, subject in zip(combined['cohort_id'], combined['subject_id'])
        ]
        combined['position_id'] = [
            str((int(cohort), int(rank)))
            for cohort, rank in zip(combined['cohort_id'], combined['subject_rank'])
        ]
        atomic_to_csv(combined, run_dir() / f'validation/{cond}_Q00.csv', index=False)
        atomic_to_csv(make_clean_df(combined), run_dir() / f'validation/{cond}_Q00_clean.csv', index=False)
        print(f'Aggregated {cond} vs Q00: {len(combined)} rows')


def test(comparison='Q50'):
    protect_stage(run_dir(), "test")
    stage_path('test')
    from simulation_regression import run_comparison
    path = run_dir() / 'validation/Q50_Q00.csv'
    if not path.exists():
        raise ValueError('Aggregate complete Q50/Q00 simulation outputs before regression.')
    run_comparison(path, run_dir())
    print(f'Regression comparison saved in {run_dir()}')


def preview():
    protect_stage(run_dir(), 'preview')
    import shared_prompts as p
    m = manifest()
    record = {(1, 1): dict(wait_time=1, subject_type=0, product_value=3.5, product_name='A-Pod', queuing_time=0, shipping_time=1)}
    for cond in SIMULATION_CONDITIONS:
        record[(1, 1)]['treatment'] = cond
        text = (
            "=== MESSAGE 1 | USER: instruction + formation ===\n"
            + p.get_instruction_prompt(cond)
            + "\n\n=== MESSAGE 2 | USER: instance details + navigation + answer request ===\n"
            + p.get_period_prompt((1, 1), record, m['prompts']['navigations'][0])
        )
        atomic_write(run_dir() / f'{cond}_prompt_preview.txt', text.encode())
    print('Saved Q50/Q00 prompt previews (first navigation candidate).')


def main():
    from env_loader import load_environment
    load_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--nav-idx', type=int)
    parser.add_argument('--condition')
    parser.add_argument('--session', type=int)
    parser.add_argument('--comparison', default='both')
    args = parser.parse_args()
    protect_stage(run_dir(), args.stage)
    for stage in ['training', 'validation', 'test']:
        (run_dir() / stage).mkdir(exist_ok=True)
    directory = run_dir() / ('training' if args.stage in ('train', 'fit') else 'validation')
    os.environ['TLV_STAGE_DIR'] = str(directory)
    os.chdir(directory)
    if args.stage in ('train', 'validate', '_simulate'):
        m = manifest()
        from experiment_runtime import MESSAGE_FORMAT
        if m.get('message_format') != MESSAGE_FORMAT:
            raise ValueError('This run uses the earlier message format. Archive the existing result directory and retrain before validation.')
        if not (os.environ.get(f'OPENAI_API_KEY_{m["api_key_set"]}') or os.environ.get('OPENAI_API_KEY')):
            raise ValueError('Set OPENAI_API_KEY in LLM_Simulation/.env or export it (optional: OPENAI_API_KEY_1 through _5).')
    if args.stage == 'train': train(args.nav_idx)
    elif args.stage == 'fit':
        from progress import Progress
        with Progress('fit', 1):
            fit()
    elif args.stage == 'validate': validate(args.condition, args.session)
    elif args.stage == '_simulate': simulate()
    elif args.stage == 'aggregate':
        from progress import Progress
        with Progress('aggregate', 1):
            aggregate()
    elif args.stage == 'test':
        from progress import Progress
        with Progress(f'test_{args.comparison}', 1):
            test(args.comparison)
    elif args.stage == 'preview': preview()
    else: raise ValueError(args.stage)


if __name__ == '__main__':
    main()
