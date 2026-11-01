#!/usr/bin/env python3
"""Separate, terminal-driven stages for a frozen formation/navigation combination."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from file_utils import atomic_write, protect_stage

ROOT = Path(__file__).resolve().parent


def resolve_run(args):
    from prompt_inputs import NAVIGATIONS
    for value in (args.formation, args.navigation):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError('IDs may contain only letters, numbers, underscores and hyphens.')
    run_id = getattr(args, 'run_id', None)
    if run_id is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('--run-id may contain only letters, numbers, underscores and hyphens.')
    weights_source = getattr(args, 'weights_csv', None)
    if getattr(args, 'training_test', False) or getattr(args, 'training_q50_q00', False):
        raise ValueError('Only full Q50 training with condition-specific formation is supported.')
    if weights_source and args.stage in ('train', 'fit'):
        raise ValueError('Use --weights-csv with preview/validate, or omit it to train and fit locally.')
    from prompt_inputs import FORMATION_GROUPS
    from formation_different import resolve_formation_group
    formation, by_condition = resolve_formation_group(
        FORMATION_GROUPS, args.formation, args.mode in ('cf', 'cfcn'))

    nav = NAVIGATIONS[args.navigation] if args.mode in ('cn', 'cfcn') else ['']
    if args.mode in ('cf', 'cfcn') and (not isinstance(formation, str) or not formation.strip()):
        raise ValueError('Fill in the selected formation in prompt_inputs.py first.')
    if args.mode in ('cn', 'cfcn') and (
        not isinstance(nav, list) or not nav or
        any(not isinstance(s, str) or not s.strip() for s in nav)
    ):
        raise ValueError('Provide a nonempty list of nonempty navigation strings in prompt_inputs.py.')
    from experiment_runtime import MESSAGE_FORMAT
    m = dict(message_format=MESSAGE_FORMAT, formation_id=args.formation, navigation_id=args.navigation, mode=args.mode,
             model=args.model, assistant_model=args.assistant_model, base_url=args.base_url,
             api_key_set=args.api_key_set, seed=args.seed,
             cohorts=dict(Q50=args.q50_cohorts, Q00=args.q00_cohorts, Q05=args.q05_cohorts),
             rounds=args.rounds, sessions=args.sessions,
             prompts=dict(formation=formation, navigations=nav))
    m['formation_different'] = True
    m['prompts']['formation_by_condition'] = by_condition
    folder = 'results'
    run = ROOT / folder / f'{args.formation}_{args.navigation}' / args.mode
    if run_id is not None:
        run = run / 'runs' / run_id
    path = run / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != m:
            raise ValueError('This run has different prompts/settings. Restore the original settings or archive the existing result directory before starting a new experiment.')
    elif args.stage not in ('train', 'preview'):
        if not (args.stage == 'validate' and (
            weights_source or any(p.is_file() for p in [run/'training/weights.csv', run/'weights.csv', run/'weight.csv'])
        )):
            raise ValueError('Run train/preview first, or use validate with --weights-csv.')
        run.mkdir(parents=True, exist_ok=True)
        from formation_different import prepare_reused_weights
        if not prepare_reused_weights(run, m, weights_source):
            raise ValueError('Supply a full-Q50 weights CSV before starting validation.')
        atomic_write(path, (json.dumps(m, indent=2) + '\n').encode())
    else:
        run.mkdir(parents=True, exist_ok=True)
        atomic_write(path, (json.dumps(m, indent=2) + '\n').encode())
    protect_stage(run, args.stage)
    from formation_different import prepare_reused_weights
    available = prepare_reused_weights(run, m, weights_source)
    if args.stage == 'validate' and args.mode in ('cn', 'cfcn') and not available:
        raise ValueError('Supply --weights-csv, place weight.csv/weights.csv in the run folder, or complete training first.')
    return run


def main():
    from env_loader import load_environment
    load_environment()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['preview', 'train', 'fit', 'validate', 'aggregate', 'test'])
    p.add_argument('--run-id', help='Separate directory for new work; existing result directories are preserved.')
    p.add_argument('--formation', default='F1')
    p.add_argument('--navigation', default='N1')
    p.add_argument('--mode', choices=['baseline', 'cf', 'cn', 'cfcn'], default='cfcn')
    p.add_argument('--weights-csv', help='Full-Q50 weights CSV to reuse for validation.')
    p.add_argument('--model', default='gpt5.4')
    p.add_argument('--assistant-model', default='gpt4omini')
    p.add_argument('--base-url', default='https://api.openai.com/v1')
    p.add_argument('--api-key-set', type=int, choices=range(1, 6), default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--q50-cohorts', type=int, default=17)
    p.add_argument('--q00-cohorts', type=int, default=8)
    p.add_argument('--q05-cohorts', type=int, default=14)
    p.add_argument('--rounds', type=int, default=26)
    p.add_argument('--sessions', type=int, default=2)
    p.add_argument('--nav-idx', type=int, help='Train one navigation only; run fit after all candidates finish.')
    p.add_argument('--condition', choices=['Q00', 'Q05', 'Q50'], help='Validate only this condition.')
    p.add_argument('--session', type=int, help='Validate only this session (1 through --sessions).')
    p.add_argument('--comparison', choices=['Q05', 'Q50', 'both'], default='both')
    args = p.parse_args()
    if min(args.q50_cohorts, args.q00_cohorts, args.q05_cohorts, args.rounds, args.sessions) < 1:
        p.error('Cohort, round and session counts must be positive.')
    if args.session is not None and not 1 <= args.session <= args.sessions:
        p.error('--session must be between 1 and --sessions.')
    run = resolve_run(args)
    env = os.environ.copy()
    env['TLV_RUN_DIR'] = str(run)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable, str(ROOT / 'stage_worker.py'), args.stage,
           '--comparison', args.comparison]
    for name in ('nav_idx', 'condition', 'session'):
        value = getattr(args, name)
        if value is not None:
            cmd += ['--' + name.replace('_', '-'), str(value)]
    print(f'Run directory: {run}', flush=True)
    subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
