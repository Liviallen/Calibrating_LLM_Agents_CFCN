"""Pre-run integration audit; temporary manifests, no API calls or research results."""
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline
import prompt_inputs
from simulation_counts import groups_for_session
from test_period_assignments import load_functions


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'inputs').mkdir()
        self.weights = ROOT / 'inputs/F1_N1_weights.csv'
        (self.root / 'inputs/F1_N1_weights.csv').write_bytes(self.weights.read_bytes())

    def prepare(self, mode):
        args = argparse.Namespace(formation='F1', navigation='N1', mode=mode, stage='validate',
            weights_csv=None, model='gpt5.4', assistant_model='gpt4omini', base_url='https://api.openai.com/v1',
            api_key_set=1, seed=42, q50_cohorts=425, q00_cohorts=200, q05_cohorts=14, rounds=26, sessions=2)
        with patch.object(pipeline, 'ROOT', self.root):
            run = pipeline.resolve_run(args)
        with patch.dict(os.environ, TLV_RUN_DIR=str(run)):
            prompts = load_module('audit_prompts', ROOT / 'shared_prompts.py')
            config = load_module('audit_config', ROOT / 'validation/simulation_config.py')
        return run, prompts, config

    def test_fresh_baseline_and_cfcn(self):
        for mode in ('baseline', 'cfcn'):
            run, prompts, config = self.prepare(mode)
            self.assertEqual(config.BASE_CONFIG['use_weighted_prompts'], mode == 'cfcn')
            if mode == 'cfcn':
                self.assertEqual((run / 'training/weights.csv').read_bytes(), self.weights.read_bytes())
            else:
                self.assertFalse((run / 'training/weights.csv').exists())
            for condition in ('Q00', 'Q50'):
                record = {(1, 1): dict(treatment=condition, subject_type=0, product_name='A-Pod',
                    product_value=3.5, shipping_time=0, queuing_time=0, wait_time=0)}
                nav = prompt_inputs.NAVIGATIONS['N1'][0] if mode == 'cfcn' else None
                text = prompts.get_instruction_prompt(condition) + prompts.get_period_prompt((1, 1), record, nav)
                formation = prompt_inputs.FORMATION_GROUPS['F1']['Q00' if condition == 'Q00' else 'Q50_Q05']
                self.assertEqual(formation.strip() in text, mode == 'cfcn')
                for navigation in prompt_inputs.NAVIGATIONS['N1']:
                    self.assertEqual(navigation in text, mode == 'cfcn' and navigation == nav)

    def test_instruction_and_period_parity(self):
        _, prompts, _ = self.prepare('baseline')
        q00 = prompts.get_instruction_prompt('Q00')
        q50 = prompts.get_instruction_prompt('Q50')
        for condition in ('Q00', 'Q50'):
            text = prompts.get_instruction_prompt(condition)
            normalized = text.replace(prompts._IN_PERSON_CONSUMER_TYPE_TEXT[condition], '<consumer type>')
            normalized = normalized.replace('Among the 2 informed consumers, one has 0 shipping weeks and one has 1 shipping week. The same applies to the 2 uninformed consumers. ', '')
            if condition == 'Q00':
                expected = normalized
            else:
                self.assertEqual(normalized, expected)
        rendered = []
        for condition in ('Q00', 'Q50'):
            record = {(1, 1): dict(treatment=condition, subject_type=0, product_name='A-Pod',
                product_value=3.5, shipping_time=1, queuing_time=0, wait_time=1)}
            text = prompts.get_period_prompt((1, 1), record)
            rendered.append(text.replace('None knew the product value before deciding.', '<information>').replace('Each had a 50% chance of knowing the product value before deciding.', '<information>'))
        self.assertEqual(*rendered)

    def test_navigation_algorithm_and_assignments_match_reference(self):
        original = ROOT.parents[1] / 'Kremer_Debo_Simulation'
        def sampling_ast(path):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'add_navigation_prompt_record')
        self.assertEqual(ast.dump(sampling_ast(ROOT / 'validation/setup.py')),
                         ast.dump(sampling_ast(original / 'validation/setup.py')))
        source = load_module('source_prompts', original / 'prompt_inputs.py')
        self.assertEqual(source.NAVIGATIONS['N1'], prompt_inputs.NAVIGATIONS['N1'])
        self.assertEqual(self.weights.read_bytes(), (original / 'results/F1_N1/cfcn/training/weights.csv').read_bytes())
        ns = load_functions(ROOT / 'validation/setup.py', {'np': np, 'pd': pd,
            'BASE_CONFIG': {'use_weighted_prompts': True, 'weights_csv_path': str(self.weights)}})
        for condition in ('Q00', 'Q50'):
            records = {(i, 1): {'treatment': condition} for i in range(100)}
            weights = pd.read_csv(self.weights).sort_values('prompt_variant')
            probabilities = weights.optimal_weight.values / weights.optimal_weight.sum()
            np.random.seed(42)
            expected = np.random.choice(8, size=100, p=probabilities)
            np.random.seed(42)
            with contextlib.redirect_stdout(io.StringIO()):
                ns['add_navigation_prompt_record'](records)
            for row, index in zip(records.values(), expected):
                self.assertEqual(row['navigation_prompt_idx'], weights.iloc[index].prompt_variant)
                self.assertEqual(row['navigation_prompt'], weights.iloc[index].prompt_text)

    def test_default_counts_and_clean_export(self):
        for total in (1, 200, 425):
            for sessions in (1, 2, 3, 500):
                self.assertEqual(sum(groups_for_session(total, sessions, s) for s in range(1, sessions + 1)), total)
        run, _, _ = self.prepare('baseline')
        manifest = json.loads((run / 'manifest.json').read_text())
        self.assertEqual(manifest['cohorts']['Q50'] * 4, 1700)
        self.assertEqual(manifest['cohorts']['Q00'] * 4, 800)
        aggregate = load_module('audit_aggregate', ROOT / 'validation/aggregate_data.py')
        row = dict(subject_id=1, cohort_id=1, profit=4, treatment='Q00', subject_rank=1,
            subject_type=0, product_value=0, wait_time=1, subject_choice=0, queuing_time=0, shipping_time=1)
        clean = aggregate.make_clean_df(pd.DataFrame([row]))
        self.assertEqual(clean.rank_information.tolist(), [1])
        self.assertEqual(clean.shipping_time.tolist(), [1])


if __name__ == '__main__':
    unittest.main()
