"""Offline checks of assignment, sequential decisions, and private information."""
import ast
import itertools
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, namespace):
    # Load pure functions without model, manifest, pandas, or numpy dependencies.
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class PeriodTests(unittest.TestCase):
    def setUp(self):
        self.setup = load_functions(ROOT / 'validation/setup.py', {'random': random.Random(42)})
        self.update = load_functions(ROOT / 'validation/update.py', {})
        self.prompts = load_functions(ROOT / 'shared_prompts.py', {
            '_FORMATION_BY_CONDITION': {'Q00': 'Assigned formation.', 'Q50': 'Assigned formation.'}})

    def records(self, condition, count=1):
        record = self.setup['create_record_sheet'](count)
        self.setup['add_treatment_record'](record, 0.5 if condition == 'Q50' else 0, condition)
        self.setup['add_product_record'](record)
        self.setup['add_wait_record'](record)
        return record

    def test_exact_assignments_and_shared_value(self):
        for condition in ('Q00', 'Q50'):
            record = self.records(condition, 100)
            values = set()
            assignments = set()
            for cohort in range(1, 101):
                group = [record[(cohort, rank)] for rank in range(1, 5)]
                pairs = [(r['subject_type'], r['shipping_time']) for r in group]
                expected = [(0, 0), (0, 1), (1, 0), (1, 1)] if condition == 'Q50' else [(0, 0), (0, 0), (0, 1), (0, 1)]
                self.assertEqual(sorted(pairs), expected)
                self.assertEqual(len({r['product_value'] for r in group}), 1)
                values.add(group[0]['product_value'])
                assignments.add(tuple(pairs))
            self.assertEqual(values, {0.0, 3.5})
            self.assertGreater(len(assignments), 1)

    def test_all_decision_sequences(self):
        for condition, decisions in itertools.product(('Q00', 'Q50'), itertools.product('01', repeat=4)):
            record = self.records(condition)
            for rank, choice in enumerate(decisions, 1):
                row = record[(1, rank)]
                queue = decisions[:rank - 1].count('1')
                total = queue + row['shipping_time']
                prompt = self.prompts['get_period_prompt']((1, rank), record, 'Assigned navigation.')
                self.assertIn(f'{rank - 1} consumers arrived before you: {queue} ordered, {rank - 1 - queue} did not.', prompt)
                self.assertIn(f'total wait time of {total} {"week" if total == 1 else "weeks"}.', prompt)
                self.assertEqual(row['wait_time'], total)
                self.assertIn('Assigned formation.', prompt)
                self.assertIn('Assigned navigation.', prompt)
                self.update['add_choice_record'](1, rank, choice, record)
                self.update['add_payoff_record'](1, rank, record)
                self.assertEqual(row['profit'], 4 if choice == '0' else 4 - total + row['product_value'])
                self.update['update_wait_time'](1, rank, record)

    def test_value_visibility(self):
        for condition, subject_type, value in itertools.product(('Q00', 'Q50'), (0, 1), (0.0, 3.5)):
            record = self.records(condition)
            record[(1, 1)].update(subject_type=subject_type, product_value=value)
            prompt = self.prompts['get_period_prompt']((1, 1), record)
            informed = condition == 'Q50' and subject_type == 1
            if informed:
                self.assertIn('You are INFORMED.', prompt)
                self.assertIn('The product value is $3.5.' if value == 3.5 else 'The product value is $0.0.', prompt)
            else:
                self.assertIn('You are UNINFORMED.', prompt)
                self.assertIn('The product will be a big hit ($3.5) or a relative disaster ($0.0).', prompt)
            self.assertIn('None knew' if condition == 'Q00' else 'Each had a 50% chance', prompt)

    def test_week_wording_and_pending_decisions(self):
        for value in range(5):
            self.assertEqual(self.prompts['_weeks'](value), f'{value} ' + ('week' if value == 1 else 'weeks'))
        record = self.records('Q50')
        with self.assertRaises(ValueError):
            self.prompts['get_period_prompt']((1, 2), record)


if __name__ == '__main__':
    unittest.main()
