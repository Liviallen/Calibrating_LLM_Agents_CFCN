"""Regression equivalence and synthetic-data integration; never calls an LLM."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test'))
import simulation_regression as regression
import human_regression_reference as reference


class RegressionTests(unittest.TestCase):
    def test_human_regression_matches_original(self):
        original = ROOT.parent / 'Human_Results/run_regression.py'
        self.assertEqual((ROOT / 'test/human_regression_reference.py').read_bytes(), original.read_bytes())
        with tempfile.TemporaryDirectory() as folder:
            old_here, old_data = reference.HERE, reference.DATA_FILE
            try:
                reference.HERE = Path(folder)
                reference.DATA_FILE = regression.DATA_FILE
                with contextlib.redirect_stdout(io.StringIO()):
                    reference.main()
                expected = pd.read_csv(Path(folder) / 'joint_h1_h2_full_sample_results.csv')
                expected_coefficients = pd.read_csv(Path(folder) / 'joint_h1_h2_full_sample_coefficients.csv')
            finally:
                reference.HERE, reference.DATA_FILE = old_here, old_data
        actual, coefficients, _ = regression.fit_regression(regression.load_human())
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
        pd.testing.assert_frame_equal(coefficients, expected_coefficients, check_dtype=False, rtol=1e-12, atol=1e-12)

    def test_simulation_mapping_and_comparison_export(self):
        rng = np.random.default_rng(42)
        rows = []
        cohort = 0
        for condition, count in [('Q50', 425), ('Q00', 200)]:
            for _ in range(count):
                cohort += 1
                queue = 0
                pairs = [(0, 0), (0, 1), (1, 0), (1, 1)] if condition == 'Q50' else [(0, 0), (0, 0), (0, 1), (0, 1)]
                rng.shuffle(pairs)
                for rank, (informed, shipping) in enumerate(pairs, 1):
                    choice = int(rng.random() < .5)
                    rows.append(dict(subject_id=rank, cohort_id=cohort, treatment=condition,
                        subject_rank=rank, subject_type=informed, queuing_time=queue,
                        shipping_time=shipping, wait_time=queue + shipping, subject_choice=choice))
                    queue += choice
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            data = regression.load_simulation(path)
            self.assertEqual(data.condition.value_counts().to_dict(), {'Q50': 850, 'Q00': 800})
            with contextlib.redirect_stdout(io.StringIO()):
                regression.run_comparison(path, folder)
            coefficients = pd.read_csv(Path(folder) / 'regression_comparison.csv')
            hypotheses = pd.read_csv(Path(folder) / 'hypothesis_comparison.csv')
            self.assertEqual(len(coefficients), 8)
            self.assertEqual(len(hypotheses), 6)
            for table in (coefficients, hypotheses):
                for source in ('human', 'simulation'):
                    self.assertTrue(table[f'p_two_{source}'].between(0, 1).all())
                    self.assertTrue(table[f'n_{source}'].eq(1650).all())
            bad = pd.DataFrame(rows)
            bad.loc[0, 'queuing_time'] = 99
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                regression.load_simulation(path)


if __name__ == '__main__':
    unittest.main()
