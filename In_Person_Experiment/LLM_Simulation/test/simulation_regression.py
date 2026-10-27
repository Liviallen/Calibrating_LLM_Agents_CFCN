"""Human-matched OLS comparison; adapted from Human_Results/run_regression.py.

The reference script is preserved verbatim in human_regression_reference.py.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT.parent / 'Human_Results/wait_time_decomp_main_rounds_results.csv'
FORMULA = 'buy ~ T*(k + rank + d)'


def load_human():
    columns = ['participant_id', 'session_code', 'treatment', 'round_number',
               'main_round_number', 'arrival_rank', 'is_informed',
               'offered_queue_time', 'shipping_time', 'total_wait_time', 'decision']
    raw = pd.read_csv(DATA_FILE, usecols=columns, encoding='utf-8-sig')
    data = raw.loc[raw.is_informed.eq(0) & raw.round_number.between(5, 29)].copy()
    data = data.rename(columns={'participant_id': 'person', 'session_code': 'session',
                               'treatment': 'condition', 'arrival_rank': 'rank',
                               'offered_queue_time': 'k', 'shipping_time': 's',
                               'total_wait_time': 'd'})
    data['buy'] = data.decision.map({'ORDER': 1., 'DO_NOT_ORDER': 0.})
    data['T'] = data.condition.eq('Q50').astype(int)
    assert len(data) == 1650 and data.person.nunique() == 100
    assert data.session.nunique() == 25
    assert data.condition.value_counts().to_dict() == {'Q50': 850, 'Q00': 800}
    assert data.main_round_number.eq(data.round_number - 4).all()
    assert not data.duplicated(['person', 'round_number']).any()
    assert data.buy.notna().all() and np.allclose(data.d, data.k + data.s)

    return data


def load_simulation(path):
    raw = pd.read_csv(path)
    required = {'subject_id', 'cohort_id', 'treatment', 'subject_rank', 'subject_type',
                'queuing_time', 'shipping_time', 'wait_time', 'subject_choice'}
    if not required <= set(raw.columns):
        raise ValueError(f'Missing simulation fields: {sorted(required - set(raw.columns))}')
    if not raw.treatment.isin(['Q00', 'Q50']).all():
        raise ValueError('Simulation data must contain only Q00/Q50.')
    if not raw.subject_type.isin([0, 1]).all() or not raw.subject_choice.isin([0, 1]).all():
        raise ValueError('Invalid information types or decisions.')
    if raw.duplicated(['cohort_id', 'subject_rank']).any():
        raise ValueError('Duplicate consumer ranks within simulation groups.')
    for _, group in raw.groupby('cohort_id'):
        if len(group) != 4 or set(group.subject_rank) != {1, 2, 3, 4} or group.treatment.nunique() != 1:
            raise ValueError('Each simulation group must contain four ordered consumers in one condition.')
        ordered = group.sort_values('subject_rank')
        expected_queue = ordered.subject_choice.cumsum() - ordered.subject_choice
        if not np.array_equal(ordered.queuing_time, expected_queue):
            raise ValueError('Queuing time differs from prior orders.')
        pairs = sorted(zip(group.subject_type, group.shipping_time))
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)] if group.treatment.iloc[0] == 'Q50' else [(0, 0), (0, 0), (0, 1), (0, 1)]
        if pairs != expected:
            raise ValueError('Invalid within-group information/shipping assignment.')
    data = raw.loc[raw.subject_type.eq(0)].copy()
    data = data.rename(columns={'treatment': 'condition', 'subject_rank': 'rank',
        'queuing_time': 'k', 'shipping_time': 's', 'wait_time': 'd', 'subject_choice': 'buy'})
    # Independent simulated groups have no repeated participant histories.
    data['session'] = data.cohort_id
    data['person'] = data.cohort_id.astype(str) + ':' + data.subject_id.astype(str)
    data['T'] = data.condition.eq('Q50').astype(int)
    data['buy'] = data.buy.astype(float)
    if set(data.condition) != {'Q00', 'Q50'} or not np.allclose(data.d, data.k + data.s):
        raise ValueError('Both conditions and consistent wait decomposition are required.')
    return data


def fit_regression(data):
    fit = smf.ols(FORMULA, data=data, missing='raise').fit(cov_type='nonrobust', use_t=True)
    assert len(fit.params) == 8 and np.linalg.matrix_rank(fit.model.exog) == 8
    metadata = dict(n=int(fit.nobs), participants=data.person.nunique(),
                    sessions=data.session.nunique(), formula=FORMULA, status='ok')

    def contrast(weights):
        vector = np.array([weights.get(term, 0) for term in fit.params.index])
        test = fit.t_test(vector)
        lo, hi = test.conf_int(alpha=.05)[0]
        return dict(estimate=np.asarray(test.effect).item(), se=np.asarray(test.sd).item(),
                    ci_low=float(lo), ci_high=float(hi), p_two=np.asarray(test.pvalue).item(),
                    df=float(fit.df_resid), **metadata)

    results = []
    for hypothesis, variable in [('H1', 'k'), ('H2', 'rank')]:
        for target, weights in [('Q00', {variable: 1}),
                                ('Q50', {variable: 1, 'T:' + variable: 1}),
                                ('Difference', {'T:' + variable: 1})]:
            results.append(dict(se_method='OLS', model='Linear', hypothesis=hypothesis,
                                target=target, **contrast(weights)))
    results = pd.DataFrame(results)
    coefficients = pd.DataFrame([dict(se_method='OLS', term=term, **contrast({term: 1}))
                                 for term in fit.params.index])
    counts = data.groupby('condition').agg(decisions=('buy', 'size'),
                                           participants=('person', 'nunique'),
                                           sessions=('session', 'nunique')).reset_index()
    return results, coefficients, counts


def run_comparison(simulation_csv, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_results, all_coefficients, all_counts, audit = [], [], [], {}
    for label, path, data in [('human', DATA_FILE, load_human()),
                              ('simulation', Path(simulation_csv), load_simulation(simulation_csv))]:
        results, coefficients, counts = fit_regression(data)
        for table in (results, coefficients, counts):
            table.insert(0, 'source', label)
        all_results.append(results)
        all_coefficients.append(coefficients)
        all_counts.append(counts)
        audit[label] = dict(input=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            n=len(data), condition_counts=data.condition.value_counts().to_dict(),
            session_unit='human experimental session' if label == 'human' else 'independent four-consumer group')
    results = pd.concat(all_results, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    counts = pd.concat(all_counts, ignore_index=True)
    results.to_csv(output / 'joint_h1_h2_full_sample_results.csv', index=False)
    coefficients.to_csv(output / 'joint_h1_h2_full_sample_coefficients.csv', index=False)
    counts.to_csv(output / 'joint_h1_h2_full_sample_counts.csv', index=False)
    # Wide tables place human and simulation estimates and two-sided p-values side by side.
    for name, tables, keys in [('regression_comparison.csv', all_coefficients, ['se_method', 'term']),
                               ('hypothesis_comparison.csv', all_results, ['se_method', 'model', 'hypothesis', 'target'])]:
        columns = ['estimate', 'se', 'ci_low', 'ci_high', 'p_two', 'df', 'n', 'status']
        left, right = [table[keys + columns] for table in tables]
        left.merge(right, on=keys, suffixes=('_human', '_simulation'), validate='one_to_one').to_csv(output / name, index=False)
    audit.update(formula=FORMULA, covariance='Ordinary homoskedastic OLS', p_values='Two-sided',
                 sample_rule='Uninformed only; human rounds 5–29; all independent simulation groups')
    (output / 'regression_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(results[['source', 'hypothesis', 'target', 'estimate', 'p_two']].to_string(index=False))
    return results, coefficients
