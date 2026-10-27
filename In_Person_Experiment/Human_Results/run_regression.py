"""Reproduce the joint H1/H2 results with ordinary OLS SE and two-sided tests.

Input: wait_time_decomp_main_rounds_results.csv in this script's directory.
Run: python run_joint_h1_h2_full_sample.py
Dependencies: numpy, pandas, statsmodels (and their package dependencies).
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / 'wait_time_decomp_main_rounds_results.csv'
FORMULA = 'buy ~ T*(k + rank + d)'


def main():
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
    results.to_csv(HERE / 'joint_h1_h2_full_sample_results.csv', index=False)
    coefficients.to_csv(HERE / 'joint_h1_h2_full_sample_coefficients.csv', index=False)
    counts.to_csv(HERE / 'joint_h1_h2_full_sample_counts.csv', index=False)
    audit = dict(sample_rule='All formal-round uninformed decisions; no additional filtering',
                 data_sha256=hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
                 **metadata, parameters=len(fit.params), ordinary_df=int(fit.df_resid),
                 condition_counts=data.condition.value_counts().to_dict(),
                 condition_participants=data.groupby('condition').person.nunique().to_dict(),
                 condition_sessions=data.groupby('condition').session.nunique().to_dict(),
                 formal_rounds=[5, 29], paid_rounds=[1, 25], is_informed=0,
                 covariance='Ordinary homoskedastic OLS', p_values='Two-sided')
    (HERE / 'joint_h1_h2_full_sample_audit.json').write_text(json.dumps(audit, indent=2))

    lines = ['# Joint H1-H2 results: full sample, ordinary OLS', '',
             'All 1,650 formal-round uninformed decisions; 100 participants; 25 sessions.', '',
             '$$', r'Y=\alpha+\tau T+b_q q+\gamma_qTq+b_r r+\gamma_rTr+b_dd+\gamma_dTd+\varepsilon.',
             '$$', '', 'Ordinary OLS SE; two-sided p-values and 95% confidence intervals.', '',
             '| Hypothesis | Target | Estimate (pp) | SE (pp) | 95% CI (pp) | Two-sided p | df |',
             '|---|---|---:|---:|---|---:|---:|']
    for row in results.itertuples():
        p = f'{row.p_two:.3e}' if row.p_two < 1e-6 else f'{row.p_two:.6f}'
        lines.append(f'| {row.hypothesis} | {row.target} | {100*row.estimate:+.2f} | '
                     f'{100*row.se:.2f} | [{100*row.ci_low:+.2f}, {100*row.ci_high:+.2f}] | '
                     f'{p} | {row.df:.0f} |')
    (HERE / 'joint_h1_h2_full_sample.md').write_text('\n'.join(lines) + '\n')
    print(results[['hypothesis', 'target', 'estimate', 'se', 'p_two', 'df']].to_string(index=False))


if __name__ == '__main__':
    main()
