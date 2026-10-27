# GPT H2 sensitivity to excluding wait 4

Uninformed consumers, run4. Q00 has no wait-4 observations, Q05 has none, and Q50 has 29 (all refusals).

Excluding Q00 wait 4 only removes zero rows in either comparison. Excluding wait 4 in all conditions leaves Q05/Q00 unchanged (n=2216) and removes 29 Q50 observations from Q50/Q00 (n=1702 instead of 1731).

## Ordinary probit

Refit purchase ~ wait_time + treat + wait_time*treat with cohort-clustered standard errors, using BFGS, L-BFGS and Newton optimizers (5000 iterations). Full diagnostics: wait4_ordinary_probit_diagnostics.csv. None of the restrictions removes Q00 quasi-complete separation. At Q00 wait 1 all outcomes are purchases, wait 2 contains two purchases and 620 refusals, and wait 3 contains two refusals. Intercept and slope can diverge while retaining the probability at wait 2.

For unchanged Q05/Q00, interaction estimates range from -2.954 to +0.757 across optimizers, with raw p-values ranging from 7.06e-34 to 0.99999 or missing. For restricted Q50/Q00, interaction estimates range from 3.723 to 6.719, with raw p-values from 5.42e-17 to 0.9968. Newton does not converge. Numerical convergence reported by BFGS/L-BFGS is insufficient under separation. These raw p-values are not reliable hypothesis tests and should not be interpreted as either support or rejection of H2.

## Bias-reduced probit and cohort bootstrap

Use the original results for unchanged samples. For Q50/Q00 with all wait-4 rows removed, refit AS_mean probit and run 1999 bootstrap resamples of whole cohorts within treatment and original batch, seed 20261024. All 1999 fits succeeded.

| Restriction | Comparison | Interaction | 95% percentile bootstrap CI | Approximate two-sided centered bootstrap p |
|---|---|---:|---|---:|
| Q00 only | Q05 vs Q00 | 0.872220 | [0.545805, 1.450438] | 0.0005 |
| Q00 only | Q50 vs Q00 | 2.080778 | [1.277021, 2.868246] | 0.0005 |
| All treatments | Q05 vs Q00 | 0.872220 | [0.545805, 1.450438] | 0.0005 |
| All treatments | Q50 vs Q00 | 2.080778 | [1.276413, 2.832107] | 0.0005 |

The bootstrap p-value is at the 1/2000 reporting floor and is approximate. Both comparisons support a positive H2 interaction under either restriction. The Q50 interaction point estimate is unchanged to six decimals because wait-4 purchasing was already predicted to be essentially zero by the fitted model; bootstrap intervals differ slightly.

The earlier clustered linear probability model uses waits 1 and 2 only, so neither restriction changes its data or results: Q05/Q00 +4.863642 percentage points, CI [3.364669,6.362615], p=3.828562e-10; Q50/Q00 +15.688309 percentage points, CI [12.090619,19.285999], p=9.974842e-17.

Restricting all treatments to waits 1-3 is a meaningful common-support sensitivity analysis. Excluding an already-empty Q00 cell is a no-op. Neither resolves separation. The restricted analysis describes waits 1-3, not the complete waiting-time range. These are post hoc robustness analyses conditional on the trained weights and this validation design, not proof or a randomized causal effect of waiting time. Human data are not modified in these analyses.
