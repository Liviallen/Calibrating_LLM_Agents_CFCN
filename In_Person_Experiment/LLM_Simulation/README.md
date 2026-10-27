# In-person LLM simulation

Run commands from this directory. Defaults: Q50 425 independent groups (1,700 decisions), Q00 200 groups (800 decisions). Edit `simulation_counts.py` or override `--q50-cohorts` / `--q00-cohorts`.

```bash
python pipeline.py preview --mode cfcn
python pipeline.py validate --mode cfcn
python pipeline.py validate --mode baseline
```

Validation calls the model API. A full validation run aggregates both conditions and automatically performs the human-matched regression. Baseline includes instructions and current decision information, without formation or navigation. CFCN uses F1 and the copied N1 weights by default, without training.

## Output

Each mode writes directly to `results/cfcn/` or `results/baseline/` (likewise `cf` and `cn`), without a formation/navigation directory level. A manifest prevents mixing changed settings in an existing mode folder.

- `validation/weights/`: raw per-session CSV and pickle records.
- `validation/Q50_Q00.csv`: combined raw decisions from both conditions.
- `validation/Q50_Q00_clean.csv`: cleaned decisions with queue/shipping decomposition.
- `regression_comparison.csv`: eight model coefficients with human and simulation estimates, standard errors, 95% intervals, and two-sided p-values side by side.
- `hypothesis_comparison.csv`: six H1/H2 contrasts with human and simulation results side by side.
- `joint_h1_h2_full_sample_results.csv`: H1/H2 results, distinguished by `source`.
- `joint_h1_h2_full_sample_coefficients.csv`: model coefficients, distinguished by `source`.
- `joint_h1_h2_full_sample_counts.csv`: sample counts by source and condition.
- `regression_audit.json`: formula, sample rules, input paths and hashes.

Regression CSVs are created only after both conditions are available; preview does not create regression results. For separately completed sessions/conditions:

```bash
python pipeline.py aggregate --mode cfcn
python pipeline.py test --mode cfcn
```

Use the same settings as the simulation when rerunning analysis. Replace `cfcn` with `baseline` for baseline analysis.

## Human-matched regression

`test/human_regression_reference.py` is a byte-identical copy of `../Human_Results/run_regression.py`. `test/simulation_regression.py` preserves its human filtering, model fitting, coefficient contrasts, and two-sided t-tests. Human data are read from the original CSV without modifying that directory.

The model is `buy ~ T*(k + rank + d)`, where T is Q50, k is queuing time, rank is arrival rank, and d is total wait. It uses ordinary homoskedastic OLS standard errors, not clustered errors. H1 tests the k slope; H2 tests the rank slope, each for Q00, Q50, and the Q50-minus-Q00 difference.

Only uninformed decisions enter the regression: human formal rounds 5–29 and all simulated independent groups. Default simulation regression counts are Q50 850 and Q00 800, totaling 1,650. Simulation session counts in the tables refer to independent four-consumer groups, not execution batches or repeated human sessions. Estimates are in probability units; multiply by 100 for percentage points. The exact formula is retained: total wait and queue are included, and shipping is not added as a separate regressor.

## Offline verification

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s checks -v
```

Tests reproduce the original human analysis and exercise synthetic simulation exports in temporary folders. They make no API calls and do not generate research results.
