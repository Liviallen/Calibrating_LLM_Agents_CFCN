# Pre-run audit — 26 October 2026

Eight offline checks pass. No model API calls or simulation results were generated.

## Verified

- Q00/Q50 instructions share one template. Outside formation, differences are consumer information status and the required Q50 information/shipping pairing explanation. Uninformed period prompts differ only in the statement about prior consumers' information.
- One 50/50 product-value draw is made per group and stored for all four sequential consumers. The update functions never redraw product value.
- Q50 randomly permutes the four distinct information/shipping pairs. Q00 randomly permutes two zero-shipping and two one-shipping uninformed consumers.
- Earlier orders determine queuing time; the next consumer's own shipping time determines their total wait. All 16 decision sequences were checked, including payoffs and singular/plural week wording.
- Correct F1 formation is selected by condition. N1 text and the copied F1_N1 CFCN weights match the original. The navigation sampling function is identical to the reference, and seeded assignments match NumPy weighted sampling. Weights are probabilities, not exact quotas.
- Fresh baseline validation requires no weights and renders no formation or navigation; decision facts and the answer request remain present.
- Default totals remain 425 Q50 groups (1,700 observations) and 200 Q00 groups (800 observations), divided exactly across execution sessions.

## Fixes made during audit

- Removed the fresh-baseline weights prerequisite.
- Matched the common product-value sentence across Q00 and Q50.
- Corrected cleaned output to mark arrival rank as known and retained queuing/shipping columns.
- Invalid classifier output now raises an error instead of becoming an artificial non-order or malformed decision.
- Main CLI exposes simulation stages and Q00/Q50 only; old copied training code remains inactive.
- Prompt preview now covers Q00/Q50 only.

## Scope and remaining limitations

This audit verifies local logic, not remote API availability or model behavior. The supplied instructions describe trial rounds and repeated-round history; the runner implements independent four-consumer groups, without trial-round simulation or cross-round history. Formation remains in both the initial instruction and the period prompt as currently configured; navigation appears in the period prompt. The active analysis now uses the copied Human_Results OLS construction; equivalence to the human script and synthetic-data exports are verified by test_regression.py. The old Kremer–Debo analysis scripts are inactive.

## Offline checks

Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s checks -v` from this folder.
