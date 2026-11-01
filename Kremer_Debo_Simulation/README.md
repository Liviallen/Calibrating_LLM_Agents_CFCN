# Inferring quality from wait time: replication artifact

This framework simulates the Kremer–Debo experiment with language-model consumers.
It trains a navigation mixture on the 875 Q50 human observations, simulates Q00,
Q05 and Q50 with frozen weights, and compares the simulated and human results.

Only **condition-specific formation** is supported: Q00 has its own formation;
Q50 and Q05 share a formation. Edit `FORMATION_GROUPS` and `NAVIGATIONS` in
`prompt_inputs.py` before creating a new run. The same-formation and sampled/joint
training workflows have been removed. Numerical simulation, fitting and hypothesis
testing routines retain their existing behavior.

Formation and navigation content is configured only in `prompt_inputs.py` and
frozen in each run's manifest. Fixed experiment instructions and decision wording
live in `shared_prompts.py`, which training and validation import directly.
Validation samples navigation text from the checked `weights.csv`; the test stage
uses saved CSV results and does not construct prompts. Stage-local `prompts/`
wrappers are not needed and have been removed.

## Layout

```text
pipeline.py                 entry point for every stage
prompt_inputs.py            condition-specific formations and navigation sets
shared_prompts.py           common instruction and decision templates
stage_worker.py             training, validation, aggregation and analysis orchestration
experiment_runtime.py       settings from each run's frozen manifest
formation_different.py      formation validation and fitted-weight import
file_utils.py               atomic writes for weight import
env_loader.py, progress.py   environment loading and runtime progress
training/                   Q50 replication runner, fitting and human training CSV
validation/                 dynamic simulation and aggregation helper
test/                       Q05/Q50 probit analyses and original human comparison CSV
results/                    preserved simulations, fitted weights and analyses
results_vis.ipynb            existing, manually transcribed summary figure
results_vis.png, .pdf        existing figure exports
data/reference/             retained supplementary human datasets
checks/                     offline checks (mock models; no API requests)
docs/                       research notes, historical provenance and reorganization record
```

`test/` contains scientific analysis code; `checks/` contains software checks.
Supporting modules remain beside the entry point so commands work from any
working directory without installing this artifact as a package.

## Navigation labels

| Current label | Previous label | Candidates |
|---|---|---:|
| N1 | N2 | 8 |
| N2 | N6 | 8 |
| N1_12 | N2_12 | 4 |
| N1_13 | N2_13 | 4 |
| N1_23 | N2_23 | 4 |

Candidate labels `nav1`, `nav2`, etc. identify prompts *within* a set and have
not changed. Result folders, manifest navigation IDs and training-loss labels use
the new set names. Historical provenance documents retain their original names
and paths as a record of the source experiments.

## Setup and replicate

From this directory, install dependencies and put your key in `.env` (see
`.env.example`). Exported environment variables take precedence.

```bash
python -m pip install -r requirements.txt

# Inspect both formation branches without calling an API.
python pipeline.py preview --formation F1 --navigation N1 --run-id new-experiment

# Train all candidates and fit their weights.
python pipeline.py train --formation F1 --navigation N1 --run-id new-experiment

# Simulate all three conditions, then aggregate both comparisons.
python pipeline.py validate --formation F1 --navigation N1 --run-id new-experiment

# Save probit analyses and text reports.
python pipeline.py test --formation F1 --navigation N1 --run-id new-experiment
```

Outputs go to `results/F1_N1/cfcn/`. Use N2 for the second set.
The old `--formation-different` flag is no longer needed or accepted.
The pipeline reads and writes directly under each combination/mode folder.
Use `--run-id NAME` for a separate new run beneath the selected mode directory. The older `results/F1_N1/cfcn/run1/` is retained
for reference and is not selected by the pipeline.
Every stage must use the same settings. A manifest prevents mixing different
prompts or settings in an existing result directory. Archive that directory before starting a new
experiment with different settings.

Default models are `gpt5.4` and classifier `gpt4omini`, with local seed 42,
26 rounds, two sessions, and 17 Q50 / 8 Q00 / 14 Q05 cohorts. The retained
per-session formula is `round(cohorts * rounds / sessions)` independent markets,
each with four consumers. Local seeds do not control remote model randomness.
Training and validation call the API; the other stages do not.

Supported ablations are `--mode baseline`, `cf`, `cn`, and `cfcn` (default).
Baseline/CF use a single fixed weight; CN/CFCN fit the selected navigation mixture.
Baseline/CN omit formation; CF/CFCN use the condition-specific formation branches.

For individual tasks, add `--nav-idx 1` to `train`, then run `fit` after all
candidates finish. Add `--condition Q05 --session 1` to `validate`, then run
`aggregate` after all sessions finish. `test --comparison Q05` limits the analysis.
Other settings are listed by `python pipeline.py --help`.

To start validation with compatible full-Q50 weights, supply `--weights-csv`:

```bash
python pipeline.py validate --formation F1 --navigation N1 --run-id new-experiment \
  --weights-csv /path/to/compatible/weights.csv
```

Import checks prompt text, weight validity, and available source-manifest settings.
Weights are copied into the new run and their provenance is recorded. Imported
weights do not create a new training-loss estimate.

## Reading preserved results

```text
results/{formation}_{navigation}/{mode}/
  manifest.json                         frozen prompts and settings
  *_prompt_preview.txt                  saved prompt previews, where available
  training/
    replication_results/Q50/...         raw training CSVs and pickle records
    weights.csv                         fitted/imported mixture weights
    training_loss.csv                   loss and optimizer status, where available
    reused_weights.json                 imported-weight provenance, where available
  validation/
    weights/weights_{condition}_sim{s}/  raw simulation CSVs and pickle records
    Q50_Q00.csv, Q05_Q00.csv              aggregated comparisons
    Q50_Q00_clean.csv, Q05_Q00_clean.csv  inputs to probit analysis
  test/
    navigation_analysis_*.csv            hypothesis results, where available
    *_report.txt                        analysis reports, including failures
    h2_robustness/                       additional analyses for selected runs
```

All seven saved runs are retained, including baseline, earlier runs, subset
experiments and analyses that could not be estimated. The current prompt templates
may differ from those used in earlier runs: the saved manifest and raw prompt
columns are the record of what actually ran. Do not overwrite them to match a
new template. Recorded optimizer warnings and unsuccessful analyses remain intact.

`results_vis.ipynb` uses manually transcribed values; it does not automatically
recompute the results. Its existing PNG/PDF exports are preserved unchanged.
Supplementary human datasets were moved to `data/reference/` rather than deleted.
The active training and comparison CSVs remain beside their respective code.

Training resumes from checkpoints. Validation checks completed session CSVs before skipping them. Incomplete session artifacts are preserved and require a new run ID; they are not automatically restarted. Runtime progress may generate new `progress/` files;
historical logs and progress snapshots have been removed from the artifact.

## Offline verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s checks -v
```

The included safeguard checks exercise atomic output, session integrity, output
protection, run IDs, and generation failure handling without API calls.


## Output safeguards

Existing scientific outputs are preserved. Training/fitting refuses to replace
fitted weights or run after validation results exist. Preview, aggregation, and
analysis refuse to replace their existing outputs. Completed validation CSVs are
checked for the expected row count, binary choices, and four unique ranks per
market before reuse. A malformed CSV or an incomplete session directory causes
an explicit error; it is not automatically regenerated. These checks do not
reinterpret model responses or change decisions, fitting, or statistical methods.

For new work, keep the archived results and select a separate run:

```bash
python pipeline.py preview --mode cfcn --run-id new-experiment
```

Use the same `--run-id` and settings for subsequent stages. Named runs live under
`results/.../{mode}/runs/{run-id}/` (the existing mode path stays unchanged).
The ID is a directory selector, not a random seed. Original-study CFCN runs still
require training or an explicit compatible weights import; in-person N1 retains
its existing default imported weights. A new run does not copy old simulation
outputs. There is deliberately no force-overwrite option. A full validation call
preserves already existing aggregate outputs instead of recomputing them.

CSV outputs and pickle checkpoints are written to a temporary file in the same
directory and atomically replaced only after a successful write. A write failure
leaves the preceding file intact. This is per-file protection, not a transaction
across an entire multi-file stage or support for concurrent writers to one run.
Use distinct run IDs for concurrent experiments. An interrupted multi-file stage
may leave some completed files; preserve them and use a new run for a clean restart.

Run the offline safeguards with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s checks -v
```
