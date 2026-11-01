"""Atomic output and structural validation; no simulation or statistical rules."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_output(path, mode="w", **kwargs):
    """Replace one file only after successful writing; preserve it on failure."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **kwargs) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write(path, data):
    with atomic_output(path, "wb") as stream:
        stream.write(data)


def atomic_to_csv(frame, path, **kwargs):
    with atomic_output(path, "w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, **kwargs)


def validate_session_csv(path, expected_rows):
    """Reject incomplete sessions before reuse. Never repair or regenerate them."""
    import pandas as pd
    try:
        frame = pd.read_csv(path)
        required = {"cohort_id", "subject_rank", "subject_choice"}
        if not required <= set(frame.columns) or len(frame) != expected_rows or expected_rows <= 0:
            raise ValueError("missing fields or unexpected row count")
        if not frame.subject_choice.isin([0, 1]).all():
            raise ValueError("missing or nonbinary decisions")
        if expected_rows % 4 or set(frame.cohort_id) != set(range(1, expected_rows // 4 + 1)):
            raise ValueError("missing or invalid market identifiers")
        if frame.duplicated(["cohort_id", "subject_rank"]).any():
            raise ValueError("duplicate consumer ranks")
        if not frame.groupby("cohort_id").subject_rank.apply(lambda ranks: sorted(ranks) == [1, 2, 3, 4]).all():
            raise ValueError("incomplete market")
    except (ValueError, OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Invalid completed session: {path}. Preserve this file and use a new --run-id; no automatic regeneration. ({error})") from error
    return frame


def protect_stage(run, stage):
    """Refuse to replace scientific outputs; use a separate run for new work."""
    run = Path(run)
    protected = []
    if stage in ("train", "fit"):
        protected = [run / "training/weights.csv", run / "training/training_loss.csv"]
        protected += list((run / "validation").rglob("*.csv"))
    elif stage == "aggregate":
        protected = list((run / "validation").glob("*.csv"))
    elif stage == "test":
        protected = list((run / "test").glob("*.csv")) + list((run / "test").glob("*_report.txt"))
        protected += list(run.glob("*comparison.csv")) + list(run.glob("joint_h1_h2_*"))
        protected.append(run / "regression_audit.json")
    elif stage == "preview":
        protected = list(run.glob("*_prompt_preview.txt"))
    existing = next((path for path in protected if path.exists()), None)
    if existing is not None:
        raise ValueError(f"Refusing to overwrite archived output: {existing}. Use a new --run-id for new work; existing results were not changed.")
