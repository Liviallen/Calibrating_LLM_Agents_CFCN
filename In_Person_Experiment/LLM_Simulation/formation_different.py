"""Condition-specific formation groups and optional reuse of full-Q50 weights."""
import csv
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path

from file_utils import atomic_write


def resolve_formation_group(groups, name, enabled):
    group = groups[name]
    if set(group) != {'Q50_Q05', 'Q00'}:
        raise ValueError('Each FORMATION_GROUPS entry requires exactly Q50_Q05 and Q00.')
    if any(not isinstance(value, str) for value in group.values()):
        raise ValueError('Both formation-group entries must be strings.')
    if enabled and any(not value.strip() for value in group.values()):
        raise ValueError('Fill in both condition-specific formation strings first.')
    q50 = group['Q50_Q05'] if enabled else ''
    q00 = group['Q00'] if enabled else ''
    return q50, dict(Q50=q50, Q05=q50, Q00=q00)


def _validate_weights(data, manifest):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    required = {'prompt_variant', 'optimal_weight', 'prompt_text'}
    if not required <= set(reader.fieldnames or []):
        raise ValueError('Weights CSV must include prompt_variant, optimal_weight and prompt_text.')
    rows = list(reader)
    prompts = manifest['prompts']['navigations']
    expected = {f'nav{i+1}': prompt for i, prompt in enumerate(prompts)}
    if len(rows) != len(expected) or {row['prompt_variant'] for row in rows} != set(expected):
        raise ValueError('Weights CSV navigation variants do not match the selected navigation set.')
    weights = []
    for row in rows:
        if row['prompt_text'] != expected[row['prompt_variant']]:
            raise ValueError(f"Weights CSV prompt text differs for {row['prompt_variant']}.")
        weight = float(row['optimal_weight'])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Weights must be finite and nonnegative.')
        weights.append(weight)
    if not math.isfinite(sum(weights)) or sum(weights) <= 0:
        raise ValueError('Weights must have positive total weight.')


def _validate_source_manifest(source, manifest):
    source_manifest = source.parent.parent / 'manifest.json'
    if not source_manifest.is_file():
        return  # A standalone CSV can be imported, but its training provenance is unknown.
    original = json.loads(source_manifest.read_text())
    if original.get('training_sample') is not None:
        raise ValueError('Reuse weights from full-Q50 training, not a sampled or joint-training run.')
    for key in ('model', 'mode', 'message_format'):
        if original.get(key) != manifest.get(key):
            raise ValueError(f'Reused weights have a different {key}.')
    if original['prompts']['formation'] != manifest['prompts']['formation']:
        raise ValueError('Reused weights require the same Q50 formation as their training run.')


def prepare_reused_weights(run, manifest, source=None):
    """Copy weights locally once, or accept an existing local CSV, with checks."""
    run = Path(run)
    target = run / 'training/weights.csv'
    candidate = Path(source).expanduser().resolve() if source else None
    if candidate is None:
        candidate = next((p for p in [target, run/'weights.csv', run/'weight.csv'] if p.is_file()), None)
    if candidate is None:
        return False
    data = candidate.read_bytes()
    _validate_weights(data, manifest)
    _validate_source_manifest(candidate, manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    provenance = target.parent/'reused_weights.json'
    with (target.parent/'.weights_import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        digest = hashlib.sha256(data).hexdigest()
        if provenance.exists():
            saved = json.loads(provenance.read_text())
            if saved['sha256'] != digest:
                raise ValueError('This run already uses different weights. Archive the existing result directory before using different weights.')
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Local weights differ from the supplied weights. Archive the existing result directory before using different weights.')
        if candidate != target:
            atomic_write(target, data)
        if candidate != target and not provenance.exists():
            metadata = dict(source=str(candidate), sha256=digest,
                            source_manifest_checked=(candidate.parent.parent/'manifest.json').is_file())
            atomic_write(provenance, (json.dumps(metadata, indent=2)+'\n').encode())
    return True
