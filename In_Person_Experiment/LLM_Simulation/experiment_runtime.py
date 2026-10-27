"""Run-local overrides for the copied configuration and shared prompt plug-ins."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from simulation_counts import groups_for_session

MESSAGE_FORMAT = "two_user_messages_v1"


def manifest():
    path = os.environ.get('TLV_RUN_DIR')
    if not path:
        raise RuntimeError('Use pipeline.py to select a combination and stage.')
    return json.loads((Path(path) / 'manifest.json').read_text())


def selection():
    return manifest()['prompts']


def configure(namespace):
    m = manifest()
    run = Path(os.environ['TLV_RUN_DIR'])
    base = namespace['BASE_CONFIG']
    base.update(model_name=m['model'], assistant_model=m['assistant_model'],
                openai_base_url=m['base_url'], api_key_set=m['api_key_set'],
                weights_csv_path=str(run / 'training/weights.csv'),
                use_weighted_prompts=m['mode'] in ('cn', 'cfcn'))
    if 'REPLICATION_CONFIG_Q50' in namespace:
        namespace['REPLICATION_CONFIG_Q50'].update(base)
        namespace['REPLICATION_CONFIG_Q50'].update(
            csv_file=str(ROOT / 'training/kremer-debo_q50_uninformed.csv'),
            condition='Q50',
            output_folder=str(run / 'training/replication_results'))
    else:
        namespace['EXPERIMENT_CONFIG'].update(base)
        config = namespace['EXPERIMENT_CONFIG']
        condition = os.environ.get('TLV_CONDITION', 'Q50')
        config.update(condition=condition, n_total_cohort=m['cohorts'][condition],
                      round=m['rounds'], n_parallel_sessions=m['sessions'],
                      n_session_cohort=groups_for_session(m['cohorts'][condition], m['sessions'],
                                                         int(os.environ.get('TLV_SESSION', '1'))))
