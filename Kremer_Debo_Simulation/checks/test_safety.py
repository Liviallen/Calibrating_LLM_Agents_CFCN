"""Offline safeguards: temporary files only, no model calls."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import file_utils as io
import pandas as pd


class SafetyTests(unittest.TestCase):
    def test_atomic_failure_keeps_original_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.csv'
            path.write_bytes(b'original')
            with self.assertRaises(RuntimeError):
                with io.atomic_output(path) as stream:
                    stream.write('partial')
                    raise RuntimeError('interrupted')
            self.assertEqual(path.read_bytes(), b'original')
            self.assertEqual(list(Path(folder).iterdir()), [path])
            with patch.object(io.os, 'replace', side_effect=OSError('replacement failed')):
                with self.assertRaises(OSError):
                    io.atomic_write(path, b'new')
            self.assertEqual(path.read_bytes(), b'original')
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_atomic_serialization_matches_previous_bytes(self):
        import pickle
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'output'
            frame = pd.DataFrame({'text': ['a, b', 'multi\nline'], 'number': [1.25, 0.]})
            reference = Path(folder) / 'reference'
            frame.to_csv(reference, index=False)
            io.atomic_to_csv(frame, path, index=False)
            self.assertEqual(path.read_bytes(), reference.read_bytes())
            record = {(1, 1): {'choice': '1'}}
            with io.atomic_output(path, 'wb') as stream:
                pickle.dump(record, stream)
            self.assertEqual(path.read_bytes(), pickle.dumps(record))

    def test_sessions_are_checked_without_modification(self):
        good = pd.DataFrame(dict(cohort_id=[1]*4, subject_rank=[1,2,3,4], subject_choice=[0,1,0,1]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'session.csv'
            for frame, valid in [(good, True), (good.iloc[:3], False),
                                 (good.assign(subject_choice=[0,1,None,1]), False),
                                 (good.assign(subject_rank=[1,2,2,4]), False),
                                 (good.drop(columns='subject_choice'), False)]:
                frame.to_csv(path, index=False)
                before = path.read_bytes()
                if valid:
                    pd.testing.assert_frame_equal(io.validate_session_csv(path,4),good)
                else:
                    with self.assertRaisesRegex(ValueError, 'no automatic regeneration'):
                        io.validate_session_csv(path,4)
                self.assertEqual(path.read_bytes(),before)

    def test_protected_stages_and_new_run(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            for stage, name in [('train','training/weights.csv'),('fit','training/weights.csv'),
                                ('aggregate','validation/Q50_Q00.csv'),('test','test/Q50_Q00_report.txt'),
                                ('preview','Q50_prompt_preview.txt')]:
                path=run/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('archived')
                with self.assertRaisesRegex(ValueError,'Refusing to overwrite'):
                    io.protect_stage(run,stage)
                io.protect_stage(run/'runs'/'new',stage)
                self.assertEqual(path.read_text(),'archived')

    def test_generation_failure_sentinels_including_cached_values(self):
        repository = ROOT.parent if ROOT.name=='Kremer_Debo_Simulation' else ROOT.parents[1]
        notebook=json.loads((repository/'CFCN_Prompt_Generation/formation_generation.ipynb').read_text())
        source=next(''.join(cell['source']) for cell in notebook['cells'] if 'def stage(' in ''.join(cell['source']))
        ns={'Path':Path,'ask':lambda *a,**k: self.fail('must not call API')}
        exec(compile(source,'generation_stage','exec'),ns)
        for sentinel in ns['FAILURE_SENTINELS']:
            with self.assertRaises(ValueError):ns['validate_stage_result'](sentinel)
            with tempfile.TemporaryDirectory() as folder:
                ns['run_dir']=Path(folder);(Path(folder)/'cached.txt').write_text(sentinel)
                with self.assertRaises(ValueError):ns['stage']('cached','unused','unused')
        self.assertEqual(ns['validate_stage_result']('valid prompt'),'valid prompt')

    def test_validation_skips_valid_and_rejects_invalid_without_api(self):
        import stage_worker as worker
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            output=run/'validation/weights/weights_Q00_sim1/weights_Q00_sim1.csv'
            output.parent.mkdir(parents=True)
            frame=pd.DataFrame(dict(cohort_id=[1]*4,subject_rank=[1,2,3,4],subject_choice=[0,1,0,1]))
            config=dict(mode='baseline',cohorts={'Q00':1},rounds=1,sessions=1)
            with patch.object(worker,'run_dir',return_value=run), patch.object(worker,'manifest',return_value=config), patch.object(worker.subprocess,'run') as calls:
                frame.to_csv(output,index=False)
                before=output.read_bytes()
                worker.validate('Q00',1)
                self.assertEqual(output.read_bytes(),before)
                frame.iloc[:3].to_csv(output,index=False)
                with self.assertRaisesRegex(ValueError,'Invalid completed session'):
                    worker.validate('Q00',1)
                output.unlink()
                (output.parent/'checkpoint.pkl').write_bytes(b'checkpoint')
                with self.assertRaisesRegex(ValueError,'Incomplete session artifacts'):
                    worker.validate('Q00',1)
                calls.assert_not_called()

    def test_run_id_separates_outputs(self):
        import argparse
        import pipeline
        args=argparse.Namespace(formation='F1',navigation='N1',mode='baseline',stage='preview',
            weights_csv=None,model='gpt5.4',assistant_model='gpt4omini',base_url='https://api.openai.com/v1',
            api_key_set=1,seed=42,q50_cohorts=17,q00_cohorts=8,q05_cohorts=14,rounds=26,sessions=2,run_id='safe-run')
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline,'ROOT',Path(folder)):
            run=pipeline.resolve_run(args)
            self.assertEqual(run.parts[-2:],('runs','safe-run'))
            args.run_id='../outside'
            with self.assertRaises(ValueError):pipeline.resolve_run(args)

if __name__=='__main__':unittest.main()
