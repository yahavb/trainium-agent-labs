"""Experiment preparation, guard and aggregation; subprocess inference is mocked."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_controlled


class RunnerTests(unittest.TestCase):
    def test_commands_share_settings_and_have_private_files(self):
        options=SimpleNamespace(rounds=2,samples=4,repeat=1)
        commands=[run_controlled.build_command(options,arm,1,Path('/private')/arm[0]) for arm in run_controlled.ARMS]
        for command in commands:
            for flag,value in [('--samples','4'),('--context','8192'),('--rounds','2'),('--max-tokens','2500')]:
                self.assertEqual(command[command.index(flag)+1],value)
            self.assertIn('--instrument',command)
            self.assertTrue(command[command.index('--grade-dir')+1].startswith('/private/'))
        self.assertEqual(len({command[command.index('--log')+1] for command in commands}),4)

    def test_baseline_guard_prevents_process_launch(self):
        with patch.object(sys,'argv',['run_controlled.py','--run']), \
             patch.object(run_controlled,'baseline_processes',return_value=[{'pid':14656}]), \
             patch.object(run_controlled.subprocess,'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_controlled.main(),2)
        run.assert_not_called()

    def test_prepare_is_unique_and_never_runs_inference(self):
        with tempfile.TemporaryDirectory(prefix='nki-plan-unit-') as private:
            with patch.object(sys,'argv',['run_controlled.py','--output-root',private]), \
                 patch.object(run_controlled,'baseline_processes',return_value=[]), \
                 patch.object(run_controlled.subprocess,'check_output',return_value='8f1ca41\n'), \
                 patch.object(run_controlled.subprocess,'run') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_controlled.main(),0)
                self.assertEqual(run_controlled.main(),0)
            run.assert_not_called()
            manifests=list(Path(private).glob('*/manifest.json'))
            self.assertEqual(len(manifests),2)
            self.assertEqual(len(json.loads(manifests[0].read_text())['plan']),8)
            manifest=json.loads(manifests[0].read_text())
            self.assertTrue((manifests[0].parent/'source/agent.py').exists())
            self.assertIn('/source/agent.py',manifest['plan'][0]['command'][3])

    def test_aggregation_counts_each_round_once_and_censors_failure(self):
        def row(round,index,reward,usage):
            return dict(run_id='run',repeat_index=0,level=1,round=round,candidate_index=index,reward=reward,
                        selected=index==0,diversity=dict(unique_ast_sources=2,duplicate_fraction=.5),round_seconds=3.,
                        failure_category='DMA_SHAPE_MISMATCH',normalized_signature='dma',prompt_tokens=usage,
                        completion_tokens=usage,total_tokens=usage)
        rows=[row(round,index,.3,10) for round in range(2) for index in range(4)]
        result=run_controlled.summarize(rows)['trials'][0]
        self.assertEqual(result['round_seconds'],6.)
        self.assertEqual(result['total_tokens'],80)
        self.assertIsNone(result['rounds_to_success'])
        rows[4]['reward']=1.
        result=run_controlled.summarize(rows)['trials'][0]
        self.assertEqual(result['rounds_to_success'],2)
        self.assertEqual(result['candidates_to_success'],8)
        rows[1]['total_tokens']=None
        self.assertIsNone(run_controlled.summarize(rows)['trials'][0]['total_tokens'])


if __name__=='__main__':unittest.main()
