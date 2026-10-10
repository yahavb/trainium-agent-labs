"""Telemetry with mocked HTTP/checker. Grading writes only private temporary files."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import agent
from experiment_metrics import CASE_RESULTS, GRADE_DIRECTORY, ModelReply, response_metadata


class MetricsTests(unittest.TestCase):
    def test_reported_usage_and_unavailable_values(self):
        payload=dict(usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30),choices=[dict(finish_reason='length')])
        result=response_metadata(payload,1.2)
        self.assertEqual(result['prompt_tokens'],10)
        self.assertEqual(result['total_tokens'],30)
        self.assertTrue(result['truncated'])
        self.assertIsNone(response_metadata({})['prompt_tokens'])
        self.assertIsNone(response_metadata({})['truncated'])
        self.assertIsNone(response_metadata(dict(usage={'prompt_tokens':True}))['prompt_tokens'])
        self.assertIsNone(response_metadata(dict(usage={'prompt_tokens':-1}))['prompt_tokens'])

    def test_http_configuration_preserved_and_no_extra_requests(self):
        payload=dict(usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30),
                     choices=[dict(message={'content':'code'},finish_reason='stop')])
        response=SimpleNamespace(status_code=200,json=lambda:payload)
        options=SimpleNamespace(context=8192,max_tokens=2500,model=agent.MODEL,think=False,base='http://localhost:8000/v1')
        bodies=[]
        for instrument in (False,True):
            options.instrument=instrument
            with patch('httpx.post',return_value=response) as http:
                reply=agent.ask(options,'prompt')
            self.assertEqual(http.call_count,1)
            bodies.append(http.call_args.kwargs['json'])
            self.assertEqual(reply,'code')
            if instrument:
                self.assertIsInstance(reply,ModelReply)
                self.assertEqual(reply.metadata['total_tokens'],30)
                self.assertFalse(reply.metadata['truncated'])
            else: self.assertIs(type(reply),str)
        self.assertEqual(bodies[0],bodies[1])
        self.assertEqual(bodies[0]['temperature'],.6)
        self.assertEqual(bodies[0]['top_p'],.95)

    def test_per_shape_logging_and_reward_parity_with_mocked_checker(self):
        cases=[dict(id='good'),dict(id='numerical'),dict(id='hazard')]
        spec=dict(entry='kernel',ref=lambda x:x.copy(),shapes=cases)
        outputs=[(np.ones((2,2)),{'bytes':0,'warnings':[]}),
                 (np.full((2,2),2.),{'bytes':0,'warnings':[]}),
                 (np.ones((2,2)),{'bytes':0,'warnings':['incorrect results on hardware']})]
        source='def kernel(a):\n return a'
        with tempfile.TemporaryDirectory(prefix='nki-metric-unit-') as private:
            directory_token=GRADE_DIRECTORY.set(private)
            try:
                results=[]
                for instrument in (False,True):
                    details=[]
                    token=CASE_RESULTS.set(details if instrument else None)
                    try:
                        with patch.dict(agent.nkibench.LEVELS,{99:spec}), \
                             patch.object(agent.nkibench,'check_rules',return_value=[]), \
                             patch.object(agent.nkibench,'load_kernel',return_value=object()), \
                             patch.object(agent.nkibench,'label',side_effect=lambda c,l:c['id']), \
                             patch.object(agent.nkibench,'make_inputs',side_effect=lambda c,l:([np.ones((2,2))],None)), \
                             patch.object(agent.nkibench,'simulate_and_count',side_effect=outputs), \
                             patch.object(agent.nkibench,'check_traffic_bar',return_value=None):
                            results.append(agent.grade(source,99))
                    finally: CASE_RESULTS.reset(token)
                    if instrument:
                        self.assertEqual([r['numerically_correct'] for r in details],[True,False,True])
                        self.assertEqual([r['verified'] for r in details],[True,False,False])
                self.assertEqual(results[0],results[1])
                self.assertTrue((Path(private)/'_agent_level99.py').exists())
            finally: GRADE_DIRECTORY.reset(directory_token)

    def test_instrumented_standard_log_has_all_metadata_without_prompt_changes(self):
        options=SimpleNamespace(terse=0,rounds=1,samples=4,offline=False,give_up_after=4,
                                instrument=True,run_id='unit',repeat_index=0)
        metadata=response_metadata(dict(usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30),
                                        choices=[dict(finish_reason='stop')]),1.)
        reply=ModelReply('```python\ndef kernel(): pass\n```',metadata)
        log=io.StringIO()
        with patch.object(agent,'ask_parallel',return_value=[reply]*4) as ask, \
             patch.object(agent,'grade',return_value=(.3,{},'No code came back.')), \
             contextlib.redirect_stdout(io.StringIO()):
            agent.solve(options,1,log)
        self.assertEqual(ask.call_args.args[1],agent.first_prompt(1))
        rows=[json.loads(line) for line in log.getvalue().splitlines()]
        row=rows[0]
        self.assertEqual(row['total_tokens'],30)
        self.assertEqual(row['run_id'],'unit')
        self.assertEqual(row['repeat_index'],0)
        self.assertEqual(row['candidate_policy'],'standard')
        self.assertEqual(row['repair_policy'],'standard')
        self.assertEqual(row['selection_policy'],'reward')
        self.assertEqual(row['prompt'],agent.first_prompt(1))
        self.assertEqual(row['diversity']['duplicate_fraction'],.75)
        self.assertEqual(len(row['shape_results']),4)
        self.assertTrue(all(not c['evaluated'] for c in row['shape_results']))
        self.assertGreaterEqual(row['round_seconds'],0.)


if __name__ == '__main__': unittest.main()
