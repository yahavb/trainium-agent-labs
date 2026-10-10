#!/usr/bin/env python3
"""Matched B/C controller, opt-in inference only. Canonical solve/ask stay intact."""

import argparse
import hashlib
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from targeted_feedback import base, evaluate, guard, verify_inputs
from instrumentation import Instrumentation, utc
from portable import PROJECT, load_agent


class TargetedInstrumentation(Instrumentation):
    def __init__(self, agent, enabled, requests, grades, policy='curriculum'):
        super().__init__(agent,policy,requests,grades)
        self.enabled = enabled

    def parallel(self, args, prompt, n):
        guard()
        return super().parallel(args,prompt,n)

    def grade(self, source, level):
        guard()
        self.sample += 1
        result,details = evaluate(self.agent,source,level,self.enabled,self.policy)
        row = dict(self.context,round=self.round-1,sample=self.sample,
                   arm='C' if self.enabled else 'B',code_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   reward=result[0],parts=result[1],feedback=result[2],observed_at_utc=utc(),**details)
        self.append(self.grades,row)
        return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,description='Canonical kernel agent with optional curriculum and targeted feedback. Original agent flags are forwarded unchanged.')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--free-dimension-guidance',action='store_true')
    parser.add_argument('--feedback-policy',choices=('official','curriculum'),default='curriculum')
    parser.add_argument('--approved-live',action='store_true')
    wrapper,args = parser.parse_known_args()
    if not wrapper.approved_live:
        parser.error('Live B/C inference requires approval and --approved-live')
    if '--offline' in args or any(arg=='--log' or arg.startswith('--log=') for arg in args):
        parser.error('The wrapper owns logs; use parity_arm_c.py for offline validation')
    output = wrapper.output_dir.resolve()
    project = PROJECT
    with base.exclusive_execution():
        guard()
        integrity = verify_inputs()
        output.mkdir(parents=True,exist_ok=False)
        os.chdir(project)
        agent = load_agent()
        attempts = output/'attempts.jsonl'
        attempts.touch(exist_ok=False)
        base.write_json(output/'execution.json',
                        {'arm':'C' if wrapper.free_dimension_guidance else 'B',
                         'guidance_enabled':wrapper.free_dimension_guidance,'canonical_args':args,
                         'feedback_policy':wrapper.feedback_policy,
                         'started_at_utc':utc(),'integrity_before':integrity})
        with (output/'requests.jsonl').open('x') as requests,(output/'grades.jsonl').open('x') as grades:
            instrumentation = TargetedInstrumentation(agent,wrapper.free_dimension_guidance,requests,grades,wrapper.feedback_policy)
            originals = instrumentation.install()
            agent._canonical_grade = originals[0]
            argv = sys.argv
            sys.argv = [str(project/'agent.py'),*args,'--log',str(attempts)]
            try:
                agent.main()
            finally:
                sys.argv = argv
                instrumentation.restore(originals)
                base.write_json(output/'integrity_after.json',
                                {'integrity':verify_inputs(),'ended_at_utc':utc()})


if __name__ == '__main__':
    main()
