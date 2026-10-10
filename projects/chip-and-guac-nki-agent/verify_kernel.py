"""Call the original checker directly, with private candidate-file I/O only."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode=True
from curriculum import exclusive_execution
from portable import load_agent,verify_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel',type=Path)
    parser.add_argument('--level',type=int,default=4)
    args=parser.parse_args()
    source=args.kernel.read_text()
    with exclusive_execution(),tempfile.TemporaryDirectory(prefix='chip-guac-original-check-') as folder:
        agent=load_agent()
        original=agent.grade
        code=original.__code__
        logical=f'/tmp/_agent_level{args.level}.py'
        physical=Path(folder)/'candidate.py'
        builtin_open=open
        loader=agent.nkibench.load_kernel
        def candidate_open(path,*rest,**kwargs):
            return builtin_open(physical if os.fspath(path)==logical else path,*rest,**kwargs)
        def candidate_load(path,entry):
            return loader(physical if os.fspath(path)==logical else path,entry)
        agent.open=candidate_open
        agent.nkibench.load_kernel=candidate_load
        try:reward,parts,feedback=original(source,args.level)
        finally:
            agent.__dict__.pop('open',None)
            agent.nkibench.load_kernel=loader
        assert agent.grade is original and original.__code__ is code
        verify_sources()
        print(json.dumps({'level':args.level,'reward':reward,'parts':parts,
                         'feedback':feedback.replace(str(physical),logical),
                         'original_grade_called_directly':True,'candidate_io_only_redirected':True},indent=2))
        if not parts['correct']:raise SystemExit(1)


if __name__=='__main__':main()
