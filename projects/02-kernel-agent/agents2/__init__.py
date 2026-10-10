"""Kernel agent v2: six roles around one ledger. See ../DESIGN.md.

    manager    threads, termination, routing            manager.py   (code)
    planner    picks the algorithm before any code      planner.py   (model)
    retriever  pulls the docs from the installed nki    retriever.py (code)
    coder      writes, repairs and improves kernels     coder.py     (model)
    debugger   one named change for a failed check      debugger.py  (rules, then model)
    reviewer   accepts a correct kernel, or improves it reviewer.py  (rules, then model)

Shared plumbing: config.py (settings), llm.py (client, token counts, prompt packer), checks.py (lint,
rules, simulate, compare, in a process pool), errors.py (error kinds and rule templates), ledger.py
(what was tried), events.py (the log).

Every model call is one fresh single-turn prompt built from the ledger, so an 8K context is a limit
per call, never on the run as a whole. agent.py is left untouched as the baseline.
"""
