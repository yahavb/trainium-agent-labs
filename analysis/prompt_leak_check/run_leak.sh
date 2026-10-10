#!/bin/bash
# feedback_v7 with V7.md's configuration (minus the compile step) against the leak-check mock
cd /h/tal-deliv/projects/02-kernel-agent
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2 KERNEL_AGENT_BASE_URL=http://127.0.0.1:8010/v1 PYTHONPATH=.:check
export MESSAGES=v5 CARD=category PROMPT1=v2 SAMPLING=qwen REPAIR_PROMPT=restructure GRADE_TIMEOUT=120 \
       GATE=static NKI_VERDICTS=/tmp/leak_nkiv.jsonl USAGE_LOG=/tmp/leak_usage.jsonl
/root/venvs/nki/bin/python feedback_v7.py --all --rounds 4 --samples 2 --context 8192 --log /tmp/leak_att.jsonl --verdicts /tmp/leak_ver.jsonl > /tmp/leak_all.txt 2>&1; echo "all exit=$?"
for l in 9 10 11 12 13 14; do
  /root/venvs/nki/bin/python feedback_v7.py --level $l --rounds 4 --samples 2 --context 8192 --log /tmp/leak_att.jsonl --verdicts /tmp/leak_ver.jsonl > /tmp/leak_L$l.txt 2>&1; echo "L$l exit=$?"
done
