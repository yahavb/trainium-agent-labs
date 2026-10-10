while pgrep -f "bench-L123-r5.jsonl" > /dev/null; do sleep 30; done
python3 -u -m kagent.agent --mode naive --levels 1 2 3 --attempts 8 --repeat 5 --out runs/naive-L123-r5.jsonl > runs/naive-L123-r5.log 2>&1
