# usage: run.sh OUTFILE "model variant" "model variant" ...
cd /workspace/infer && export PATH=$PWD/env/bin:$PATH
out=$1; shift; : > $out
for job in "$@"; do
  set -- $job
  line=$(timeout 2400 python model_infer.py "$1" "$2" 50 2>/dev/null | grep -E '^\{' | tail -1)
  [ -z "$line" ] && line="{\"model\":\"$1\",\"variant\":\"$2\",\"error\":\"no output\"}"
  neff=$(echo "$line" | python -c 'import sys,json; print(json.load(sys.stdin).get("neff",""))')
  prof="{}"; [ -n "$neff" ] && prof=$(timeout 900 python profile_neff.py "$neff" 3 2>/dev/null | tail -1)
  echo "{\"run\":$line,\"profile\":${prof:-{\}}}" >> $out
done
echo DONE >> $out
