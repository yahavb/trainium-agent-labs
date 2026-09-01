#!/usr/bin/env python3
"""
web.py — the same chat, in a browser. Zero dependencies beyond httpx.

    export GPTOSS_BASE_URL="https://<the-load-balancer>"
    python web.py            # then open http://localhost:8080

Why this exists: a browser tab is the fastest way to show a demo to a judge, and
this file is small enough to read in one sitting and modify during the event. It is
a dumb streaming proxy plus one HTML page — no framework, no build step, no npm.
"""

import argparse
import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import httpx

warnings.filterwarnings("ignore")

MODEL = "gpt-oss-20b"
BASE = os.environ.get("GPTOSS_BASE_URL", "")
# gpt-oss streams its chain of thought on a separate channel. This vLLM build names the
# field `reasoning`; other builds name it `reasoning_content`. Accept either.
REASONING_KEYS = ("reasoning", "reasoning_content")

PAGE = """<!doctype html>
<meta charset=utf-8><title>gpt-oss-20b on Trainium</title>
<style>
 :root{color-scheme:dark}
 body{background:#111418;color:#e6e6e6;font:15px/1.55 ui-sans-serif,system-ui,sans-serif;
      margin:0;display:flex;flex-direction:column;height:100vh}
 header{padding:10px 16px;border-bottom:1px solid #262c33;display:flex;gap:14px;
        align-items:center;flex-wrap:wrap;font-size:13px}
 header b{font-size:14px}
 select,button,input{background:#1b2027;color:#e6e6e6;border:1px solid #333c46;
        border-radius:6px;padding:6px 9px;font:inherit;font-size:13px}
 button{cursor:pointer}
 #log{flex:1;overflow-y:auto;padding:18px 16px;max-width:900px;margin:0 auto;width:100%}
 .msg{margin:0 0 18px}
 .who{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#7d8896;margin-bottom:5px}
 .you .who{color:#5fb3d4}
 .body{white-space:pre-wrap;word-wrap:break-word}
 .meta{font-size:11px;color:#6d7681;margin-top:6px;font-variant-numeric:tabular-nums}
 .think{color:#6d7681;font-size:13px;white-space:pre-wrap;border-left:2px solid #2c343d;
        padding-left:10px;margin:6px 0;display:none}
 .think.on{display:block}
 form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid #262c33;
      max-width:900px;margin:0 auto;width:100%;box-sizing:border-box}
 #q{flex:1}
 .err{color:#e06c6c}
</style>
<header>
  <b>gpt-oss-20b</b><span style=color:#7d8896>on AWS Trainium</span>
  <select id=ep><option value=agg>aggregated (TP32, 1 server)</option>
                <option value=disagg>disaggregated (prefill+decode, TP16)</option></select>
  <select id=effort><option>low</option><option selected>medium</option><option>high</option></select>
  <label style=color:#7d8896><input type=checkbox id=showthink> show reasoning</label>
  <button id=clear type=button>new conversation</button>
  <span id=running style=color:#7d8896></span>
</header>
<div id=log></div>
<form id=f><input id=q placeholder="Ask something…" autocomplete=off autofocus>
  <button>send</button></form>
<script>
let history=[];
const log=document.getElementById('log'), f=document.getElementById('f'),
      q=document.getElementById('q');

function add(who,cls){
  const d=document.createElement('div'); d.className='msg '+(cls||'');
  d.innerHTML='<div class=who></div><div class=think></div><div class=body></div><div class=meta></div>';
  d.querySelector('.who').textContent=who; log.appendChild(d);
  log.scrollTop=log.scrollHeight; return d;
}
document.getElementById('clear').onclick=()=>{history=[];log.innerHTML='';q.focus()};

f.onsubmit=async e=>{
  e.preventDefault();
  const text=q.value.trim(); if(!text) return;
  q.value=''; add('you','you').querySelector('.body').textContent=text;
  history.push({role:'user',content:text});

  const el=add('gpt-oss-20b'), body=el.querySelector('.body'),
        meta=el.querySelector('.meta'), think=el.querySelector('.think');
  if(document.getElementById('showthink').checked) think.classList.add('on');
  document.getElementById('running').textContent='thinking…';
  const t0=performance.now(); let ttft=null, out='';

  let r;
  try{
    r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:history,endpoint:document.getElementById('ep').value,
                           effort:document.getElementById('effort').value})});
  }catch(err){ body.className='body err'; body.textContent=String(err);
               document.getElementById('running').textContent=''; return; }

  const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
  while(true){
    const {done,value}=await rd.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    const lines=buf.split('\\n'); buf=lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: ')) continue;
      const ev=JSON.parse(line.slice(6));
      if(ev.error){ body.className='body err'; body.textContent=ev.error; continue; }
      if(ev.think) think.textContent+=ev.think;
      if(ev.text){ if(ttft===null) ttft=(performance.now()-t0)/1000;
                   out+=ev.text; body.textContent=out; }
      if(ev.done){
        const dt=(performance.now()-t0)/1000;
        meta.textContent=`${dt.toFixed(1)}s · ttft ${ttft?ttft.toFixed(1):'–'}s · `+
          `${ev.out_tok?(ev.out_tok/dt).toFixed(0):'?'} tok/s · in ${ev.in_tok??'?'} · `+
          `out ${ev.out_tok??'?'} · ${ev.fingerprint||''}`+(ev.truncated?' · TRUNCATED':'');
      }
      log.scrollTop=log.scrollHeight;
    }
  }
  if(!out && think.textContent) body.textContent='(only hidden reasoning was produced — try effort=low)';
  history.push({role:'assistant',content:out});
  document.getElementById('running').textContent='';
};
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req.get("messages", [])
        path = f"/{req.get('endpoint', 'agg')}/v1/chat/completions"
        system = ("You are a concise, accurate assistant.\n"
                  f"Reasoning: {req.get('effort', 'medium')}")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        # Trim history to fit: max_model_len is 8192 for prompt + completion together,
        # and there is no prefix caching, so old turns cost full price every time.
        budget, kept, used = 6000, [], 0
        for m in reversed(msgs):
            t = max(1, len(m["content"]) // 4)
            if used + t > budget:
                break
            kept.append(m)
            used += t
        kept.reverse()

        body = {"model": MODEL,
                "messages": [{"role": "system", "content": system}] + kept,
                "max_tokens": max(256, min(2048, 8192 - used - 256)),
                "stream": True}

        in_tok = out_tok = None
        fingerprint, finish = None, None
        try:
            with httpx.Client(base_url=BASE, verify=False,
                              timeout=httpx.Timeout(900.0, connect=20.0)) as c:
                with c.stream("POST", path, json=body) as r:
                    if r.status_code != 200:
                        emit({"error": f"HTTP {r.status_code}: "
                                       f"{r.read()[:400].decode(errors='replace')}"})
                        emit({"done": True})
                        return
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        fingerprint = ev.get("system_fingerprint") or fingerprint
                        u = ev.get("usage") or {}
                        in_tok = u.get("prompt_tokens") or in_tok
                        out_tok = u.get("completion_tokens") or out_tok
                        ch = (ev.get("choices") or [{}])[0]
                        finish = ch.get("finish_reason") or finish
                        d = ch.get("delta", {}) or {}
                        think = next((d[k] for k in REASONING_KEYS if d.get(k)), "")
                        if think:
                            emit({"think": think})
                        if d.get("content"):
                            emit({"text": d["content"]})
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}  "
                           "(a hang with no response usually means your network is not "
                           "allowlisted for the endpoint)"})
        emit({"done": True, "in_tok": in_tok or used, "out_tok": out_tok,
              "fingerprint": fingerprint, "truncated": finish == "length"})


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    if not a.base:
        sys.exit("Set GPTOSS_BASE_URL (or pass --base). Ask the organisers for the URL.")
    BASE = a.base.rstrip("/")
    print(f"upstream: {BASE}\nopen:     http://localhost:{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
