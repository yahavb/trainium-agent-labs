"""Separate localhost CPU adapter endpoint. Does not modify Neuron/vLLM serving."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer


def completion_ids(ids,eos):
    """Count actual generated tokens through the first EOS, including EOS."""
    return ids[:ids.index(eos)+1] if eos in ids else ids


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--port',type=int,default=8001);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();os.nice(15);sys.path.insert(0,str(Path(__file__).parent/'vendor'))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(2026)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
    model=PeftModel.from_pretrained(base,a.adapter,is_trainable=False).eval();model.config.use_cache=True
    jobs=queue.Queue()
    def worker():
        while True:
            batch=[jobs.get()];deadline=time.monotonic()+1
            while len(batch)<4 and time.monotonic()<deadline:
                try:batch.append(jobs.get(timeout=max(.001,deadline-time.monotonic())))
                except queue.Empty:break
            try:
                limits={j['body'].get('max_tokens',2500) for j in batch}
                if len(limits)!=1:raise ValueError('Batch token budgets must match')
                prompts=[tokenizer.apply_chat_template(j['body']['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for j in batch]
                inputs=tokenizer(prompts,padding=True,return_tensors='pt',add_special_tokens=False)
                max_tokens=int(next(iter(limits)))
                if inputs.input_ids.shape[1]+max_tokens>8192:raise ValueError('Context exceeds 8192; refusing truncation')
                modes={j['body'].get('adapter_mode','lora') for j in batch}
                if len(modes)!=1 or not modes.issubset({'base','lora'}):raise ValueError('Adapter mode must be uniform and base or lora')
                adapter_mode=next(iter(modes))
                first=batch[0]['body'];temperature=float(first.get('temperature',.6));top_p=float(first.get('top_p',.95))
                if any(float(j['body'].get('temperature',.6))!=temperature or float(j['body'].get('top_p',.95))!=top_p for j in batch):raise ValueError('Batch sampling settings must match')
                adapter_context=model.disable_adapter() if adapter_mode=='base' else contextlib.nullcontext()
                with adapter_context, torch.inference_mode():
                    generated=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=temperature>0,temperature=temperature if temperature>0 else 1.,top_p=top_p,pad_token_id=tokenizer.pad_token_id)
                for i,j in enumerate(batch):
                    ids=completion_ids(generated[i,inputs.input_ids.shape[1]:].tolist(),tokenizer.eos_token_id)
                    prompt_tokens=int(inputs.attention_mask[i].sum());count=len(ids)
                    j['result']={'id':'cpu-'+str(time.time_ns()),'object':'chat.completion','model':'Qwen/Qwen3-8B','adapter_mode':adapter_mode,'choices':[{'index':0,'message':{'role':'assistant','content':tokenizer.decode(ids,skip_special_tokens=True)},'finish_reason':'stop' if ids and ids[-1]==tokenizer.eos_token_id else 'length'}],'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':count,'total_tokens':prompt_tokens+count}}
            except Exception as e:
                for j in batch:j['error']=str(e)
            finally:
                for j in batch:j['event'].set()
    threading.Thread(target=worker,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body):
            data=json.dumps(body).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/v1/models':self.send(200,{'data':[{'id':'Qwen/Qwen3-8B','adapter':a.adapter,'device':'cpu'}]})
            else:self.send(404,{'error':'Unknown path'})
        def do_POST(self):
            if self.path!='/v1/chat/completions':return self.send(404,{'error':'Unknown path'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1000000:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length))
                if body.get('n',1)!=1:raise ValueError('Use independent candidate requests')
                j={'body':body,'event':threading.Event()};jobs.put(j);j['event'].wait()
                self.send(500,{'error':j['error']}) if 'error' in j else self.send(200,j['result'])
            except Exception as e:self.send(400,{'error':str(e)})
    print(f'CPU LoRA endpoint ready: http://127.0.0.1:{a.port}/v1',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
