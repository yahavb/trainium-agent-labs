"""CPU-isolated Qwen LoRA SFT; never touches the serving endpoint or Neuron SDK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='Qwen/Qwen3-8B');p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--rank',type=int,default=8)
    p.add_argument('--max-length',type=int,default=2048);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--tiny-test',action='store_true')
    a=p.parse_args()
    os.nice(15)
    sys.path.insert(0,str(Path(__file__).parent/'vendor'))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,Qwen3Config,Qwen3ForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(2026)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenized={}
    for split in ['train','heldout']:
        records=[json.loads(x) for x in (Path(a.data)/(split+'.jsonl')).read_text().splitlines()]
        samples=[]
        for r in records:
            prefix=tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if hasattr(prefix,'keys'):prefix=prefix['input_ids']
            target=tokenizer.encode(r['completion']+tokenizer.eos_token,add_special_tokens=False)
            ids=prefix+target
            if len(ids)>a.max_length:raise ValueError(f"{r['id']} exceeds token budget; refusing silent truncation")
            samples.append({'id':r['id'],'ids':ids,'labels':[-100]*len(prefix)+target})
        tokenized[split]=samples
    if not tokenized['train'] or not tokenized['heldout']:raise ValueError('Empty split')
    metadata={'base_model':a.model,'hardware':'CPU','threads':a.threads,'rank':a.rank,'epochs':a.epochs,'seed':2026,'learning_rate':a.learning_rate,'completion_only_loss':True,'thinking':False,'tiny_test':a.tiny_test,'train_count':len(tokenized['train']),'heldout_count':len(tokenized['heldout']),'max_actual_length':max(len(s['ids']) for rows in tokenized.values() for s in rows),'dataset_sha256':hashlib.sha256((Path(a.data)/'train.jsonl').read_bytes()).hexdigest(),'serving_model_modified':False}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print('Loading model:',metadata,flush=True)
    if a.tiny_test:
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8))
    else:
        model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[v for v in model.parameters() if v.requires_grad]
    print('Trainable parameters:',sum(v.numel() for v in params),flush=True)
    optimizer=torch.optim.AdamW(params,lr=a.learning_rate)
    def loss(sample):
        ids=torch.tensor([sample['ids']],dtype=torch.long)
        labels=torch.tensor([sample['labels']],dtype=torch.long)
        return model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss
    def evaluate():
        model.eval()
        with torch.no_grad():values=[float(loss(s)) for s in tokenized['heldout']]
        return sum(values)/len(values)
    before=evaluate();print('Held-out loss before:',before,flush=True)
    started=time.perf_counter();step=0
    with (out/'metrics.jsonl').open('x',buffering=1) as log:
        for epoch in range(a.epochs):
            order=torch.randperm(len(tokenized['train'])).tolist()
            if a.tiny_test:order=order[:2]
            for index in order:
                model.train();optimizer.zero_grad(set_to_none=True)
                t=time.perf_counter();value=loss(tokenized['train'][index])
                if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
                value.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();step+=1
                row={'step':step,'epoch':epoch,'example_id':tokenized['train'][index]['id'],'loss':float(value.detach()),'step_seconds':time.perf_counter()-t}
                log.write(json.dumps(row)+'\n');print(row,flush=True)
                if step%10==0:model.save_pretrained(out/f'checkpoint-{step}',safe_serialization=True)
    after=evaluate();model.save_pretrained(out/'adapter',safe_serialization=True)
    result={'steps':step,'heldout_loss_before':before,'heldout_loss_after':after,'elapsed_seconds':time.perf_counter()-started,'adapter_saved':True,'benchmark_correctness_evaluated':False,'serving_model_modified':False,'tiny_test':a.tiny_test}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':main()
