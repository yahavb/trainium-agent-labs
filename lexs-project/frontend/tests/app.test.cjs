const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '..');
const reference = 'import torch\ndef reference(a): return a\ndef cases(seed): yield (torch.zeros(128,512),)\n';
const metrics = [
  {iteration:0,'kernel/correctness':1,'kernel/hardware_measured':1,'kernel/speedup':1,'kernel/best_speedup':1,'kernel/latency_us':25,'kernel/best_latency_us':25},
  {iteration:1,'kernel/correctness':0,'kernel/hardware_measured':0,'kernel/speedup':0,'kernel/best_speedup':1,'kernel/best_latency_us':25},
  {iteration:2,'kernel/correctness':1,'kernel/hardware_measured':1,'kernel/speedup':2,'kernel/best_speedup':2,'kernel/latency_us':12.5,'kernel/best_latency_us':12.5}
];
function fixture() {
  const dom = new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'), {url:'http://localhost:8000',runScripts:'outside-only'});
  const w=dom.window, requests=[], jobs=[];
  w.fetch=async (url,options={})=>{
    const route=new URL(url).pathname;
    requests.push({route,options});
    let data;
    if(route==='/health')data={status:'ok',active_job:null};
    else if(route==='/references/generate')data={reference_code:reference,requires_verification:true};
    else if(route==='/runs'&&options.method==='POST'){
      const payload=JSON.parse(options.body);
      const run={id:'run-123',status:'completed',module_name:payload.module_name,model:payload.model,iterations:payload.iterations,metrics,kernel_available:true,wandb:{url:'https://wandb.ai/entity/project/runs/123'}};
      jobs.push(run);data=run;
    }else if(route==='/runs')data=jobs;
    else if(route==='/runs/run-123')data=jobs[0];
    else if(route.endsWith('/logs'))data='Compiler finished\nEvaluation passed';
    else if(route.endsWith('/kernel'))data='# BEST KERNEL\ndef kernel(a): return a\n';
    else throw new Error('Unexpected route '+route);
    return {ok:true,json:async()=>JSON.parse(JSON.stringify(data)),text:async()=>String(data)};
  };
  w.eval(fs.readFileSync(path.join(root,'app.js'),'utf8'));
  const $=id=>w.document.getElementById(id);
  const settle=()=>new Promise(resolve=>setTimeout(resolve,15));
  const event=(id,type)=>$(id).dispatchEvent(new w.Event(type,{bubbles:true,cancelable:true}));
  const connect=async()=>{$('api-token').value='test-only-access-token';event('connection-form','submit');await settle();};
  return {dom,w,$,requests,jobs,settle,event,connect};
}

test('description requires review, model selection is sent, and results render across three tabs',async()=>{
  const f=fixture();try{
    const {$,event,settle,connect,requests}=f;
    assert.equal($('launch').disabled,true);
    await connect();
    assert.match($('connection-status').textContent,/connected/);
    event('mode-description','click');
    $('description').value='Add one to a float32 tensor with shape 128 by 512';event('description','input');
    assert.equal($('generate').disabled,false);
    event('generate','click');await settle();
    assert.equal($('reference-code').value,reference);
    assert.equal($('reference-pane').hidden,false);
    assert.equal($('verified').checked,false);
    assert.equal($('configuration').disabled,true);
    assert.equal(requests.filter(r=>r.route==='/runs'&&r.options.method==='POST').length,0);
    $('verified').checked=true;event('verified','change');
    assert.equal($('configuration').disabled,false);
    f.w.document.querySelector('[value="gpt-6-astra"]').checked=true;
    assert.equal($('launch').disabled,false);
    event('launch','click');await settle();
    const launch=requests.find(r=>r.route==='/runs'&&r.options.method==='POST');
    const body=JSON.parse(launch.options.body);
    assert.equal(body.model,'gpt-6-astra');assert.equal(body.verified,true);assert.equal(body.reference_code,reference);
    assert.equal($('logs').hidden,false);
    assert.equal($('best-speedup').textContent,'2.00×');
    assert.equal($('best-latency').textContent,'12.50');
    assert.equal($('correct-candidates').textContent,'1 / 2');
    assert.equal($('evaluations').children.length,3);
    assert.equal($('speedup-chart').querySelectorAll('circle').length,4); // failed evaluation has no chart point
    assert.equal($('wandb-link').hidden,false);
    event('tab-kernel','click');
    assert.equal($('kernel').hidden,false);
    assert.match($('kernel-code').textContent,/# BEST KERNEL/);
    assert.equal($('download-kernel').disabled,false);
    event('tab-launcher','click');
    $('reference-code').value+='\n# change';event('reference-code','input');
    assert.equal($('verified').checked,false);assert.equal($('launch').disabled,true);
  }finally{f.dom.window.close();}
});

test('disconnect clears session credentials and tabs support keyboard navigation',async()=>{
  const f=fixture();try{
    await f.connect();
    assert.ok(f.w.sessionStorage.getItem('nkievolve.token'));
    f.$('tab-launcher').dispatchEvent(new f.w.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
    assert.equal(f.$('logs').hidden,false);
    assert.equal(f.$('tab-logs').getAttribute('aria-selected'),'true');
    f.event('disconnect','click');
    assert.equal(f.w.sessionStorage.getItem('nkievolve.token'),null);
    assert.equal(f.$('api-token').value,'');
    assert.equal(f.$('launch').disabled,true);
  }finally{f.dom.window.close();}
});
