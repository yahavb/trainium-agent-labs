'use strict';
const $ = id => document.getElementById(id);
const ACTIVE = new Set(['queued', 'running', 'cancelling']);
const state = {connected: false, endpoint: '', token: '', runId: '', run: null, workerActive: '', kernel: '', timer: null, generation: 0, busy: false, generating: false, polling: false, mode: 'code'};
const example = `"""Elementwise add: float32 CPU tensors, including tile tails."""
import torch

def reference(a, b):
    return a + b

def cases(seed):
    g = torch.Generator().manual_seed(seed)
    for shape in [(128, 512), (256, 1024), (129, 513)]:
        yield (torch.randn(shape, generator=g),
               torch.randn(shape, generator=g))
    yield (torch.zeros(128, 512), torch.zeros(128, 512))

RTOL = 1e-5
ATOL = 1e-6
`;
const storage = {get(key){try{return sessionStorage.getItem('nkievolve.' + key) || '';}catch{return '';}}, set(key,value){try{sessionStorage.setItem('nkievolve.' + key,value);}catch{}}, clear(){try{['endpoint','token','run'].forEach(k=>sessionStorage.removeItem('nkievolve.'+k));}catch{}}};
function notice(message, error = false) { $('notice').textContent = message; $('notice').className = 'notice' + (error ? ' error' : ''); $('notice').hidden = !message; }
function tab(name, focus = false) {
  document.querySelectorAll('[data-tab]').forEach(button => { const selected = button.dataset.tab === name; button.classList.toggle('active', selected); button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1; if(selected && focus) button.focus(); });
  ['launcher','logs','kernel'].forEach(id => $(id).hidden = id !== name);
}
document.querySelectorAll('[data-tab]').forEach(button => {
  button.addEventListener('click', () => tab(button.dataset.tab));
  button.addEventListener('keydown', event => {const ids=['launcher','logs','kernel'];let i=ids.indexOf(button.dataset.tab);if(event.key==='ArrowRight')i=(i+1)%3;else if(event.key==='ArrowLeft')i=(i+2)%3;else if(event.key==='Home')i=0;else if(event.key==='End')i=2;else return;event.preventDefault();tab(ids[i],true);});
});
function updateControls() {
  const ready = $('verified').checked && $('reference-code').value.trim().length > 0;
  const active = state.run && ACTIVE.has(state.run.status);
  $('configuration').disabled = !ready || state.busy;
  $('launch').disabled = !state.connected || !ready || state.busy || !!state.workerActive;
  $('generate').disabled = !state.connected || state.generating || $('description').value.trim().length < 10;
  $('verified').disabled = !state.connected || !$('reference-code').value.trim() || state.generating || state.busy;
  $('refresh-runs').disabled = !state.connected;
  $('cancel').disabled = !state.connected || !active || state.busy;
  $('copy-kernel').disabled = !state.kernel;
  $('download-kernel').disabled = !state.kernel;
  $('launch-hint').textContent = !state.connected ? 'Connect to your worker to continue.' : !ready ? 'Review and verify your reference to continue.' : state.workerActive ? 'This worker is evolving a kernel.' : 'Ready to evolve. This starts paid model calls.';
}
function invalidateReference() {
  $('verified').checked = false;
  const lines = $('reference-code').value.split('\n').length;
  $('line-count').textContent = `${lines} ${lines === 1 ? 'line' : 'lines'}`;
  updateControls();
}
function inputMode(mode) {
  state.mode = mode;
  ['code','description'].forEach(id=>{$('mode-'+id).classList.toggle('selected',id===mode);$('mode-'+id).setAttribute('aria-pressed',String(id===mode));});
  $('description-pane').hidden = mode !== 'description';
  $('reference-pane').hidden = mode === 'description' && !$('reference-code').value.trim();
  $('verified').checked = false;
  updateControls();
}
$('mode-code').addEventListener('click',()=>inputMode('code'));
$('mode-description').addEventListener('click',()=>inputMode('description'));
$('reference-code').addEventListener('input',()=>{$('reference-origin').textContent='EDITED REFERENCE';invalidateReference();});
$('description').addEventListener('input',()=>{$('verified').checked=false;updateControls();});
$('verified').addEventListener('change',updateControls);
$('load-example').addEventListener('click',()=>{$('reference-code').value=example;$('reference-origin').textContent='TENSOR ADD EXAMPLE';$('module-name').value='tensor-add';invalidateReference();});
for(const id of ['reference-code','initial-code']) $(id).addEventListener('keydown',event=>{if(event.key==='Tab'){event.preventDefault();const editor=event.target;editor.setRangeText('    ',editor.selectionStart,editor.selectionEnd,'end');editor.dispatchEvent(new Event('input'));}});

async function api(path, options = {}, kind = 'json') {
  const headers = {'Authorization':'Bearer '+state.token,'ngrok-skip-browser-warning':'true',...options.headers};
  if(options.body) headers['Content-Type'] = 'application/json';
  const response = await fetch(state.endpoint + path, {...options, headers});
  if(!response.ok) {
    let detail;try{const data=await response.json();detail=Array.isArray(data.detail)?data.detail.map(x=>`${x.loc?.slice(1).join('.') || 'Request'}: ${x.msg}`).join('\n'):data.detail;}catch{detail=await response.text().catch(()=>'');}
    if(response.status===401)throw new Error('The API token is invalid or has changed. Reconnect with the token printed by backend.py.');
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return kind==='text' ? response.text() : response.json();
}
function resetRun() {
  state.runId='';state.run=null;state.kernel='';storage.set('run','');
  renderRun(null);updateControls();
}
function disconnect() {
  state.generation++;clearTimeout(state.timer);state.connected=false;state.workerActive='';state.token='';storage.clear();
  $('api-token').value='';$('endpoint').disabled=false;$('api-token').disabled=false;$('connect').hidden=false;$('disconnect').hidden=true;
  $('connection-status').className='status';$('connection-status').replaceChildren(Object.assign(document.createElement('i'),{}),document.createTextNode('Disconnected'));
  resetRun();notice('Disconnected. Jobs already running on the worker continue.');
}
$('disconnect').addEventListener('click',disconnect);
async function connect(event) {
  event?.preventDefault();
  const endpoint = $('endpoint').value.trim().replace(/\/$/,'');
  let url;try{url=new URL(endpoint);if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw new Error();}catch{notice('Enter a valid HTTP or HTTPS worker URL.',true);return;}
  state.endpoint=endpoint;state.token=$('api-token').value.trim();
  $('connect').disabled=true;$('connect').textContent='Connecting…';notice('');
  const generation = ++state.generation;
  try {
    const health = await api('/health');
    if(generation!==state.generation)return;
    state.connected=true;storage.set('endpoint',endpoint);storage.set('token',state.token);
    $('connection-status').className='status connected';$('connection-status').replaceChildren(document.createElement('i'),document.createTextNode('Worker connected'));
    $('connect').hidden=true;$('disconnect').hidden=false;$('endpoint').disabled=true;$('api-token').disabled=true;
    await refreshRuns();
    const previous=storage.get('run');
    const validPrevious=Array.from($('run-select').options).some(o=>o.value===previous);
    const selected=health.active_job || (validPrevious ? previous : '');
    if(selected)selectRun(selected);else resetRun();
    notice('Connected. Your token is kept in this browser tab’s session only.');
  } catch(error) {state.connected=false;notice('Could not connect: '+error.message,true);}
  finally {$('connect').disabled=false;$('connect').textContent='Connect ↗';updateControls();}
}
$('connection-form').addEventListener('submit',connect);

$('generate').addEventListener('click',async()=>{
  state.generating=true;$('verified').checked=false;$('generate').textContent='Generating reference…';$('description').readOnly=true;updateControls();notice('Generating PyTorch with GPT-6 Luna. This can take a few minutes.');
  const generation=state.generation;
  try {
    const result=await api('/references/generate',{method:'POST',body:JSON.stringify({description:$('description').value.trim()})});
    if(generation!==state.generation)return;
    $('reference-code').value=result.reference_code;$('reference-origin').textContent='GENERATED · REVIEW REQUIRED';$('reference-pane').hidden=false;invalidateReference();
    notice('Reference generated. Review the code and test cases, then verify it before selecting your evolution model.');
    $('reference-code').focus();
  }catch(error){if(generation===state.generation)notice(error.message,true);}
  finally{state.generating=false;$('description').readOnly=false;$('generate').textContent='✧ Generate PyTorch';updateControls();}
});

$('launch').addEventListener('click',async()=>{
  const fields=['module-name','iterations','timeout'];
  if(!fields.every(id=>$(id).reportValidity()))return;
  if(!$('verified').checked||!state.connected)return;
  state.busy=true;$('launch').textContent='Starting evolution…';updateControls();notice('');
  try {
    const body={reference_code:$('reference-code').value,verified:true,module_name:$('module-name').value,model:document.querySelector('[name="model"]:checked').value,iterations:Number($('iterations').value),timeout:Number($('timeout').value),'wandb':$('wandb-enabled').checked};
    if($('initial-code').value.trim())body.initial_kernel_code=$('initial-code').value;
    const run=await api('/runs',{method:'POST',body:JSON.stringify(body)});
    await refreshRuns();selectRun(run.id);tab('logs');
    notice('Evolution started. Preparing and validating the starting kernel can take several minutes.');
  }catch(error){notice(error.message,true);}
  finally{state.busy=false;$('launch').innerHTML='Start evolution <span aria-hidden="true">→</span>';updateControls();}
});
async function refreshRuns() {
  const runs=await api('/runs');
  state.workerActive=runs.find(run=>ACTIVE.has(run.status))?.id || '';
  const select=$('run-select');select.replaceChildren(new Option('Select a run', ''));
  runs.forEach(run=>select.add(new Option(`${run.module_name} · ${run.status} · ${run.id.slice(0,6)}`,run.id)));
  if(state.runId)select.value=state.runId;
  updateControls();
}
$('refresh-runs').addEventListener('click',()=>refreshRuns().catch(error=>notice(error.message,true)));
$('run-select').addEventListener('change',()=>selectRun($('run-select').value));
function selectRun(id) {
  clearTimeout(state.timer);state.generation++;state.polling=false;
  state.runId=id;state.run=null;state.kernel='';$('run-select').value=id;storage.set('run',id);
  renderRun(null);updateControls();if(id)poll();
}
async function poll() {
  const id=state.runId,generation=state.generation;
  if(!state.connected||!id||state.polling)return;
  state.polling=true;
  try {
    const [run,logs]=await Promise.all([api('/runs/'+id),api('/runs/'+id+'/logs',{},'text')]);
    if(generation!==state.generation)return;
    const wasActive=state.run&&ACTIVE.has(state.run.status);
    state.run=run;
    if(ACTIVE.has(run.status))state.workerActive=run.id;
    else if(state.workerActive===run.id)state.workerActive='';
    renderRun(run);$('console').textContent=logs||'Preparing the worker…';
    if($('follow-logs').checked)$('console').scrollTop=$('console').scrollHeight;
    $('last-update').textContent='Updated '+new Date().toLocaleTimeString();
    if(run.kernel_available){const code=await api('/runs/'+id+'/kernel',{},'text');if(generation!==state.generation)return;state.kernel=code;renderKernel();}
    if(wasActive&&!ACTIVE.has(run.status)){await refreshRuns();notice(run.status==='completed'?'Evolution complete. Open Kernel Code to copy or download the best kernel.':`Run ${run.status}. Check the worker output for details.`,run.status==='failed'||run.status==='interrupted');}
    updateControls();
  }catch(error){if(generation===state.generation)notice('Live updates interrupted: '+error.message,true);}
  finally{if(generation===state.generation){state.polling=false;if(state.connected&&state.runId)state.timer=setTimeout(poll,state.run&&!ACTIVE.has(state.run.status)?10000:2500);}}
}
$('cancel').addEventListener('click',async()=>{
  if(!state.runId)return;state.busy=true;updateControls();
  try{await api('/runs/'+state.runId+'/cancel',{method:'POST'});await refreshRuns();clearTimeout(state.timer);if(!state.polling)await poll();notice('Run stopped. Any best kernel already measured remains available.');}
  catch(error){notice(error.message,true);}
  finally{state.busy=false;updateControls();}
});
const number = value => Number.isFinite(value) ? value.toFixed(2) : '—';
const measured = row => row['kernel/correctness']===1 && row['kernel/hardware_measured']===1;
function renderRun(run) {
  const rows=run?.metrics || [];
  const good=rows.filter(measured);
  const last=rows.at(-1);
  $('run-title').textContent=run?.module_name || 'Waiting for a run';
  $('run-description').textContent=run ? `${run.model} · ${run.id.slice(0,8)} · hardware evaluation` : 'Launch an operation to follow its progress here.';
  $('run-status').textContent=run?.status || 'Idle';$('run-status').className='status '+(run?.status || '');
  $('live-dot').hidden=!run||!ACTIVE.has(run.status);
  $('best-speedup').textContent=good.length ? number(last?.['kernel/best_speedup'])+'×' : '—';
  $('best-latency').textContent=good.length ? number(last?.['kernel/best_latency_us']) : '—';
  $('current-iteration').textContent=last ? `${last.iteration} / ${run.iterations}` : '—';
  $('iteration-help').textContent=run ? 'Baseline is iteration 0' : '0 is the baseline';
  const candidates=rows.filter(row=>row.iteration>0),passed=candidates.filter(row=>row['kernel/correctness']===1);
  $('correct-candidates').textContent=candidates.length ? `${passed.length} / ${candidates.length}` : '—';
  $('correct-help').textContent=candidates.length ? 'passed / evaluated · excludes baseline' : 'Waiting for evaluations';
  $('job-error').hidden=!run?.error;$('job-error').textContent=run?.error || '';
  const link=run?.wandb?.url;
  let safe=false;try{const url=new URL(link);safe=url.protocol==='https:'&&(url.hostname==='wandb.ai'||url.hostname.endsWith('.wandb.ai'));}catch{}
  $('wandb-link').hidden=!safe;if(safe)$('wandb-link').href=link;
  drawChart('speedup-chart',good,'kernel/speedup','kernel/best_speedup');
  drawChart('latency-chart',good,'kernel/latency_us','kernel/best_latency_us');
  const tbody=$('evaluations');tbody.replaceChildren();
  if(!rows.length){const cell=document.createElement('td');cell.colSpan=5;cell.className='empty-row';cell.textContent='Evaluated candidates will appear here.';const tr=document.createElement('tr');tr.append(cell);tbody.append(tr);}
  rows.slice().reverse().forEach(row=>{const tr=document.createElement('tr');const passed=row['kernel/correctness']===1;const values=[row.iteration===0?'0 · baseline':row.iteration,passed?'Passed':'Failed',measured(row)?number(row['kernel/latency_us']):'—',measured(row)?number(row['kernel/speedup'])+'×':'—',number(row['kernel/best_speedup'])+'×'];values.forEach((value,i)=>{const cell=document.createElement('td');cell.textContent=value;if(i===1)cell.className=passed?'pass':'fail';tr.append(cell);});tbody.append(tr);});
  if(!run){$('console').textContent='Select or launch a run to see live output.';$('last-update').textContent='No data yet';}
  renderKernel();
}
function drawChart(id,rows,key,bestKey) {
  const container=$(id),points=rows.filter(row=>Number.isFinite(row[key]));
  container.replaceChildren();
  if(!points.length){const p=document.createElement('p');p.className='empty-chart';p.textContent=id==='speedup-chart'?'Speedup appears after hardware evaluation.':'Latency appears after hardware evaluation.';container.append(p);return;}
  const ns='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 520 230');
  const make=(tag,attrs,text)=>{const element=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>element.setAttribute(k,String(v)));if(text!==undefined)element.textContent=text;svg.append(element);return element;};
  const left=48,top=10,width=456,height=186;
  const values=points.flatMap(p=>[p[key],p[bestKey]]).filter(Number.isFinite);
  const low=Math.min(...values),high=Math.max(...values),pad=Math.max((high-low)*.15,Math.abs(high)*.1,.1);
  const min=Math.max(0,low-pad),max=high+pad,xMax=Math.max(1,...points.map(p=>p.iteration));
  const x=i=>left+(i/xMax)*width,y=v=>top+height-(v-min)/(max-min)*height;
  for(let i=0;i<=4;i++){const value=min+(max-min)*i/4,at=y(value);make('line',{x1:left,y1:at,x2:left+width,y2:at,stroke:'#e8ece2','stroke-dasharray':'3 4'});make('text',{x:left-9,y:at+3,'text-anchor':'end',fill:'#8c9980','font-size':9,'font-family':'monospace'},value.toFixed(value<10?2:1));}
  const ticks=new Set([0,Math.round(xMax/2),xMax]);ticks.forEach(i=>make('text',{x:x(i),y:219,'text-anchor':'middle',fill:'#8c9980','font-size':9,'font-family':'monospace'},i));
  for(const [field,color] of [[key,'#a1b28f'],[bestKey,'#547b3c']]){
    const series=points.filter(p=>Number.isFinite(p[field]));
    make('polyline',{points:series.map(p=>`${x(p.iteration)},${y(p[field])}`).join(' '),fill:'none',stroke:color,'stroke-width':field===bestKey?2.3:1.5,'stroke-linejoin':'round'});
    series.forEach(p=>{const dot=make('circle',{cx:x(p.iteration),cy:y(p[field]),r:field===bestKey?3:2.5,fill:color});const title=document.createElementNS(ns,'title');title.textContent=`Iteration ${p.iteration}: ${p[field].toFixed(2)} ${id==='speedup-chart'?'×':'µs'}`;dot.append(title);});
  }
  container.append(svg);container.setAttribute('aria-label',`${id==='speedup-chart'?'Speedup':'Latency'} from ${points.length} hardware evaluations. Latest candidate ${number(points.at(-1)[key])}.`);
}
function renderKernel() {
  const available=!!state.kernel;
  $('kernel-empty').hidden=available;$('kernel-code').hidden=!available;$('kernel-summary').hidden=!available;
  $('kernel-code').querySelector('code').textContent=state.kernel;
  $('kernel-state').textContent=available ? (state.run&&ACTIVE.has(state.run.status)?'BEST SO FAR · EVOLVING':'BEST MEASURED KERNEL') : 'AWAITING HARDWARE EVALUATION';
  const rows=state.run?.metrics || [],last=rows.at(-1);
  // The minimum latency can belong to a different winner; show only the selected speedup.
  $('kernel-performance').textContent=`${number(last?.['kernel/best_speedup'])}× geometric mean speedup`;
}
$('back-launcher').addEventListener('click',()=>tab('launcher'));
$('copy-kernel').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(state.kernel);notice('Kernel code copied.');}catch{notice('Clipboard access was blocked. Use Download .py or select the code to copy it.',true);}});
$('download-kernel').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([state.kernel],{type:'text/x-python'}));const a=document.createElement('a');a.href=url;a.download='best_kernel.py';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.addEventListener('visibilitychange',()=>{if(!document.hidden&&state.connected&&state.runId&&!state.polling){clearTimeout(state.timer);poll();}});
$('endpoint').value=storage.get('endpoint') || (location.protocol.startsWith('http')?location.origin:'');
$('api-token').value=storage.get('token');
updateControls();
if($('api-token').value)connect();
