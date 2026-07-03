const FIELDS=['library_path','workers','trust_db','no_cache',
  'label_path','label_panns','label_clap','gpu_python','redo','limit',
  'analyze_seconds','panns_min_duration','proj_method','proj_n_neighbors','proj_min_dist'];
const BOOLS=new Set(['trust_db','no_cache','label_path','label_panns','label_clap']);

function getForm(){
  const d={};
  for(const k of FIELDS){
    const el=document.getElementById(k);
    if(!el) continue;
    if(BOOLS.has(k)) d[k]=el.checked;
    else if(el.tagName==='SELECT') d[k]=el.value;
    else if(el.type==='number') d[k]=el.value===''?null:Number(el.value);
    else d[k]=el.value.trim();
  }
  return d;
}
function setForm(cfg){
  for(const k of FIELDS){
    const el=document.getElementById(k);
    if(!el) continue;
    if(BOOLS.has(k)) el.checked=!!cfg[k];
    else if(el.tagName==='SELECT') el.value=cfg[k]||'auto';
    else el.value=cfg[k]!=null?cfg[k]:'';
  }
}
function setMsg(txt,isErr){
  const el=document.getElementById('msg');
  el.textContent=txt; el.className='msg '+(isErr?'err':'ok');
}

async function loadConfig(){
  const r=await fetch('/api/config');
  setForm(await r.json());
}

document.getElementById('btnSave').onclick=async()=>{
  try{
    const r=await fetch('/api/config',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(getForm())});
    if(r.ok) setMsg('Config saved.');
    else setMsg('Save failed: '+(await r.text()),true);
  }catch(e){setMsg('Save error: '+e,true);}
};

async function runStage(stage){
  try{
    const r=await fetch('/api/run/'+stage,{method:'POST'});
    const d=await r.json();
    if(d.ok) setMsg(stage.charAt(0).toUpperCase()+stage.slice(1)+' started (PID '+d.pid+').');
    else setMsg('Failed: '+d.msg,true);
  }catch(e){setMsg('Error: '+e,true);}
  updateStatus();
}
document.getElementById('btnDiscover').onclick=async()=>runStage('discover');
document.getElementById('btnLabel').onclick=async()=>runStage('label');

document.getElementById('btnStop').onclick=async()=>{
  if(!confirm('Send SIGTERM to the running process?')) return;
  try{
    const r=await fetch('/api/run/stop',{method:'POST'});
    const d=await r.json();
    setMsg(d.pid?'Stop signal sent to PID '+d.pid+'.':'No running process found.');
  }catch(e){setMsg('Stop error: '+e,true);}
  updateStatus();
};

async function updateStatus(){
  try{
    const s=await fetch('/api/run/status').then(r=>r.json());
    const badge=document.getElementById('runBadge');
    const info=document.getElementById('runInfo');
    const running=s.running;
    if(running){
      badge.className='badge run'; badge.textContent='RUNNING';
      let txt='PID '+s.pid;
      if(s.progress&&s.progress.total){
        const done=s.progress.done||0;
        const rem=s.progress.total-done;
        txt+=' · '+done.toLocaleString()+' / '+s.progress.total.toLocaleString();
        if(s.progress.eta_min!=null) txt+=' · eta '+(s.progress.eta_min>60?(s.progress.eta_min/60).toFixed(1)+'h':s.progress.eta_min+'m');
      }
      info.textContent=txt;
    } else {
      badge.className='badge idle'; badge.textContent='IDLE';
      info.textContent='';
    }
    document.getElementById('btnDiscover').disabled=running;
    document.getElementById('btnLabel').disabled=running;
    document.getElementById('btnStop').disabled=!running;
  }catch(e){}
  setTimeout(updateStatus,3000);
}

document.getElementById('btnML').onclick=async()=>{
  try{
    const r=await fetch('/api/run/ml',{method:'POST'});
    const d=await r.json();
    if(!d.ok) document.getElementById('mlMsg').textContent='Failed: '+d.msg;
    else document.getElementById('mlMsg').textContent='';
  }catch(e){document.getElementById('mlMsg').textContent='Error: '+e;}
  updateMLStatus();
};

document.getElementById('btnMLStop').onclick=async()=>{
  if(!confirm('Stop the ML pipeline?')) return;
  await fetch('/api/run/ml/stop',{method:'POST'});
  updateMLStatus();
};

async function updateMLStatus(){
  try{
    const s=await fetch('/api/run/ml/status').then(r=>r.json());
    const badge=document.getElementById('mlBadge');
    const info=document.getElementById('mlInfo');
    const box=document.getElementById('mlLogBox');
    const lastEl=document.getElementById('mlLastTrained');
    if(s.last_trained){
      lastEl.textContent='Last trained: '+new Date(s.last_trained*1000).toLocaleString();
    } else {
      lastEl.textContent='Not yet trained.';
    }
    if(s.running){
      badge.className='badge run'; badge.textContent='RUNNING';
      info.textContent=s.pid?'PID '+s.pid:'';
    } else {
      const cls={idle:'idle',done:'done',error:'run'}[s.state]||'idle';
      badge.className='badge '+cls; badge.textContent=s.state.toUpperCase();
      info.textContent='';
    }
    if(s.log_tail&&s.log_tail.length){
      box.style.display='block';
      box.textContent=s.log_tail.join('\n');
      box.scrollTop=box.scrollHeight;
    }
    document.getElementById('btnML').disabled=s.running;
    document.getElementById('btnMLStop').disabled=!s.running;
  }catch(e){}
  setTimeout(updateMLStatus,3000);
}

loadConfig();
updateStatus();
updateMLStatus();