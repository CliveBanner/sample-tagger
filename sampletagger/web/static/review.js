let COLORS={};

let INSTRUMENTS=[], queue=[], cur=-1, _toastTimer=null;
let lblBuf = '';
let step = 0;                 // 0 rate · 1 type · 2 label · 3 propagate+submit
let sessionCount = 0;
let _cAudio = null;           // dedicated player for auditioning candidates
let _cPlaying = -1;           // index of candidate currently playing (-1 = none)
const STEPS = ['Rate','Type','Label','Propagate'];

async function fetchLabels(){
  INSTRUMENTS=await fetch('/api/labels').then(r=>r.json());
  COLORS=await fetch('/api/colors').then(r=>r.json());
  renderLModal();
  const sel=document.getElementById('modesel');
  if(sel){
    Array.from(sel.options).forEach(opt => { if(opt.value.startsWith('class_')) opt.remove(); });
    INSTRUMENTS.forEach(inst => {
      const opt = document.createElement('option');
      opt.value = 'class_' + inst;
      opt.textContent = 'Target: ' + inst;
      sel.appendChild(opt);
    });
  }
  if(cur>=0)renderDetail(queue[cur],cur);
}

function renderLModal(){
  document.getElementById('llist').innerHTML=INSTRUMENTS.map(name=>`
    <div class=ltag><span>${name}</span>
    <button onclick="delLabelUI('${name}')" title="remove">✕</button></div>`).join('');
}

async function addLabelUI(){
  const inp=document.getElementById('linput');
  const name=inp.value.trim().toLowerCase();
  if(!name)return;
  const r=await fetch('/api/labels/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})}).then(r=>r.json());
  if(r.ok){inp.value='';await fetchLabels();}
  else showToast(r.msg, '#f92672');
}

async function delLabelUI(name){
  await fetch('/api/labels/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})});
  await fetchLabels();
}

function openLModal(){document.getElementById('lmodal').classList.add('open');}
function closeLModal(){document.getElementById('lmodal').classList.remove('open');}

function showToast(msg,color){
  const t=document.getElementById('toast');
  t.textContent=msg; t.style.color=color; t.style.borderColor=color;
  t.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer=setTimeout(()=>t.classList.remove('show'),1400);
}

function basename(p){return p.split('/').pop()}
function col(inst){return inst?(COLORS[inst]||'#ae81ff'):'#555'}

function pill(inst){
  if(!inst)return '';
  return `<span class=pill style="background:${col(inst)}">${inst}</span>`;
}

function disagreeing(it){
  const vals=[it.path_instrument,it.panns_instrument,it.audio_instrument,it.model_instrument].filter(Boolean);
  return new Set(vals).size > 1;
}

function suggestedFor(it){
  let s = it.model_instrument || it.panns_instrument || it.audio_instrument || it.path_instrument;
  return INSTRUMENTS.includes(s) ? s : null;
}

function replay(){
  const p=document.getElementById('player');
  if(p){p.currentTime=0;p.play().catch(()=>{});}
}

/* ---------- queue list ---------- */

function listHTML(){
  if(!queue.length)return '<div class=empty>Queue empty.</div>';
  return queue.map((it,i)=>`
    <div class="item${it._done?' done':''}${it._seen?' seen':''}${i===cur?' sel':''}" onclick="select(${i})">
      <div class=iname>${basename(it.path)}</div>
      <div class=pills>
        ${pill(it.path_instrument)}${pill(it.panns_instrument)}${pill(it.audio_instrument)}${pill(it.model_instrument)}
        ${it.human_instrument?`<span class=pill style="background:#a6e22e">✓ ${it.human_instrument}</span>`:''}
        ${it.rating?`<span class=pill style="background:#e6db74;color:#000">★${it.rating}</span>`:''}
      </div>
    </div>`).join('');
}

function renderList(){
  const h=listHTML();
  document.getElementById('list').innerHTML=h;
  document.getElementById('overlay-list').innerHTML=h;
}

/* ---------- stars ---------- */

function starInner(rating){
  let h='';
  for(let n=1;n<=5;n++){const on=n<=rating;
    h+=`<span class="star${on?' on':''}" onclick="rate(${n})">${on?'★':'☆'}</span>`;}
  return h;
}

/* ---------- detail: sample card + stepper + active step ---------- */

function renderDetail(it,i){
  if(!it){document.getElementById('detail').innerHTML='<div class=empty>Select a sample.</div>';return;}
  const rows=[
    it.model_instrument?`<tr><td>model</td><td style="color:${col(it.model_instrument)}">${it.model_instrument}${it.model_conf?`<span class=conf>${(it.model_conf*100).toFixed(0)}%</span>`:''}</td><td></td></tr>`:'',
    it.path_instrument?`<tr><td>path</td><td style="color:${col(it.path_instrument)}">${it.path_instrument}</td><td></td></tr>`:'',
    it.panns_instrument?`<tr><td>PANNs</td><td style="color:${col(it.panns_instrument)}">${it.panns_instrument}${it.panns_conf?`<span class=conf>${(it.panns_conf*100).toFixed(0)}%</span>`:''}</td><td>${disagreeing(it)?'<span class=disagree>⚡ disagrees</span>':''}</td></tr>`:'',
    it.audio_instrument?`<tr><td>audio</td><td style="color:${col(it.audio_instrument)}">${it.audio_instrument}</td><td></td></tr>`:'',
  ].join('');

  const effType=it.human_sample_type||it.sample_type;
  const dur=it.duration_s?it.duration_s.toFixed(2)+'s':'';

  document.getElementById('detail').innerHTML=`
    <div class=sample-card>
      <div class=fname>${basename(it.path)}</div>
      <div class=path>${it.path}</div>
      <audio id=player controls preload=auto></audio>
      <div id=audio-hint>⏳ loading audio…</div>
      <div class=meta-row>
        ${effType?`<span class=pill style="background:${typeColor(effType)}">${effType}</span>`:'<span class=dim>type unknown</span>'}
        ${dur?`<span class=dim>${dur}</span>`:''}
        ${it.sonic&&(it.sonic.grain||it.sonic.family)?`<span class=pill style="background:#1e1f1c;border:1px solid #a6e22e;color:#a6e22e" title="audio-only sonic descriptor${it.sonic.family?' · '+it.sonic.family+' family':''}">🎧 ${it.sonic.grain||it.sonic.family}</span>`:''}
      </div>
      <table>${rows||'<tr><td colspan=3 class=dim>No classifier results yet.</td></tr>'}</table>
    </div>
    <div id=stepper class=stepper></div>
    <div id=step-panel class=step-panel></div>
    <div class=kbhint>1-5 rate · type to label · enter next/submit · ←→ step · space replay · z back · s skip</div>`;

  const player=document.getElementById('player');
  player.classList.add('loading');
  player.addEventListener('canplay',()=>{
    player.classList.remove('loading');
    const h=document.getElementById('audio-hint');
    if(h)h.textContent='';
  },{once:true});
  player.src='/api/audio?norm=1&path='+encodeURIComponent(it.path);
  player.play().catch(()=>{});

  renderStepper();
  renderStepPanel();
}

function typeColor(t){return t==='loop'?'#66d9ef':t==='oneshot'?'#ae81ff':'#75715e';}

function stepDone(it,s){
  if(s===0) return (it.rating||0)>0;
  if(s===1) return !!it.human_sample_type;
  if(s===2) return !!it.human_instrument;
  return false;
}

function renderStepper(){
  const it=queue[cur]; if(!it)return;
  document.getElementById('stepper').innerHTML=STEPS.map((name,s)=>{
    const done=stepDone(it,s);
    const cls='step'+(s===step?' active':'')+(done?' done':'');
    return `<div class="${cls}" onclick="goStep(${s})">
      <span class=step-num>${done?'✓':(s+1)}</span><span class=step-name>${name}</span></div>`;
  }).join('<span class=step-sep>›</span>');
}

function stepActions(){
  const back=`<button class=step-back ${step===0?'disabled':''} onclick="goStep(${step-1})">← Back</button>`;
  const fwd= step<3
    ? `<button class=step-next onclick="goStep(${step+1})">Next →</button>`
    : `<button class=step-submit onclick="submit()">Submit ✓</button>`;
  return `<div class=step-actions>${back}${fwd}</div>`;
}

function renderStepPanel(){
  const it=queue[cur]; if(!it)return;
  const el=document.getElementById('step-panel');
  let body='';
  if(step===0){
    body=`<div class=step-title>1 · Rate the sample</div>
      <div class="stars stars-lg" id=stars>${starInner(it.rating||0)}</div>
      <div class=step-hint>Press 1-5, or click. Optional — Next to skip.</div>`;
  } else if(step===1){
    const eff=it.human_sample_type;
    body=`<div class=step-title>2 · Sample type</div>
      <div class=ibtns>
        ${['oneshot','loop'].map((t,k)=>`<button class="ibtn${eff===t?' active':''}"
          style="border-color:${typeColor(t)};color:${typeColor(t)}"
          onclick="saveType('${t}')">${k+1} · ${t}${eff===t?' ✓':''}</button>`).join('')}
      </div>
      <div class=step-hint>Press 1 oneshot · 2 loop, or click.</div>`;
  } else if(step===2){
    const suggested=suggestedFor(it);
    const match = lblBuf ? INSTRUMENTS.find(x => x.startsWith(lblBuf)) : null;
    const btns=INSTRUMENTS.map((inst)=>{
      let isAct = it.human_instrument===inst;
      let isSug = !isAct && suggested === inst;
      let isMatch = lblBuf && match === inst;
      let cls = 'ibtn' + (isAct?' active':'') + (isSug?' suggested':'') + (isMatch?' match':'');
      return `<button class="${cls}" style="border-color:${col(inst)};color:${col(inst)}"
        onclick="save('${inst}')">${inst}${isAct?' ✓':''}</button>`;
    }).join('');
    body=`<div class=step-title>3 · Instrument label</div>
      <div class="lblbuf-display">${lblBuf ? `Type to label: <strong>${lblBuf}</strong>` : (suggested?`Suggested: <strong style="color:${col(suggested)}">${suggested}</strong> (enter)`:'&nbsp;')}</div>
      <div class=ibtns>${btns}</div>`;
  } else {
    const r=it.rating||0;
    const eff=it.human_sample_type||it.sample_type;
    const summary=`<div class=summary>
        <div><span class=k>Rating</span> ${r?`<span style="color:var(--yellow)">${'★'.repeat(r)}</span>`:'<span class=dim>—</span>'}</div>
        <div><span class=k>Type</span> ${eff?`<span style="color:${typeColor(eff)}">${eff}</span>${it.human_sample_type?'':' <span class=dim>(auto)</span>'}`:'<span class=dim>—</span>'}</div>
        <div><span class=k>Label</span> ${it.human_instrument?`<span style="color:${col(it.human_instrument)};font-weight:bold">${it.human_instrument}</span>`:'<span class=dim>—</span>'}</div>
      </div>`;
    const prop = it.human_instrument ? renderPropagate(it)
      : `<div class=step-hint>No label assigned — nothing to propagate. Press enter to submit.</div>`;
    body=`<div class=step-title>4 · Propagate &amp; submit</div>${summary}${prop}`;
  }
  el.innerHTML=body+stepActions();
}

function goStep(n){
  if(n<0||n>3)return;
  step=n;
  renderStepper();
  renderStepPanel();
}

/* ---------- propagate candidates ---------- */

function renderPropagate(it){
  const head=`Propagate <strong style="color:${col(it.human_instrument)}">${it.human_instrument}</strong> to similar samples`;
  if(it._cands===undefined){fetchCandidates(it);}
  if(it._cands===undefined||it._cands===null)
    return `<div class=prop-box><div class=prop-head>${head}</div><div class=step-hint>finding neighbors…</div></div>`;
  if(!it._cands.length)
    return `<div class=prop-box><div class=prop-head>${head}</div><div class=step-hint>No unlabeled neighbors found.</div></div>`;
  const selN=it._cands.filter(c=>c._sel).length;
  const rows=it._cands.map((c,i)=>`
    <div class="cand${c._sel?' sel':''}" onclick="toggleCand(${i})">
      <button class=cand-play onclick="event.stopPropagation();cplay(${i})" title="audition">${_cPlaying===i?'■':'▶'}</button>
      <span class=cand-chk>${c._sel?'☑':'☐'}</span>
      <span class=cand-name>${c.name}</span>
      <span class=cand-meta>${c.model_instrument?`<span class=pill style="background:${col(c.model_instrument)}">${c.model_instrument}</span>`:''}<span class=cand-score>${(c.score*100).toFixed(0)}%</span></span>
    </div>`).join('');
  return `<div class=prop-box>
    <div class=prop-head>${head}
      <span class=prop-tools><a href=# onclick="candAll(true);return false">all</a> · <a href=# onclick="candAll(false);return false">none</a></span></div>
    <div class=cand-list>${rows}</div>
    <div class=step-hint>${selN} selected → will get <strong>${it.human_instrument}</strong> on submit. ▶ audition · click row to toggle.</div>
  </div>`;
}

async function fetchCandidates(it){
  it._cands=null;                                   // loading
  try{
    const d=await fetch('/api/propagate?k=24&path='+encodeURIComponent(it.path)).then(r=>r.json());
    const cands=d.items||[];
    cands.forEach(c=>{ c._sel = c.score>=0.93; });  // pre-select very-close neighbors
    it._cands=cands;
  }catch(e){ it._cands=[]; }
  if(queue[cur]===it && step===3) renderStepPanel();
}

// Update only the play-button glyphs in place — a full renderStepPanel() rebuilds
// the panel's innerHTML and resets the candidate list's scroll position.
function refreshCandPlay(){
  document.querySelectorAll('.cand-list .cand-play').forEach((b,i)=>{
    b.textContent=(_cPlaying===i)?'■':'▶';
  });
}

function cplay(i){
  const it=queue[cur]; if(!it||!it._cands)return;
  const c=it._cands[i]; if(!c)return;
  if(!_cAudio){
    _cAudio=new Audio();
    _cAudio.onended=()=>{_cPlaying=-1;refreshCandPlay();};
  }
  if(_cPlaying===i&&!_cAudio.paused){
    _cAudio.pause();_cPlaying=-1;refreshCandPlay();return;
  }
  _cAudio.pause();
  _cPlaying=i;
  _cAudio.src='/api/audio?norm=1&path='+encodeURIComponent(c.path);
  _cAudio.play().catch(()=>{_cPlaying=-1;refreshCandPlay();});
  refreshCandPlay();
}

function toggleCand(i){
  const it=queue[cur]; if(!it||!it._cands||!it._cands[i])return;
  const c=it._cands[i];
  c._sel=!c._sel;
  // Update the row in place to preserve the list's scroll position.
  const list=document.querySelector('.cand-list');
  const row=list&&list.children[i];
  if(!row){renderStepPanel();return;}
  row.classList.toggle('sel',c._sel);
  const chk=row.querySelector('.cand-chk'); if(chk)chk.textContent=c._sel?'☑':'☐';
  const selN=it._cands.filter(x=>x._sel).length;
  const hint=document.querySelector('.prop-box .step-hint');
  if(hint)hint.innerHTML=`${selN} selected → will get <strong>${it.human_instrument}</strong> on submit. ▶ audition · click row to toggle.`;
}

function candAll(v){
  const it=queue[cur]; if(!it||!it._cands)return;
  it._cands.forEach(c=>c._sel=v);
  // In-place update to preserve scroll position (see toggleCand).
  const list=document.querySelector('.cand-list');
  if(!list){renderStepPanel();return;}
  Array.from(list.children).forEach(row=>{
    row.classList.toggle('sel',v);
    const chk=row.querySelector('.cand-chk'); if(chk)chk.textContent=v?'☑':'☐';
  });
  const selN=v?it._cands.length:0;
  const hint=document.querySelector('.prop-box .step-hint');
  if(hint)hint.innerHTML=`${selN} selected → will get <strong>${it.human_instrument}</strong> on submit. ▶ audition · click row to toggle.`;
}

/* ---------- actions (each saves immediately) ---------- */

function rate(n){
  const it=queue[cur]; if(!it)return;
  if((it.rating||0)===n) n=0;            // click the current star to clear
  it.rating=n;
  fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:it.path,rating:n})});
  const el=document.getElementById('stars'); if(el)el.innerHTML=starInner(n);
  renderStepper(); renderList();
  if(n>0) goStep(1);
}

function saveType(sample_type){
  const it=queue[cur]; if(!it)return;
  it.human_sample_type=sample_type;
  showToast('✓ '+sample_type, typeColor(sample_type));
  fetch('/api/label_type',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:it.path,sample_type})});
  renderStepper();
  goStep(2);
}

function save(instrument){
  const it=queue[cur]; if(!it)return;
  if(!it.human_instrument && instrument) sessionCount++;
  it._done=true; it.human_instrument=instrument;
  lblBuf='';
  showToast('✓ '+instrument, col(instrument));
  fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:it.path,instrument})});
  renderList(); renderStepper(); updatePos();
  goStep(3);
}

function submit(){
  const it=queue[cur]; if(!it)return;
  const sel=(it._cands||[]).filter(c=>c._sel).map(c=>c.path);
  if(it.human_instrument && sel.length){
    fetch('/api/label_propagate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({paths:sel,instrument:it.human_instrument})});
    sessionCount+=sel.length;
    const sset=new Set(sel);
    queue.forEach(q=>{ if(sset.has(q.path)){q.human_instrument=it.human_instrument;q._done=true;} });
    showToast(`✓ +${sel.length} propagated`, '#a6e22e');
  } else {
    showToast('✓ submitted', '#a6e22e');
  }
  if(_cAudio){_cAudio.pause();_cPlaying=-1;}
  it._done=true;
  renderList(); updatePos();
  next();
}

function undo() {
  const it=queue[cur];
  if(!it)return;
  if (it.human_instrument) {
    it.human_instrument = null;
    it._done = false;
    sessionCount--;
    fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:it.path,instrument:''})});
    renderList(); updatePos();
  }
  goStep(Math.max(0, step-1));
}

/* ---------- navigation ---------- */

function select(i){
  if(_cAudio){_cAudio.pause();_cPlaying=-1;}
  cur=i;
  step=0;
  lblBuf='';
  renderList();
  renderDetail(queue[i],i);
  updatePos();
  closeOverlay();
  const el=document.getElementById('list').children[i];
  if(el)el.scrollIntoView({block:'nearest'});
}

function updatePos(){
  const el=document.getElementById('pos');
  if(el)el.textContent = cur>=0 ? `${cur+1} / ${queue.length} · Labeled: ${sessionCount}` : '';
}

function skip() {
  if (cur < 0) return;
  queue[cur]._seen = true;
  next();
}

function next(){
  if(cur<queue.length-1) select(cur+1);
  else if(cur===queue.length-1) loadQueue();
}
function prev(){if(cur>0)select(cur-1);}
function openOverlay(){document.getElementById('overlay').classList.add('open');}
function closeOverlay(){document.getElementById('overlay').classList.remove('open');}

async function loadQueue(){
  document.getElementById('list').innerHTML='<div class=empty>Loading…</div>';
  document.getElementById('overlay-list').innerHTML='';
  document.getElementById('detail').innerHTML='<div class=empty>Select a sample.</div>';
  cur=-1; queue=[]; step=0;
  const mode=document.getElementById('modesel').value;
  const d=await fetch('/api/review/queue?mode='+mode).then(r=>r.json());
  queue=d.items||[];
  const ctxt=`${queue.length} loaded / ${(d.total||0).toLocaleString()} total`;
  document.getElementById('count').textContent=ctxt;
  document.getElementById('overlay-count').textContent=ctxt;
  renderList();
  if(queue.length)select(0);
  else {document.getElementById('detail').innerHTML='<div class=empty>Nothing to review in this mode.</div>';updatePos();}
}

/* ---------- keyboard ---------- */

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  const it=queue[cur];

  // global
  if(e.key===' '){e.preventDefault();replay();return;}
  if(e.shiftKey && e.key==='ArrowLeft'){e.preventDefault();prev();return;}
  if(e.shiftKey && e.key==='ArrowRight'){e.preventDefault();skip();return;}
  if(e.key==='ArrowRight'){e.preventDefault();goStep(step+1);return;}
  if(e.key==='ArrowLeft'){e.preventDefault();goStep(step-1);return;}
  if(!it)return;

  if(step===0){                                   // rate
    if(e.key>='1' && e.key<='5'){rate(parseInt(e.key));}
    else if(e.key==='Enter'){goStep(1);}
    else if(e.key==='s'){skip();}
    else if(e.key==='z'){undo();}
  } else if(step===1){                            // type
    if(e.key==='1'){saveType('oneshot');}
    else if(e.key==='2'){saveType('loop');}
    else if(e.key==='Enter'){goStep(2);}
    else if(e.key==='s'){skip();}
    else if(e.key==='z'){goStep(0);}
  } else if(step===2){                            // label (letters reserved for filter)
    if(e.key==='Enter'){
      if(lblBuf){const m=INSTRUMENTS.find(x=>x.startsWith(lblBuf)); if(m)save(m); else {lblBuf='';renderStepPanel();}}
      else {const s=suggestedFor(it); if(s)save(s); else goStep(3);}
    } else if(e.key==='Escape'){lblBuf='';renderStepPanel();}
    else if(e.key==='Backspace'){e.preventDefault();
      if(lblBuf){lblBuf=lblBuf.slice(0,-1);renderStepPanel();} else goStep(1);}
    else if(e.key.length===1 && /[a-z]/i.test(e.key)){
      if(e.ctrlKey||e.metaKey||e.altKey)return;
      lblBuf+=e.key.toLowerCase(); renderStepPanel();}
  } else if(step===3){                            // submit
    if(e.key==='Enter'){submit();}
    else if(e.key==='s'){skip();}
    else if(e.key==='z'){goStep(2);}
  }
});

fetchLabels().then(()=>loadQueue());
