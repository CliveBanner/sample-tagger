let COLORS={};
const DIM='#3a3a35';

function autoColor(i, n){ return `hsl(${Math.round(i*360/n)},60%,55%)`; }
function pie(items, sz=140){
  const total=items.reduce((s,i)=>s+i.n,0);
  if(!total)return '<span class=muted>no data yet</span>';
  const cx=sz/2,cy=sz/2,r=sz/2-2;
  let a=-Math.PI/2,paths='',leg='';
  const n=items.filter(i=>i.n).length;
  let ci=0;
  for(const it of items){
    if(!it.n)continue;
    const c=it.color||(COLORS[it.label])||autoColor(ci++,n);
    const sw=it.n/total*2*Math.PI;
    const pct=Math.round(it.n/total*100);
    if(sw>=2*Math.PI-0.001){
      paths+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${c}"/>`;
    } else {
      const x1=cx+r*Math.cos(a),y1=cy+r*Math.sin(a);
      a+=sw;
      const x2=cx+r*Math.cos(a),y2=cy+r*Math.sin(a);
      paths+=`<path d="M${cx},${cy}L${x1.toFixed(1)},${y1.toFixed(1)}A${r},${r},0,${sw>Math.PI?1:0},1,${x2.toFixed(1)},${y2.toFixed(1)}Z" fill="${c}"><title>${it.label}: ${it.n.toLocaleString()} (${pct}%)</title></path>`;
    }
    leg+=`<div style="display:flex;align-items:center;gap:5px;margin:2px 0">
      <span style="display:inline-block;width:9px;height:9px;background:${c};border-radius:2px;flex-shrink:0"></span>
      <span style="color:var(--fg)">${it.label}</span>
      <span class=muted style="margin-left:auto;padding-left:8px">${pct}%</span>
    </div>`;
  }
  return `<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
    <svg width="${sz}" height="${sz}" style="flex-shrink:0">${paths}</svg>
    <div style="font-size:12px;flex:1;min-width:80px">${leg}</div>
  </div>`;
}

function classifierCard(title, cov, dist, hitColor, missLabel){
  const n=cov?cov.n:0, tot=cov?cov.total:0;
  const pct=tot?Math.round(n/tot*100):0;
  const covItems=[
    {label:'labeled', n, color:hitColor},
    {label:missLabel, n:tot-n, color:DIM}
  ];
  const header=`<div style="display:flex;align-items:baseline;gap:10px;
    padding-bottom:10px;margin-bottom:14px;border-bottom:1px solid #3e3d32">
    <span style="font-size:28px;font-weight:bold;color:${hitColor}">${pct}%</span>
    <span class=muted style="font-size:12px">${n.toLocaleString()} / ${tot.toLocaleString()}</span>
  </div>`;
  const body=`<div style="display:flex;gap:16px;flex-wrap:wrap">
    <div style="flex:0 0 auto">${pie(covItems,100)}</div>
    <div style="flex:1;min-width:120px;border-left:1px solid #3e3d32;padding-left:16px">${pie(dist,120)}</div>
  </div>`;
  return card(title, header+body);
}

function bars(items){
  if(!items||!items.length)return '<span class=muted>no data yet</span>';
  const max=Math.max(...items.map(i=>i.n));
  return items.map(i=>{const c=COLORS[i.label]||'#ae81ff';
    return `<div class=bar><span class=name title="${i.label}">${i.label}</span>
    <span class=track><span class=fill style="width:${100*i.n/max}%;background:${c}"></span></span>
    <span class=num>${i.n.toLocaleString()}</span></div>`}).join('');
}
function card(title,inner,span){return `<div class="card${span?' span2':''}">
  <h2>${title}</h2>${inner}</div>`}

let _lastTick=0;
function updateClock(){
  if(!_lastTick)return;
  const age=Math.round((Date.now()-_lastTick)/1000);
  document.getElementById('clock').textContent='updated '+(age===0?'just now':age+'s ago');
}
setInterval(updateClock,1000);

async function tick(){
  if(Object.keys(COLORS).length===0){try{COLORS=await(await fetch('/api/colors')).json()}catch(e){}}
  let s; try{s=await (await fetch('/api/stats')).json()}catch(e){return}
  _lastTick=Date.now(); updateClock();
  const st=document.getElementById('status');
  if(!s.ready){st.innerHTML='<span class="badge done">waiting for db</span>';
    document.getElementById('app').innerHTML=card('Status',s.msg||'');return}
  st.innerHTML = s.scan_running
    ? '<span class="badge run">SCANNING</span>'
    : '<span class="badge done">IDLE / DONE</span>';
  const eta=s.eta_min!=null?(s.eta_min>60?(s.eta_min/60).toFixed(1)+' h':s.eta_min+' min'):'—';
  const anyN=(s.coverage&&s.coverage.any)?s.coverage.any.n:0;
  const kpis=`<div class=kpis>
    <div class=kpi><div class=v>${(s.active||0).toLocaleString()}</div><div class=l>total files</div></div>
    <div class=kpi><div class=v>${anyN.toLocaleString()}</div><div class=l>any label</div></div>
    <div class=kpi><div class=v>${(s.coverage&&s.coverage.human)?s.coverage.human.n:0}</div><div class=l>human corrections</div></div>
    <div class=kpi><div class=v>${(s.errors||0).toLocaleString()}</div><div class=l>errors</div></div>
    </div>
    ${s.label_total?`<div style="margin-top:12px">
      <div class=prog><div style="width:${s.pct||0}%"></div></div>
      <div class=muted style="font-size:12px;margin-top:4px">label stage: ${(s.label_done||0).toLocaleString()} / ${s.label_total.toLocaleString()} · ${s.rate} f/s · eta ${eta}</div>
    </div>`:''}`;
  const app=document.getElementById('app');
  const cov=s.coverage||{};
  const sonic=s.sonic_dist||[];
  const sonicCard=card('Sonic families — audio-only clusters',
    (sonic.length
      ? `<div class=muted style="font-size:12px;margin-bottom:8px">${sonic.length} families grouped by timbre; labels describe sound, not instrument.</div>${bars(sonic)}`
      : '<span class=muted>not computed yet — run scripts/sonic_label.py</span>'), true);
  app.innerHTML=
    card('Progress',kpis,true)+
    sonicCard+
    classifierCard('Human labels', cov.human, s.human_dist||[], '#f92672', 'unlabeled')+
    (cov.model ? classifierCard('Model predictions', cov.model, s.model_dist||[], '#e6db74', 'no prediction') : '')+
    (s.model_conf_hist && s.model_conf_hist.length ? card('Model confidence', bars(s.model_conf_hist)) : '')+
    classifierCard('PANNs (mapped)', cov.panns, s.panns_dist||[], '#ae81ff', 'pending')+
    card('PANNs raw (AudioSet)', pie(s.panns_raw_dist||[]))+
    classifierCard('Path',  cov.path,  s.path_dist||[],  '#a6e22e', 'no path hint')+
    card('Sample type',pie(s.sample_type||[]))+
    card('Key distribution',bars(s.keys||[]))+
    card('BPM (loops)',bars(s.bpm_hist));
  const lb=document.getElementById('logbox');
  if(lb)lb.textContent=(s.log_tail||[]).join('\n')||'(no log yet)';
}
tick();setInterval(tick,4000);