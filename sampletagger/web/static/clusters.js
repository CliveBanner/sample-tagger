// Sonic cluster browser: coarse families -> fine grains -> audition members.
// All groupings/labels are audio-only (DSP descriptors), independent of filenames
// and the instrument taxonomy.
let FAMILIES = [], curFamily = null, curGrain = null;

function fmtAxes(g){
  return `<span class=muted style="font-size:11px">`
    + `${g.centroid|0}Hz · flat ${g.flatness} · atk ${g.attack}s · ${g.duration}s`
    + `</span>`;
}

async function loadFamilies(){
  const d = await (await fetch('/api/sonic/families')).json();
  FAMILIES = d.families || [];
  const list = document.getElementById('list');
  if(!FAMILIES.length){
    list.innerHTML = `<div class=empty style="padding:14px">`
      + (d.pending ? 'Sonic families not computed yet — run <code>scripts/sonic_label.py</code>.'
                   : 'No sonic clusters found.') + `</div>`;
    document.getElementById('detail').innerHTML = '<div class=empty>—</div>';
    return;
  }
  list.innerHTML = FAMILIES.map(f=>`
    <div class="cl-row${curFamily===f.id?' active':''}" onclick="selectFamily(${f.id})">
      <div class=cl-main><span class=cl-label>${f.label}</span></div>
      <div class=cl-sub>${f.n.toLocaleString()} samples · ${fmtAxes(f)}</div>
    </div>`).join('');
  if(curFamily===null && FAMILIES.length) selectFamily(FAMILIES[0].id);
}

async function selectFamily(id){
  curFamily = id; curGrain = null;
  document.querySelectorAll('#list .cl-row').forEach((el,i)=>
    el.classList.toggle('active', FAMILIES[i] && FAMILIES[i].id===id));
  const d = await (await fetch('/api/sonic/grains?family='+id)).json();
  const grains = d.grains || [];
  const fam = FAMILIES.find(f=>f.id===id);
  document.getElementById('detail').innerHTML = `
    <div class=cl-head><b>${fam?fam.label:'family '+id}</b> — ${grains.length} grains</div>
    <div id=grains>${grains.map(g=>`
      <div class="grain" onclick="selectGrain(${g.id})">
        <div><span class=cl-label>${g.label}</span> <span class=muted>(${g.n})</span>
        ${g.medoid?`<button class=cand-play onclick="event.stopPropagation();playPath('${encodeURIComponent(g.medoid)}','${(g.medoid_name||'').replace(/'/g,"")}')" title="audition medoid">▶</button>`:''}</div>
        <div>${fmtAxes(g)}</div>
      </div>`).join('')}</div>
    <div id=members></div>`;
}

async function selectGrain(id){
  curGrain = id;
  const d = await (await fetch('/api/sonic/members?grain='+id)).json();
  const m = d.members || [];
  document.getElementById('members').innerHTML = `
    <div class=cl-head style="margin-top:10px">grain ${id} — ${m.length} members (closest to core first)</div>
    ${m.map(x=>`
      <div class=cand onclick="playPath('${encodeURIComponent(x.path)}','${(x.name||'').replace(/'/g,"")}')">
        <button class=cand-play onclick="event.stopPropagation();playPath('${encodeURIComponent(x.path)}','${(x.name||'').replace(/'/g,"")}')">▶</button>
        <span class=cand-name>${x.name}</span>
        <span class=cand-meta><span class=cand-score>${x.duration_s?x.duration_s+'s':''}</span></span>
      </div>`).join('')}`;
}

function playPath(encPath, name){
  const a = document.getElementById('cplayer');
  a.src = '/api/audio?norm=1&path=' + encPath;
  a.play().catch(()=>{});
  const now = document.getElementById('cnow');
  if(now){ now.textContent = '▶ ' + (name || decodeURIComponent(encPath).split('/').pop()); now.classList.remove('dim'); }
}

loadFamilies();
