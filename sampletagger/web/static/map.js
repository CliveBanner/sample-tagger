const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const dpr=window.devicePixelRatio||1;
let M=null,sel=-1,scale=1,tx=0,ty=0,base=1,actC=null,actFam=null,actT=null,actS=null,minD=0,maxD=0,stride=1,viewBy='instrument';
let selectMode=false, selPoly=null, selIdx=new Set();
// The map shows one field at a time. Instrument-label views (resolved/human/model/
// path/panns/audio) share the instrument taxonomy (cats/colors, filter set actC);
// the 'family' view is the audio-only sonic families (famCats/famColors, set actFam).
// catOf(i) is the active field's category index; <len> is the "none" bucket.
function curCats(){return viewBy==='family'?(M.famCats||[]):M.cats;}
function curColors(){return viewBy==='family'?(M.famColors||[]):M.colors;}
function curActC(){return viewBy==='family'?actFam:actC;}
function catOf(i){return M.fields[viewBy][i];}
function dotColor(i){const c=catOf(i),cc=curCats();return c>=cc.length?'#4a4a42':(curColors()[c]||'#888');}
// Optimistic local update after labeling: a human label sets both the resolved
// instrument and the human field, plus its provenance, so the dot recolors in any
// view immediately (the DB reconciles fully on the next predict/map rebuild).
function setLocalLabel(i,ci,prov){M.fields.instrument[i]=ci;M.fields.human[i]=ci;if(M.ls)M.ls[i]=prov;}
function shown(i){
  if(!curActC().has(catOf(i))||!actT.has(M.t?M.t[i]:0))return false;
  if(actS&&!actS.has(M.ls?M.ls[i]:5))return false;   // provenance (label_source)
  const d=M.d?M.d[i]:0;
  if(d>0){if(minD>0&&d<minD)return false;if(maxD>0&&d>maxD)return false;}
  return true;
}
function resize(){cv.width=cv.clientWidth*dpr;cv.height=cv.clientHeight*dpr;
  base=Math.min(cv.width,cv.height);draw();}
window.addEventListener('resize',resize);
function sx(i){return tx+M.x[i]*base*scale;}
function sy(i){return ty+(1-M.y[i])*base*scale;}
function fit(){scale=0.92;tx=(cv.width-base*scale)/2;ty=(cv.height-base*scale)/2;}
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if(!M)return;
  const r=Math.max(1.2,1.6*Math.sqrt(scale))*dpr,W=cv.width,H=cv.height;
  // fraction of total map area currently visible; clamp to 1 when zoomed out
  const visF=Math.min(1,W/(base*scale))*Math.min(1,H/(base*scale));
  stride=Math.max(1,Math.ceil(M.n*visF/20000));
  for(let i=0;i<M.n;i+=stride){if(!shown(i))continue;const X=sx(i),Y=sy(i);
    if(X<-2||X>W+2||Y<-2||Y>H+2)continue;
    ctx.fillStyle=dotColor(i);ctx.fillRect(X-r/2,Y-r/2,r,r);}
  if(sel>=0){
    if(stride>1&&shown(sel)){const X=sx(sel),Y=sy(sel);
      ctx.fillStyle=dotColor(sel);ctx.fillRect(X-r/2,Y-r/2,r,r);}
    ctx.strokeStyle='#fff';ctx.lineWidth=2*dpr;ctx.beginPath();
    ctx.arc(sx(sel),sy(sel),8*dpr,0,7);ctx.stroke();}
    
  if (selIdx.size > 0) {
    ctx.fillStyle = '#fff';
    for (let i of selIdx) {
      if(stride>1 && !shown(i)) continue;
      const X=sx(i), Y=sy(i);
      ctx.fillRect(X-r/2, Y-r/2, r, r);
    }
  }
  if (selPoly && selPoly.length > 0) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = dpr;
    ctx.beginPath();
    ctx.moveTo(selPoly[0][0], selPoly[0][1]);
    for (let i = 1; i < selPoly.length; i++) ctx.lineTo(selPoly[i][0], selPoly[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}
// Convert client coords to canvas device-pixel coords (canvas-relative)
function cvPos(cx,cy){const r=cv.getBoundingClientRect();
  return [(cx-r.left)*dpr,(cy-r.top)*dpr];}
// --- mouse ---
cv.addEventListener('wheel',e=>{e.preventDefault();
  const f=e.deltaY<0?1.15:1/1.15,[mx,my]=cvPos(e.clientX,e.clientY);
  tx=mx-(mx-tx)*f;ty=my-(my-ty)*f;scale*=f;draw();},{passive:false});
let down=null,moved=false;
cv.addEventListener('mousedown',e=>{
  if (selectMode || e.shiftKey) {
    const [x, y] = cvPos(e.clientX, e.clientY);
    selPoly = [[x, y]];
    down = null;
  } else {
    down=[...cvPos(e.clientX,e.clientY),tx,ty];moved=false;
  }
});
window.addEventListener('mousemove',e=>{
  if (selPoly) {
    const [x, y] = cvPos(e.clientX, e.clientY);
    const last = selPoly[selPoly.length - 1];
    if (Math.hypot(x - last[0], y - last[1]) > 4 * dpr) {
      selPoly.push([x, y]);
      draw();
    }
    return;
  }
  if(!down)return;
  const[mx,my]=cvPos(e.clientX,e.clientY);
  if(Math.abs(mx-down[0])+Math.abs(my-down[1])>4)moved=true;
  tx=down[2]+(mx-down[0]);ty=down[3]+(my-down[1]);draw();});
window.addEventListener('mouseup', e => {
  if (selPoly) {
    if (!e.shiftKey) selIdx.clear();
    if (selPoly.length > 2) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let p of selPoly) {
        if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
      }
      for (let i = 0; i < M.n; i++) {
        if (!shown(i)) continue;
        const X = sx(i), Y = sy(i);
        if (X < minX || X > maxX || Y < minY || Y > maxY) continue;
        let inside = false;
        for (let j = 0, k = selPoly.length - 1; j < selPoly.length; k = j++) {
          const xj = selPoly[j][0], yj = selPoly[j][1];
          const xk = selPoly[k][0], yk = selPoly[k][1];
          const intersect = ((yj > Y) !== (yk > Y)) && (X < (xk - xj) * (Y - yj) / (yk - yj) + xj);
          if (intersect) inside = !inside;
        }
        if (inside) selIdx.add(i);
      }
    }
    selPoly = null;
    draw();
    updateBatchPanel();
    return;
  }
  if(down&&!moved)pick(down[0],down[1]);down=null;
});
// --- touch ---
let t0=null,t1=null,pd=0;
function tPos(t){return cvPos(t.clientX,t.clientY);}
cv.addEventListener('touchstart',e=>{e.preventDefault();
  if(e.touches.length===1){
    const[px,py]=tPos(e.touches[0]);
    down=[px,py,tx,ty];moved=false;t0=e.touches[0];t1=null;
  }else if(e.touches.length===2){
    down=null;t0=e.touches[0];t1=e.touches[1];
    pd=Math.hypot(t1.clientX-t0.clientX,t1.clientY-t0.clientY);
  }},{passive:false});
cv.addEventListener('touchmove',e=>{e.preventDefault();
  if(e.touches.length===1&&down){
    const[mx,my]=tPos(e.touches[0]);
    if(Math.abs(mx-down[0])+Math.abs(my-down[1])>4)moved=true;
    tx=down[2]+(mx-down[0]);ty=down[3]+(my-down[1]);draw();
  }else if(e.touches.length===2&&t0&&t1){
    const a=e.touches[0],b=e.touches[1];
    const nd=Math.hypot(b.clientX-a.clientX,b.clientY-a.clientY);
    const f=nd/pd;
    const[cx,cy]=cvPos((a.clientX+b.clientX)/2,(a.clientY+b.clientY)/2);
    tx=cx-(cx-tx)*f;ty=cy-(cy-ty)*f;scale*=f;pd=nd;t0=a;t1=b;draw();
  }},{passive:false});
cv.addEventListener('touchend',e=>{e.preventDefault();
  if(e.touches.length===0&&down&&!moved)pick(down[0],down[1]);
  if(e.touches.length<1){down=null;}
  if(e.touches.length<2){t0=null;t1=null;}},{passive:false});
function pick(px,py){let best=-1,bd=16*dpr*16*dpr;
  for(let i=0;i<M.n;i++){if(!shown(i))continue;
    const dx=sx(i)-px,dy=sy(i)-py,d=dx*dx+dy*dy;
    if(d<bd){bd=d;best=i;}}
  if(best>=0){sel=best;draw();inspect(best);}
  else { selIdx.clear(); updateBatchPanel(); draw(); }
}
function updateBatchPanel() {
  const panel = document.getElementById('batchPanel');
  if(selIdx.size > 0) {
    panel.style.display = 'block';
    document.getElementById('batchCount').textContent = selIdx.size + ' selected';
  } else {
    panel.style.display = 'none';
  }
}
const FADE_MS=180;
let xfading=false;
function playPath(path){
  const a=document.getElementById('player');
  const x=document.getElementById('xplayer');
  const auto=document.getElementById('auto').checked;
  const url='/api/audio?path='+encodeURIComponent(path);
  // abort any in-progress crossfade
  if(xfading){x.pause();x.src='';x.load();x.volume=1;xfading=false;a.volume=1;}
  // no crossfade if nothing is playing
  if(a.paused||a.ended||!a.src){
    a.src=url;if(auto)a.play().catch(()=>{});return;}
  // crossfade: outgoing → x, incoming → a
  const vol=a.volume;
  x.src=a.src;x.currentTime=a.currentTime;x.volume=vol;x.play().catch(()=>{});
  a.volume=0;a.src=url;if(auto)a.play().catch(()=>{});
  xfading=true;
  const t0=performance.now();
  const tick=()=>{
    const p=Math.min(1,(performance.now()-t0)/FADE_MS);
    x.volume=vol*(1-p);a.volume=vol*p;
    if(p<1){requestAnimationFrame(tick);}
    else{x.pause();x.src='';x.load();x.volume=1;xfading=false;}};
  requestAnimationFrame(tick);
}
let _selPath=null;
async function inspect(i){const p=await(await fetch('/api/point?i='+i)).json();
  _selPath=p.path;showSel(p);playPath(p.path);
  document.getElementById('btnSim').disabled=false;
  document.getElementById('hits').innerHTML='<span class=muted>—</span>';}
document.getElementById('btnSim').onclick=()=>{
  if(_selPath)loadSimilar('path='+encodeURIComponent(_selPath));};
document.getElementById('btnLabel').onclick=async()=>{
  const instr=document.getElementById('labelSel').value;
  if(!_selPath||!instr)return;
  const btn=document.getElementById('btnLabel');
  btn.disabled=true;
  const r=await fetch('/api/label',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:_selPath,instrument:instr})});
  const d=await r.json();
  const msg=document.getElementById('labelMsg');
  if(d.ok){
    msg.style.color='#a6e22e';msg.textContent='Saved ✓';
    // update the pill in the sel panel immediately
    const pill=document.querySelector('#sel .pill');
    if(pill)pill.textContent=instr;
    // update map dot color (single label → provenance 'single')
    if(sel>=0){const ci=M.cats.indexOf(instr);if(ci>=0)setLocalLabel(sel,ci,0);draw();}
  } else {
    msg.style.color='#f92672';msg.textContent=d.msg||'error';}
  btn.disabled=false;};
const SRC_LABEL={'path':'via path','panns':'via PANNs','audio':'via audio','none':'unknown','human':'✎ human'};
const SRC_COLOR={'human':'#f6d860'};
let _selRating=0;
function mapStarInner(rating){
  let h='';
  for(let n=1;n<=5;n++){const on=n<=rating;
    h+=`<span class="star${on?' on':''}" onclick="rateMap(${n})">${on?'★':'☆'}</span>`;}
  return h;
}
function rateMap(n){
  if(!_selPath)return;
  if(_selRating===n)n=0;
  _selRating=n;
  fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:_selPath,rating:n})});
  const el=document.getElementById('mstars');if(el)el.innerHTML=mapStarInner(n);}
function showSel(p){
  const el=document.getElementById('sel');el.classList.remove('muted');
  _selRating=p.rating||0;
  const srcCol=SRC_COLOR[p.source]||'var(--dim)';
  // build classification breakdown rows
  const conf=p.panns_conf?` <span style="color:var(--dim);font-size:10px">${(p.panns_conf*100).toFixed(0)}%</span>`:'';
  const rawConf=p.panns_label_conf?` <span style="color:var(--dim);font-size:10px">${(p.panns_label_conf*100).toFixed(0)}%</span>`:'';
  const rawTip=(p.panns_topk||[]).map(t=>`${t[0]} ${(t[1]*100).toFixed(0)}%`).join(' · ');
  const rows=[
    p.path_instrument  ?`<tr><td style="color:var(--dim);padding-right:8px">path</td><td>${p.path_instrument}</td></tr>`:'',
    p.panns_instrument ?`<tr><td style="color:var(--dim);padding-right:8px">PANNs</td><td>${p.panns_instrument}${conf}</td></tr>`:'',
    p.audio_instrument ?`<tr><td style="color:var(--dim);padding-right:8px">audio</td><td>${p.audio_instrument}</td></tr>`:'',
    (p.model_labels&&p.model_labels.length)
      ?`<tr><td style="color:var(--dim);padding-right:8px">model</td><td>${p.model_labels.map(x=>`${x[0]} <span style="color:var(--dim);font-size:10px">${(x[1]*100).toFixed(0)}%</span>`).join(' · ')}</td></tr>`
      :(p.model_instrument?`<tr><td style="color:var(--dim);padding-right:8px">model</td><td>${p.model_instrument}${p.model_conf?` <span style="color:var(--dim);font-size:10px">${(p.model_conf*100).toFixed(0)}%</span>`:''}</td></tr>`:''),
    (p.human_labels&&p.human_labels.length>1)
      ?`<tr><td style="color:var(--dim);padding-right:8px">human</td><td>${p.human_labels.join(' + ')}</td></tr>`:'',
    p.panns_label      ?`<tr><td style="color:var(--dim);padding-right:8px">raw</td><td title="${rawTip}">${p.panns_label}${rawConf}</td></tr>`:'',
  ].join('');
  const srcCol2=SRC_COLOR[p.source]||'var(--dim)';
  el.innerHTML=`<div style="word-break:break-all;font-weight:500">${p.name||'?'}</div>
  <div style="margin-top:5px">
  <span class=pill style="font-weight:600">${p.instrument||'?'}</span><span class=pill>${p.sample_type||''}</span>
  ${p.bpm?`<span class=pill>${p.bpm} bpm</span>`:''}
  ${p.key?`<span class=pill>${p.key}</span>`:''}
  ${p.duration_s?`<span class=pill>${p.duration_s}s</span>`:''}
  <span class=pill style="color:${srcCol2}">${SRC_LABEL[p.source]||p.source||''}</span></div>
  <div class=rate-row style="margin-top:6px"><span class=rate-lbl>Rating</span><span id=mstars class=stars title="rate 1-5 · click again to clear">${mapStarInner(p.rating||0)}</span></div>
  ${p.sonic?`<div style="margin-top:6px;font-size:11px" title="audio-only sonic descriptor (no filename / taxonomy)"><span style="color:var(--dim)">sonic</span> <span style="color:#a6e22e;font-weight:600">${p.sonic.grain}</span>${p.sonic.family?` <span style="color:var(--dim)">· ${p.sonic.family} family</span>`:''}</div>`:''}
  ${rows?`<table style="margin-top:6px;font-size:11px;border-collapse:collapse">${rows}</table>`:''}`
  // pre-fill the label dropdown with the current instrument
  const lsel=document.getElementById('labelSel');
  lsel.value=p.instrument||'';
  document.getElementById('labelRow').style.display='flex';
  document.getElementById('labelMsg').textContent='';
  document.getElementById('btnLabel').disabled=false;}
async function loadSimilar(qs){
  const h=document.getElementById('hits');h.innerHTML='<span class=muted>…</span>';
  const d=await(await fetch('/api/similar?k=24&'+qs)).json();
  if(!d.matched){h.innerHTML='<span class=muted>no match ('+d.n+' indexed)</span>';return;}
  h.classList.remove('muted');
  h.innerHTML=d.hits.map(x=>`<div class=hit data-p="${encodeURIComponent(x.path)}">
    <span class=s>${x.score}</span>${x.name}
    <div class=muted>${x.instrument||''} ${x.sample_type||''} ${x.bpm?x.bpm+'bpm':''} ${x.key||''}</div>
    </div>`).join('');
  h.querySelectorAll('.hit').forEach(el=>el.onclick=()=>{
    const path=decodeURIComponent(el.dataset.p);
    _selPath=path;showSelName(path);playPath(path);loadSimilar('path='+el.dataset.p);});}
function showSelName(path){const el=document.getElementById('sel');
  el.classList.remove('muted');
  el.innerHTML='<div style="word-break:break-all">'+path.split('/').pop()+'</div>';}

/* ---------- side tabs & search tab ---------- */

function switchSideTab(name){
  document.getElementById('tab-detail').style.display=(name==='detail')?'':'none';
  document.getElementById('tab-search').style.display=(name==='search')?'':'none';
  document.querySelectorAll('#sideTabs .stab').forEach(b=>
    b.classList.toggle('active',b.dataset.tab===name));
}

let _playingRow=null;
function playFromRow(el,path){
  // reuse the main player (keeps the crossfade); mark the active row
  _selPath=path;showSelName(path);playPath(path);
  if(_playingRow)_playingRow.textContent='▶';
  el.textContent='■';_playingRow=el;
}

function searchLocate(path){
  const idx=M&&M.paths?M.paths.indexOf(path):-1;
  if(idx>=0){sel=idx;scrollToSel();inspect(idx);}     // full detail incl. labels
  else{_selPath=path;showSelName(path);}              // not on the map (yet)
  switchSideTab('detail');
}

function searchRow(x){
  const hl=(x.human_labels||[]).map(l=>`<span class=pill style="background:#a6e22e22;border:1px solid #a6e22e;color:#a6e22e">${l}</span>`).join('');
  const ml=(x.model_labels||[]).slice(0,2).map(([l,c])=>`<span class=pill style="background:#66d9ef22;border:1px solid #66d9ef55;color:#66d9ef">${l} ${(c*100).toFixed(0)}%</span>`).join('');
  return `<div class="hit srow" data-p="${encodeURIComponent(x.path)}">
    <button class=hit-play title=audition>▶</button>
    <span class=s>${x.score}</span><span class=srow-name>${x.name}</span>
    <div class=srow-meta>${hl}${ml}
      <span class=muted>${x.sample_type||''} ${x.duration_s?x.duration_s.toFixed(1)+'s':''} ${x.bpm?x.bpm+'bpm':''} ${x.key||''} ${x.source?'· '+x.source:''}</span></div>
    </div>`;
}

async function loadTextSearch(qs){
  switchSideTab('search');
  const h=document.getElementById('searchResults');
  h.innerHTML='<span class=muted>…</span>';
  const d=await(await fetch('/api/search_text?k=24&'+qs)).json();
  document.getElementById('searchInfo').textContent=`${(d.hits||[]).length} hits · ${(d.n||0).toLocaleString()} files indexed`;
  if(!d.hits || d.hits.length===0){h.innerHTML='<span class=muted>no match</span>';return;}
  h.classList.remove('muted');
  h.innerHTML=d.hits.map(searchRow).join('');
  _playingRow=null;

  // highlight all hits on the map, but don't hijack selection or audio
  selIdx.clear();
  if(M&&M.paths)d.hits.forEach(hit=>{const i=M.paths.indexOf(hit.path);if(i>=0)selIdx.add(i);});
  updateBatchPanel();
  draw();

  h.querySelectorAll('.srow').forEach(el=>{
    const path=decodeURIComponent(el.dataset.p);
    el.querySelector('.hit-play').onclick=(e)=>{e.stopPropagation();playFromRow(e.target,path);};
    el.onclick=()=>searchLocate(path);
  });
}

function runSearch(v){
  if(!v)return;
  if(v.includes('/')||v.includes('\\'))loadSimilar('q='+encodeURIComponent(v));
  else{document.getElementById('q2').value=v;loadTextSearch('q='+encodeURIComponent(v));}
}
document.getElementById('q').addEventListener('keydown',e=>{
  if(e.key==='Enter')runSearch(e.target.value.trim());});
document.getElementById('q2').addEventListener('keydown',e=>{
  if(e.key==='Enter')runSearch(e.target.value.trim());});
const VIEW_FIELDS=[['instrument','resolved'],['human','human'],['model','model'],
  ['path','path'],['panns','PANNs'],['audio','audio'],['family','sonic family']];
const PROV=[[0,'single'],[1,'cluster'],[2,'map'],[3,'propagate'],[4,'llm'],[5,'none']];
function legend(){
  const el=document.getElementById('legend');
  const cats=curCats(), colors=curColors(), act=curActC(), noneIdx=cats.length;
  let h='<div style="margin-bottom:4px"><b>view by</b> '+
    `<select id="viewBy" style="background:#1e1f1c;border:1px solid #3e3d32;color:var(--fg);border-radius:3px;font-family:inherit;font-size:11px">`+
    VIEW_FIELDS.map(([v,lbl])=>`<option value="${v}" ${viewBy===v?'selected':''}>${lbl}</option>`).join('')+
    `</select></div>`;
  const catTitle=viewBy==='family'?'sonic family':'instrument';
  h+=`<div style="margin-bottom:4px"><b>${catTitle}</b> `+
    '<a href=# id=lall>all</a> · <a href=# id=lnone>none</a></div>';
  if(viewBy==='family'&&!cats.length){
    h+='<div class=step-hint style="margin-bottom:4px">sonic families not computed yet</div>';
  }
  h+=cats.map((c,i)=>`<span class="lg${act.has(i)?'':' off'}" data-i="${i}">`+
    `<span class=dot style="background:${colors[i]}"></span>${c}</span>`).join('');
  h+=`<span class="lg${act.has(noneIdx)?'':' off'}" data-i="${noneIdx}">`+
    `<span class=dot style="background:#4a4a42"></span>— none —</span>`;
  h+='<div style="margin-top:5px"><b>type</b> '+
    `<label><input type=checkbox class=tt data-t=0 ${actT.has(0)?'checked':''}>oneshot</label>`+
    `<label><input type=checkbox class=tt data-t=1 ${actT.has(1)?'checked':''}>loop</label>`+
    `<label><input type=checkbox class=tt data-t=2 ${actT.has(2)?'checked':''}>other</label></div>`;
  h+='<div style="margin-top:5px"><b>label source</b> '+
    PROV.map(([s,lbl])=>`<label><input type=checkbox class=ss data-s=${s} ${actS.has(s)?'checked':''}>${lbl}</label>`).join('')+
    '</div>';
  h+=`<div style="margin-top:5px"><b>length</b> `+
    `<input class=dn id=dmin type=number min=0 step=0.1 placeholder=min value="${minD||''}" `+
    `style="width:54px;padding:1px 4px;background:#272822;border:1px solid #3e3d32;color:var(--fg);border-radius:3px;font-family:inherit;font-size:11px"> – `+
    `<input class=dn id=dmax type=number min=0 step=0.1 placeholder=max value="${maxD||''}" `+
    `style="width:54px;padding:1px 4px;background:#272822;border:1px solid #3e3d32;color:var(--fg);border-radius:3px;font-family:inherit;font-size:11px"> s</div>`;
  el.innerHTML=h;
  el.querySelectorAll('.lg').forEach(s=>s.onclick=()=>{const i=+s.dataset.i;
    act.has(i)?act.delete(i):act.add(i);legend();draw();});
  lall.onclick=e=>{e.preventDefault();cats.forEach((_,i)=>act.add(i));act.add(noneIdx);legend();draw();};
  lnone.onclick=e=>{e.preventDefault();act.clear();legend();draw();};
  el.querySelectorAll('.tt').forEach(cb=>cb.onchange=()=>{const t=+cb.dataset.t;
    cb.checked?actT.add(t):actT.delete(t);draw();});
  el.querySelectorAll('.ss').forEach(cb=>cb.onchange=()=>{const s=+cb.dataset.s;
    cb.checked?actS.add(s):actS.delete(s);draw();});
  el.querySelectorAll('.dn').forEach(inp=>inp.oninput=()=>{
    minD=parseFloat(document.getElementById('dmin').value)||0;
    maxD=parseFloat(document.getElementById('dmax').value)||0;
    draw();});
  document.getElementById('viewBy').onchange=e=>{viewBy=e.target.value;legend();draw();};}
async function loadMap(reset){
  M=await(await fetch('/api/map')).json();
  document.getElementById('count').textContent=
    M.n?M.n.toLocaleString()+' samples':'no projection yet — click Update map';
  if(actC===null){actC=new Set(M.cats.map((_,i)=>i));actC.add(M.cats.length);  // incl. "none" bucket
    actFam=new Set((M.famCats||[]).map((_,i)=>i));actFam.add((M.famCats||[]).length);
    actT=new Set([0,1,2]);actS=new Set([0,1,2,3,4,5]);}                          // all provenances
  legend();if(reset){resize();fit();}draw();}
let _toastTimer=null;
function showToast(msg,color){
  const t=document.getElementById('toast');
  if(!t)return;
  t.textContent=msg; t.style.color=color; t.style.borderColor=color;
  t.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer=setTimeout(()=>t.classList.remove('show'),1400);
}

const upd=document.getElementById('upd');
upd.onclick=async()=>{upd.disabled=true;upd.textContent='projecting…';
  await fetch('/api/reproject');pollReproj();};
async function pollReproj(){
  const s=await(await fetch('/api/reproject_status')).json();
  if(s.running){setTimeout(pollReproj,2000);return;}
  await loadMap(false);
  upd.disabled=false;upd.textContent='↻ Update map';
  if(s.ok===false) showToast('projection failed: '+s.msg, '#f92672');
}
// Keep selected point inside the canvas with a margin
function scrollToSel(){
  if(sel<0)return;
  const m=80*dpr,px=sx(sel),py=sy(sel);
  if(px<m)tx+=m-px;else if(px>cv.width-m)tx-=px-(cv.width-m);
  if(py<m)ty+=m-py;else if(py>cv.height-m)ty-=py-(cv.height-m);}
// Arrow-key navigation: jump to nearest visible point in that direction.
// Direction cone: only candidates with dot(normalised offset, dir) >= 0.3 (~73°).
// Score = dist / dot — penalises off-axis candidates so we always move "forward".
// y-axis: M.y=1 is top of screen (render uses 1-y), so ArrowUp means larger M.y.
let _inspectTimer=null;
document.addEventListener('keydown',e=>{
  if(!M||sel<0)return;
  if(e.key==='Enter'){
    const btn = document.getElementById('btnLabel');
    if(!btn.disabled) btn.click();
    return;
  }
  const dir={ArrowRight:[1,0],ArrowLeft:[-1,0],ArrowUp:[0,1],ArrowDown:[0,-1]}[e.key];
  if(!dir)return;
  e.preventDefault();
  const[dx,dy]=dir,cx=M.x[sel],cy=M.y[sel];
  let best=-1,bestScore=Infinity;
  for(let i=0;i<M.n;i++){
    if(i===sel||!shown(i))continue;
    if(stride>1&&i%stride!==0)continue;
    const vx=M.x[i]-cx,vy=M.y[i]-cy,dist=Math.sqrt(vx*vx+vy*vy);
    if(!dist)continue;
    const dot=(vx*dx+vy*dy)/dist;
    if(dot<0.3)continue;
    const score=dist/dot;
    if(score<bestScore){bestScore=score;best=i;}}
  if(best<0)return;
  sel=best;scrollToSel();draw();
  clearTimeout(_inspectTimer);
  _inspectTimer=setTimeout(()=>inspect(best),200);});
async function loadLabels(){
  const labels=await(await fetch('/api/labels')).json();
  const sel=document.getElementById('labelSel');
  const bsel=document.getElementById('batchSel');
  while(sel.options.length>1)sel.remove(1);
  bsel.innerHTML = '<option value="">— choose —</option>';
  labels.forEach(l=>{
    const o=document.createElement('option');o.value=l;o.textContent=l;sel.add(o);
    const bo=document.createElement('option');bo.value=l;bo.textContent=l;bsel.add(bo);
  });
}
loadMap(true);loadLabels();

document.getElementById('btnLegend').onclick = e => {
  const lg = document.getElementById('legend');
  const hidden = lg.style.display === 'none';
  lg.style.display = hidden ? '' : 'none';
  e.target.style.background = hidden ? '' : '#555';
};
document.getElementById('btnSelectMode').onclick = e => {
  selectMode = !selectMode;
  e.target.style.color = selectMode ? '#fff' : '';
  e.target.style.background = selectMode ? '#555' : '';
  cv.style.cursor = selectMode ? 'crosshair' : '';
};

document.getElementById('btnBatch').onclick = async () => {
  const instr = document.getElementById('batchSel').value;
  if(selIdx.size === 0 || !instr) return;
  const btn = document.getElementById('btnBatch');
  btn.disabled = true;
  const r = await fetch('/api/label_map', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      indices: Array.from(selIdx),
      instrument: instr,
      sidecar_mtime: M.sidecar_mtime,
      mode: document.getElementById('batchUnlabeledOnly').checked ? "unlabeled" : "all"
    })
  });
  const d = await r.json();
  if (d.ok) {
    showToast(`Labeled ${d.n} ✓`, '#a6e22e');
    const ci = M.cats.indexOf(instr);
    if(ci >= 0) { for(let i of selIdx) setLocalLabel(i, ci, 2); }  // provenance 'map'
    selIdx.clear();
    updateBatchPanel();
    draw();
  } else if (d.stale) {
    showToast("Map changed — reloading...", '#f92672');
    selIdx.clear();
    updateBatchPanel();
    loadMap(false);
  } else {
    showToast("Error: " + (d.msg || "failed"), '#f92672');
  }
  btn.disabled = false;
};