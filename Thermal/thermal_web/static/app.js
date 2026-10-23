'use strict';
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
let meta=null, tab='launcher', axis='moving', showHistory=true, states={}, versions={}, pendingConfidence=null;
let dragging=null, chartSamples=[], chartGeometry=null, seekDragging=false;
const flags={launcher:{circles:false,boxes:true,ids:true,raw:false},simulator:{circles:false,boxes:true,ids:true,raw:false}};
const series=[['estimate','Estimate','#eee',''],['observed','Observed','#888','5 4'],['detected','Detected','#bbb','2 4'],['raw_heat','rFₕ','#777','7 3'],['heat','Fₕ','#fff',''],['plausible_detected','pD','#ddd','10 3 2 3']];
const heatFields=[['low_c','Lower temperature (°C)',0,60,.5],['high_c','Upper temperature (°C)',0,80,.5],['near_weight','Near / bottom pixel weight',.1,20,.1],['far_weight','Far / top pixel weight',.1,40,.1],['power','Vertical gradient exponent',.1,5,.1],['area','Weighted pixels per occupancy unit',1,10000,1],['a','a · raw heat weight',0,.49,.01],['b','b · detection baseline weight',.51,1,.01],['warmup_s','Continuous Faster K seconds',0,120,1],['switch_window_s','Recent switch window seconds',1,240,1],['max_precise_entries','Maximum Precise K entries',0,20,1]];
const s=()=>states[tab]||{};
const fmt=(value,digits=0)=>Number.isFinite(value)?value.toFixed(digits):'—';
function error(message){$('error').textContent=message||'';$('error').hidden=!message;}
async function command(action,values={},workspace=tab){
  if(!meta)return;
  try{
    const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Workspace-Token':meta.token},body:JSON.stringify({workspace,command:{action,...values}})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Command failed');
    error('');
  }catch(e){error(e.message);pendingConfidence=null;}
}
function svgNode(name,attributes={},text){const node=document.createElementNS(NS,name);for(const [k,v]of Object.entries(attributes))if(v!==null)node.setAttribute(k,String(v));if(text!==undefined)node.textContent=text;return node;}
function selected(button,on){button.classList.toggle('selected',on);button.setAttribute('aria-pressed',String(on));}
function paintSlider(input){const min=Number(input.min),max=Number(input.max);input.style.setProperty('--fill',`${100*(Number(input.value)-min)/(max-min)}%`);}
function updateConfidence(changed){
  let conf=Number($('confidence').value),low=Number($('search-confidence').value);
  if(changed==='confidence')low=Math.min(low,conf-.01);else conf=Math.max(conf,low+.01);
  conf=Math.max(.02,Math.min(.99,conf));low=Math.max(.01,Math.min(conf-.01,low));
  $('confidence').value=conf;$('search-confidence').value=low;
  $('confidence-value').textContent=conf.toFixed(3);$('search-confidence-value').textContent=low.toFixed(3);
  paintSlider($('confidence'));paintSlider($('search-confidence'));
  pendingConfidence={conf,low_conf:low};
  const values={...pendingConfidence},workspace=tab;
  clearTimeout(updateConfidence.timer);updateConfidence.timer=setTimeout(()=>command('confidence',values,workspace),100);
}
for(const id of ['confidence','search-confidence']){
  $(id).addEventListener('input',()=>updateConfidence(id));
  $(id).addEventListener('pointerdown',()=>dragging=id);
  $(id).addEventListener('pointerup',()=>dragging=null);paintSlider($(id));
}
function drawChart(state){
  const row=state.row,active=Boolean(row?.heat_active&&row.mode==='faster');
  $('heat-metrics').hidden=!active;
  if(active)for(const item of $('heat-metrics').children)item.querySelector('b').textContent=fmt(row[item.dataset.field],item.dataset.field==='raw_heat'?2:0);
  const svg=$('chart');svg.replaceChildren();$('chart-tooltip').hidden=true;
  const samples=axis==='compressed'?(state.overview?.length?state.overview:state.history||[]):state.history||[];
  chartSamples=samples;
  const width=svg.clientWidth||900;svg.setAttribute('viewBox',`0 0 ${width} 130`);
  const left=33,right=width-6,top=8,bottom=104;
  const fields=active?series:series.slice(0,2);
  let maximum=2;for(const item of samples)for(const [key]of fields)maximum=Math.max(maximum,item[key]||0);
  const step=Math.max(1,Math.ceil(maximum/3));maximum=step*3;
  const t0=axis==='compressed'?0:(samples[0]?.t||0),t1=samples.at(-1)?.t||t0,span=t1-t0;
  const px=t=>span?left+(t-t0)/span*(right-left):right,py=n=>bottom-n/maximum*(bottom-top);
  chartGeometry={left,right,top,bottom,px,py};
  for(let i=0;i<4;i++){
    const y=py(step*i);svg.append(svgNode('line',{x1:left,y1:y,x2:right,y2:y,stroke:'#303030','stroke-dasharray':'2 5'}));
    svg.append(svgNode('text',{x:4,y:y+3},String(step*i)));
    const t=t0+span*i/3;svg.append(svgNode('text',{x:left+(right-left)*i/3,y:125,'text-anchor':i===3?'end':i===0?'start':'middle'},`${t.toFixed(span<10?1:0)}s`));
  }
  if(samples.length){
    const points=[];samples.forEach((item,i)=>{const x=px(item.t),y=py(item.estimate);if(i)points.push(`${x},${py(samples[i-1].estimate)}`);points.push(`${x},${y}`);});
    const defs=svgNode('defs'),gradient=svgNode('linearGradient',{id:'area-shade',x1:'0%',y1:'0%',x2:'0%',y2:'100%'});
    gradient.append(svgNode('stop',{offset:'0%','stop-color':'#ddd','stop-opacity':'.27'}),svgNode('stop',{offset:'100%','stop-color':'#ddd','stop-opacity':'.01'}));defs.append(gradient);svg.append(defs);
    svg.append(svgNode('polygon',{points:`${px(samples[0].t)},${bottom} ${points.join(' ')} ${px(samples.at(-1).t)},${bottom}`,fill:'url(#area-shade)'}));
    for(const [fieldIndex,[key,label,color,dash]]of fields.entries()){
      let segment=[];
      const flush=()=>{if(segment.length>1)svg.append(svgNode('polyline',{points:segment.join(' '),fill:'none',stroke:color,'stroke-width':key==='heat'||key==='plausible_detected'?1.8:1.1,'stroke-dasharray':dash,'data-series':key}));else if(segment.length){const [cx,cy]=segment[0].split(',');svg.append(svgNode('circle',{cx,cy,r:2,fill:color,'data-series':key}));}segment=[];};
      samples.forEach(item=>{const value=item[key];if(value==null||(fieldIndex>=2&&!item.heat_active)){flush();return;}const x=px(item.t);if(segment.length&&key!=='raw_heat')segment.push(`${x},${segment.at(-1).split(',')[1]}`);segment.push(`${x},${py(value)}`);});flush();
    }
  }else svg.append(svgNode('text',{x:50,y:52},'Waiting for processed frames'));
  $('history-note').textContent=(axis==='compressed'?'Compressed overview · decimated':'Moving window · last 120 frames')+(active?' · heat / pD diagnostics':' · Estimate: Precise legacy / Faster observed');
}
$('chart').addEventListener('pointermove',event=>{
  if(!chartSamples.length||!chartGeometry)return;
  const rect=$('chart').getBoundingClientRect(),x=event.clientX-rect.left;
  const item=chartSamples.reduce((best,item)=>Math.abs(chartGeometry.px(item.t)-x)<Math.abs(chartGeometry.px(best.t)-x)?item:best);
  let label=`${item.t.toFixed(2)}s · frame ${item.frame}\nEstimate ${item.estimate}   Observed ${item.observed}\n${item.mode==='faster'?'Faster':'Precise'} K`;
  if(s().row?.heat_active&&item.heat_active)label+=`\nDetected ${fmt(item.detected)} · rFₕ ${fmt(item.raw_heat,2)}\nFₕ ${fmt(item.heat)} · pD ${fmt(item.plausible_detected)}`;
  const tooltip=$('chart-tooltip');tooltip.textContent=label;tooltip.hidden=false;tooltip.style.left=`${Math.max(0,Math.min(rect.width-190,event.clientX-rect.left+12))}px`;
});
$('chart').addEventListener('pointerleave',()=>$('chart-tooltip').hidden=true);
function drawDetections(row){
  const svg=$('detections');svg.replaceChildren();if(!row)return;const display=flags[tab];
  const width=Math.max(160,svg.getBoundingClientRect().width),stroke=Math.max(.23,160/width),labelSize=Math.max(2.4,10.5*160/width);
  const box=(bbox,color,dash='')=>{const [x,y,w,h]=bbox;svg.append(svgNode('rect',{x,y,width:w,height:h,fill:'none',stroke:color,'stroke-width':stroke,'stroke-dasharray':dash}));};
  if(display.raw)for(const detection of row.base_detections)box(detection.bbox,'#aaa','1 .7');
  if(display.circles&&row.mode==='faster')for(const region of row.regions){svg.append(svgNode('circle',{cx:region.cx,cy:region.cy,r:region.radius,fill:'none',stroke:'#ccc','stroke-width':.23,'stroke-dasharray':'1 1'}));}
  for(const track of row.observed){const color=track.source==='base'?'#b4e58e':'#eeeeee';if(display.boxes)box(track.bbox,color);if(display.ids){const [x,y]=track.bbox;svg.append(svgNode('text',{x:Math.max(0,Math.min(140,x)),y:Math.max(labelSize,y-.9),fill:color,'font-size':labelSize,'font-family':'Arial'},`#${track.id} ${fmt(track.conf,2)}`));}}
}
function render(state){
  const row=state.row,loaded=Boolean(state.loaded);
  $('status').textContent=state.status||'Ready';$('status-dot').style.background=loaded?'#ddd':'#666';error(state.error);
  $('live-tools').hidden=tab!=='launcher';$('replay-tools').hidden=tab!=='simulator';$('timeline').hidden=tab!=='simulator'||!loaded;
  $('play').innerHTML=state.paused?'Play <kbd>SPACE</kbd>':'Pause <kbd>SPACE</kbd>';
  $('play').disabled=!loaded;$('step').disabled=!loaded;$('reset').disabled=!loaded;
  $('record').innerHTML=(state.recording?'Stop recording':'Start recording')+' <kbd>R</kbd>';$('saved').textContent=`${state.saved||0} saved`;
  $('record').disabled=!row||!loaded;$('shot').disabled=!row||!loaded;
  $('frame-id').textContent=row?`FRAME ${String(row.frame).padStart(6,'0')}`:'FRAME —';
  const mode=row?.mode==='faster'?'Faster K':row?.mode==='precise'?'Precise K':'Not connected';$('tracker-badge').textContent=mode;$('frame-mode').textContent=mode;
  $('history-status').textContent=row?`Observed ${row.observed.length}  ·  Unresolved ${row.unresolved}  ·  ${row.processing_ms.toFixed(0)} ms`:'Open a capture or connect Lepton to begin.';
  if(pendingConfidence&&Math.abs(state.conf-pendingConfidence.conf)<.00001&&Math.abs(state.low_conf-pendingConfidence.low_conf)<.00001)pendingConfidence=null;
  if(!dragging&&!pendingConfidence){$('confidence').value=state.conf??.374;$('search-confidence').value=state.low_conf??.18;for(const id of ['confidence','search-confidence'])paintSlider($(id));$('confidence-value').textContent=Number($('confidence').value).toFixed(3);$('search-confidence-value').textContent=Number($('search-confidence').value).toFixed(3);}
  for(const button of document.querySelectorAll('[data-policy]'))selected(button,button.dataset.policy===state.policy);
  for(const button of document.querySelectorAll('[data-toggle]'))selected(button,flags[tab][button.dataset.toggle]);
  selected($('heat-band'),Boolean(state.heat_band));$('heat-band').textContent=`${state.heat_config?.low_c??28}–${state.heat_config?.high_c??37}°C band`;
  const names=['auto p1–p99','crowd p40–p99.5','dense p70–p100'];$('thermal-caption').textContent='160 × 120 / '+(state.heat_band?'temperature band':state.palette==='inferno'?'Inferno · '+names[state.contrast||0]:state.palette==='viridis'?'Viridis · fixed 15–45°C':'Grayscale · fixed 15–45°C');
  if(state.image){$('image-wrap').hidden=false;$('empty').hidden=true;const url='data:image/png;base64,'+state.image;if($('thermal-image').src!==url)$('thermal-image').src=url;}else{$('image-wrap').hidden=true;$('empty').hidden=false;}
  $('empty-title').textContent=tab==='launcher'?'Your thermal workspace, in the browser.':'Replay your captures. Inspect every frame.';
  $('empty-description').textContent=tab==='launcher'?'Connect a Lepton camera to this computer. Use Simulator to replay a saved capture.':'Open a capture folder to inspect detections, tracking and heat metrics with recorded timing.';
  $('base-ms').textContent=row?fmt(row.base_ms,1)+' ms':'—';$('recovery-ms').textContent=row?fmt(row.recovery_ms,1)+' ms':'—';$('tracking-ms').textContent=row?fmt(row.tracking_ms,1)+' ms':'—';
  $('mean-score').textContent=row?.observed.length?fmt(row.observed.reduce((a,t)=>a+(t.conf||0),0)/row.observed.length,3):'—';$('recovered').textContent=row?row.observed.filter(t=>t.source!=='base').length:'—';
  $('temp-range').textContent=row?.temperature?`${fmt(row.temperature.min,1)}–${fmt(row.temperature.max,1)}°C`:'—';
  $('speed').value=state.speed||1;$('frame-progress').textContent=`${(state.index??-1)+1} / ${state.total||0}`;
  $('seek').max=Math.max(0,(state.total||1)-1);if(!seekDragging)$('seek').value=Math.max(0,state.index??0);paintSlider($('seek'));
  drawChart(state);drawDetections(row);
}
async function poll(){
  const current=tab;
  try{const response=await fetch(`/api/state?workspace=${current}&since=${versions[current]??-1}`);if(response.status===204)return;if(!response.ok)throw Error('Local backend unavailable');const state=await response.json();states[current]=state;versions[current]=state.version;if(current===tab)render(state);}
  catch(e){$('status').textContent='Local backend disconnected. Restart web_workspace.py to reconnect.';$('status-dot').style.background='#555';}
  finally{setTimeout(poll,120);}
}
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{
  if(button.dataset.tab===tab)return;command('pause',{value:true},tab);tab=button.dataset.tab;pendingConfidence=null;versions[tab]=-1;
  for(const item of document.querySelectorAll('[data-tab]')){selected(item,item.dataset.tab===tab);item.setAttribute('aria-selected',String(item.dataset.tab===tab));}render(s());
}));
document.querySelectorAll('[data-policy]').forEach(button=>button.addEventListener('click',()=>command('mode',{value:button.dataset.policy})));
document.querySelectorAll('[data-toggle]').forEach(button=>button.addEventListener('click',()=>{flags[tab][button.dataset.toggle]=!flags[tab][button.dataset.toggle];render(s());}));
const open=id=>$(id).showModal();
document.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>button.closest('dialog').close()));
$('connect').onclick=()=>open('connect-dialog');$('open-capture').onclick=async()=>{await refreshCaptures();open('source-dialog');};
$('help').onclick=()=>open('help-dialog');$('options').onclick=()=>{for(const input of $('options-form').querySelectorAll('input[name]'))input.value=s().tracker_config?.[input.name]??input.value;open('options-dialog');};
$('capture-select').onchange=()=>{$('capture-path').value=$('capture-select').value;};
$('source-form').onsubmit=event=>{event.preventDefault();command('load',{path:$('capture-path').value.trim(),fps:$('fps').value||null});$('source-dialog').close();};
$('connect-form').onsubmit=event=>{event.preventDefault();command('connect',{camera_index:Number($('camera-index').value),scale:Number($('camera-scale').value),device:$('device').value});$('connect-dialog').close();};
$('options-form').onsubmit=event=>{event.preventDefault();const values=Object.fromEntries([...new FormData(event.target)].map(([k,v])=>[k,Number(v)]));command('tracker_options',{values});$('options-dialog').close();};
$('play').onclick=()=>command('pause',{value:!s().paused});$('step').onclick=()=>command('step');$('reset').onclick=()=>command('reset');$('back').onclick=()=>command('seek',{index:Math.max(0,(s().index||0)-1)});
$('record').onclick=()=>command('record');$('shot').onclick=()=>command('shot');$('disconnect').onclick=()=>command('disconnect');$('speed').onchange=()=>command('speed',{value:Number($('speed').value)});
$('seek').onpointerdown=()=>seekDragging=true;$('seek').oninput=()=>{paintSlider($('seek'));$('frame-progress').textContent=`${Number($('seek').value)+1} / ${s().total||0}`;};$('seek').onchange=()=>{seekDragging=false;command('seek',{index:Number($('seek').value)});};
$('export').onclick=()=>{if(!s().report){error('Process a frame before exporting.');return;}const a=document.createElement('a');a.href=`/api/export?workspace=${tab}`;a.download='occupancy-session.json';a.click();};
$('history-toggle').onclick=()=>{showHistory=!showHistory;$('history-card').hidden=!showHistory;selected($('history-toggle'),showHistory);};
$('axis-toggle').onclick=()=>{axis=axis==='moving'?'compressed':'moving';$('axis-toggle').textContent='Time: '+axis;drawChart(s());};
$('palette').onclick=()=>{const items=['inferno','viridis','gray'];command('display',{palette:items[(items.indexOf(s().palette||'inferno')+1)%3]});};
$('contrast').onclick=()=>command('display',{contrast:((s().contrast||0)+1)%3,palette:'inferno'});$('heat-band').onclick=()=>command('display',{heat_band:!s().heat_band});
function heatValues(values){for(const [key,label,min,max,step]of heatFields)$('heat-'+key).value=values[key];}
$('heat-settings').onclick=()=>{heatValues(s().heat_config||meta.heat_defaults);open('heat-dialog');};
$('heat-defaults').onclick=()=>heatValues(meta.heat_defaults);
$('heat-form').onsubmit=event=>{event.preventDefault();const values={};for(const [key]of heatFields)values[key]=Number($('heat-'+key).value);command('heat',{values});$('heat-dialog').close();};
document.addEventListener('keydown',event=>{
  if(document.querySelector('dialog[open]')||['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)||event.ctrlKey||event.metaKey||event.altKey)return;
  const key=event.key.toLowerCase();const buttons={' ':'play',n:'step',x:'reset',r:'record',s:'shot',q:'disconnect',h:'help',o:'history-toggle',p:'palette',v:'contrast'};
  if(buttons[key]){event.preventDefault();$(buttons[key]).click();return;}
  if(['1','2','3'].includes(key)){event.preventDefault();command('mode',{value:{1:'auto',2:'precise',3:'faster'}[key]});return;}
  const toggles={c:'circles',b:'boxes',i:'ids',g:'raw'};if(toggles[key]){document.querySelector(`[data-toggle="${toggles[key]}"]`).click();return;}
  if(['[',']',';',"'"].includes(key)){event.preventDefault();const id=['[',']'].includes(key)?'confidence':'search-confidence';$(id).value=Number($(id).value)+([']',"'"].includes(key)?.01:-.01);updateConfidence(id);}
});
async function refreshCaptures(){const response=await fetch('/api/meta');if(!response.ok)throw Error('Cannot read saved captures');meta=await response.json();const select=$('capture-select');select.replaceChildren(new Option('Choose a capture…',''));for(const capture of meta.captures)select.add(new Option(`${capture.name} · ${capture.frames} frames`,capture.path));}
async function init(){
  try{await refreshCaptures();$('workspace-path').textContent=meta.path;$('workspace-path').title=meta.path;$('camera-help').textContent=meta.platform==='Darwin'?'Mac uses libuvc. If USB access is denied, restart this backend with sudo in Terminal.':'Windows uses 16-bit Y16 via DirectShow / Media Foundation. Install the camera driver and thermal model on this PC.';
    for(const [key,label,min,max,step]of heatFields){const l=document.createElement('label');l.htmlFor='heat-'+key;l.textContent=label;const input=document.createElement('input');Object.assign(input,{id:'heat-'+key,type:'number',min,max,step,required:true});$('heat-fields').append(l,input);}
    $('heat-a').oninput=()=>$('heat-b').value=(1-Number($('heat-a').value)).toFixed(2);$('heat-b').oninput=()=>$('heat-a').value=(1-Number($('heat-b').value)).toFixed(2);
    render({});poll();
  }catch(e){error(e.message);$('status').textContent='Could not connect to the local backend.';}
}
init();
new ResizeObserver(()=>{drawChart(s());drawDetections(s().row);}).observe($('chart'));
