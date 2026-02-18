// ==UserScript==
// @name         Watchmode Local Mapping - packed arrays + sample (fast, LE-optimized)
// @namespace    http://tampermonkey.net/
// @version      2.12
// @description  Fast in-RAM lookups for watchmode<->tmdb. Uses direct Uint32 views when keys/values are little-endian (no copy). Minimal hot-path overhead.
// @match        https://www.themoviedb.org/movie/*
// @match        https://www.themoviedb.org/tv/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_xmlhttpRequest
// @grant        GM_setClipboard
// @connect      hcgiub001.github.io
// @connect      raw.githubusercontent.com
// @connect      api.watchmode.com
// @connect      watchmode.com
// @run-at       document-idle
// ==/UserScript==

(function () {
  'use strict';

  /* ---------- CONFIG ---------- */
  const WM_VERSION = '2.12-le-optimized';
  const BASE = "https://hcgiub001.github.io/FLAMWM"; // update to your base if needed
  const STORAGE_PREFIX = "wm_local_";
  const REQUEST_TIMEOUT_MS = 30000;

  // filenames (must match builder outputs)
  const TMDB2WM_MOVIE = {
    keys:  "tmdb2wm.movie.tmdb2wm.movie.keys.bin",
    values:"tmdb2wm.movie.watchmode.movie.values.bin",
    sample:"tmdb2wm.movie.sample.u32",
    meta:  "tmdb2wm.movie.meta.json"
  };
  const TMDB2WM_TV = {
    keys:  "tmdb2wm.tv.tmdb2wm.tv.keys.bin",
    values:"tmdb2wm.tv.watchmode.tv.values.bin",
    sample:"tmdb2wm.tv.sample.u32",
    meta:  "tmdb2wm.tv.meta.json"
  };
  const WM2TMDB = {
    keys:  "wm2tmdb.wm2tmdb.keys.bin",
    values:"wm2tmdb.packed_tmdb.values.bin",
    sample:"wm2tmdb.sample.u32",
    meta:  "wm2tmdb.meta.json"
  };

  const ALL_FILES = [
    TMDB2WM_MOVIE.meta, TMDB2WM_MOVIE.sample, TMDB2WM_MOVIE.keys, TMDB2WM_MOVIE.values,
    TMDB2WM_TV.meta, TMDB2WM_TV.sample, TMDB2WM_TV.keys, TMDB2WM_TV.values,
    WM2TMDB.meta, WM2TMDB.sample, WM2TMDB.keys, WM2TMDB.values
  ];

  /* ---------- UI (kept simple) ---------- */
  function now(){ return performance.now(); }
  function escapeHtml(s){ if (s===null||s===undefined) return ''; return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  const panel = document.createElement('div');
  panel.setAttribute('wm-version', WM_VERSION);
  Object.assign(panel.style, {
    position: 'fixed', top: '20px', right: '20px', width: '560px',
    background: 'white', border: '1px solid #888', padding: '10px', zIndex: 2147483647,
    fontFamily: 'monospace', fontSize: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)', borderRadius: '8px'
  });
  panel.innerHTML = `
    <div style="display:flex;align-items:center;">
      <b>Watchmode Local Mapping (fast)</b>
      <div style="margin-left:8px;color:#666;font-size:11px">LE-optimized typed views</div>
      <div style="margin-left:8px;color:#999;font-size:11px">ver ${WM_VERSION}</div>
      <button id="wm-close" style="margin-left:auto;">✕</button>
    </div>

    <div style="margin-top:8px;">
      Watchmode ID:<br>
      <input id="wm-id" style="width:100%;" placeholder="Enter Watchmode ID (optional)">
    </div>

    <div style="margin-top:6px;display:flex;gap:8px;align-items:center;">
      <button id="wm-run">Run</button>
      <button id="wm-settings">Settings</button>
      <span id="wm-meta-status" style="margin-left:8px;color:#888;font-size:12px"></span>
      <span id="wm-status" style="margin-left:auto;color:#666"></span>
    </div>

    <div id="wm-log" style="margin-top:8px; height:360px; overflow:auto; border-top:1px solid #ccc; padding-top:8px;"></div>
  `;
  document.body.appendChild(panel);
  document.getElementById('wm-close').onclick = () => panel.remove();
  const logDiv = document.getElementById('wm-log');
  const statusEl = document.getElementById('wm-status');
  const metaStatusSpan = document.getElementById('wm-meta-status');
  function logMsg(msg, strong=false){
    const el = document.createElement('div');
    el.style.marginBottom = '4px';
    el.innerHTML = strong ? `<b>${escapeHtml(msg)}</b>` : escapeHtml(msg);
    logDiv.appendChild(el);
    logDiv.scrollTop = logDiv.scrollHeight;
  }
  function logHtml(html){
    const el = document.createElement('div');
    el.style.marginBottom = '4px';
    el.innerHTML = html;
    logDiv.appendChild(el);
    logDiv.scrollTop = logDiv.scrollHeight;
  }

  /* ---------- network helpers ---------- */
  function gmArrayBufferWithTiming(url, opts={}) {
    return new Promise((resolve,reject)=>{
      const t0 = now();
      let timedOut=false;
      try {
        GM_xmlhttpRequest({
          method: opts.method || 'GET',
          url,
          responseType: 'arraybuffer',
          headers: opts.headers || {},
          timeout: REQUEST_TIMEOUT_MS,
          ontimeout: ()=>{ timedOut=true; reject(new Error('Request timed out')); },
          onerror: ()=>{ if(!timedOut) reject(new Error('Network error')); },
          onload: (res)=>{
            const t1 = now();
            try { resolve({ status: res.status, buffer: res.response, fetchTimeMs: t1 - t0, finalUrl: res.finalUrl || url }); }
            catch(e){ reject(e); }
          }
        });
      } catch(e){ reject(e); }
    });
  }
  function gmJsonWithTiming(url) {
    return new Promise((resolve,reject)=>{
      const t0 = now();
      let timedOut=false;
      try {
        GM_xmlhttpRequest({
          method: 'GET', url, headers:{}, timeout: REQUEST_TIMEOUT_MS,
          ontimeout: ()=>{ timedOut=true; reject(new Error('Request timed out')); },
          onerror: ()=>{ if(!timedOut) reject(new Error('Network error')); },
          onload: (res)=>{
            const t1 = now();
            try { const data = JSON.parse(res.responseText); const t2 = now(); resolve({ status: res.status, data, fetchTimeMs: t1 - t0, parseTimeMs: t2 - t1, finalUrl: res.finalUrl || url }); }
            catch(e){ reject(e); }
          }
        });
      } catch(e){ reject(e); }
    });
  }

  /* ---------- base64 storage helpers ---------- */
  function arrayBufferToBase64(ab){
    const bytes = new Uint8Array(ab);
    const chunk = 0x8000;
    let binary = '';
    for(let i=0;i<bytes.length;i+=chunk) binary += String.fromCharCode.apply(null, bytes.subarray(i, i+chunk));
    return btoa(binary);
  }
  function base64ToArrayBuffer(b64){
    const bin = atob(b64), len = bin.length, u8 = new Uint8Array(len);
    for(let i=0;i<len;i++) u8[i] = bin.charCodeAt(i);
    return u8.buffer;
  }
  async function saveLocalFile(name, arrayBuffer){
    try { await GM_setValue(STORAGE_PREFIX + name, arrayBufferToBase64(arrayBuffer)); }
    catch(e){ logMsg(`saveLocalFile error: ${e && e.message ? e.message : e}`, true); }
  }
  async function loadLocalFileArrayBuffer(name){
    const b64 = await GM_getValue(STORAGE_PREFIX + name, null);
    if(!b64) return null;
    return base64ToArrayBuffer(b64);
  }

  /* ---------- auto-download files once ---------- */
  async function ensureAllLocalFiles(){
    for(const fname of ALL_FILES){
      try {
        const key = STORAGE_PREFIX + fname;
        const exists = await GM_getValue(key, null);
        if (!exists) {
          logMsg(`Downloading ${fname}...`);
          try {
            const resp = await gmArrayBufferWithTiming(`${BASE}/${fname}`);
            if(!(resp && resp.buffer)) throw new Error('no buffer');
            await saveLocalFile(fname, resp.buffer);
            logMsg(`${fname} saved locally`);
          } catch(e){
            logMsg(`Failed to download ${fname}: ${e && e.message ? e.message : e}`, true);
          }
        }
      } catch(e){ console.warn('ensureAllLocalFiles error', e); }
    }
  }

  /* ---------- in-memory packed structures (numeric views) ---------- */
  let packedTmdb2wmMovie = null;
  let packedTmdb2wmTv = null;
  let packedWm2tmdb = null;
  let bench = { numBinarySearches:0, numKeyReads:0, numValueReads:0, numSamplesUsed:0, loadMs:0 };

  /* ---------- load mapping: create fast numeric views (direct view if LE) ---------- */
  async function loadPackedMapping(spec){
    const t0 = now();
    const keysAb = await loadLocalFileArrayBuffer(spec.keys);
    const valsAb = await loadLocalFileArrayBuffer(spec.values);
    const sampleAb = await loadLocalFileArrayBuffer(spec.sample);
    const metaAb = await loadLocalFileArrayBuffer(spec.meta);
    if(!keysAb || !valsAb || !sampleAb || !metaAb){ logMsg(`Missing files for ${spec.meta}`, true); return null; }

    const metaTxt = new TextDecoder().decode(metaAb);
    let meta = null;
    try { meta = JSON.parse(metaTxt); } catch(e){ meta = null; }

    // sample is little-endian u32
    const sampleU8 = new Uint8Array(sampleAb);
    const sampleCount = Math.floor(sampleU8.byteLength / 4);
    const sample32 = new Uint32Array(sampleCount);
    const sampleDV = new DataView(sampleU8.buffer, sampleU8.byteOffset, sampleU8.byteLength);
    for(let i=0;i<sampleCount;i++) sample32[i] = sampleDV.getUint32(i*4, true) >>> 0;

    // keys/values raw bytes
    const keysU8 = new Uint8Array(keysAb);
    const valsU8 = new Uint8Array(valsAb);
    const keyBytes = meta && meta.key_bytes ? Number(meta.key_bytes) : 4;
    const valBytes = meta && meta.value_bytes ? Number(meta.value_bytes) : 4;
    const count = Math.floor(keysU8.length / keyBytes);
    const sampleStride = meta && meta.sample_stride ? Number(meta.sample_stride) : 64;

    // Build keys32/vals32 with zero-copy when files are little-endian and aligned.
    let keys32 = null;
    if(keyBytes === 4){
      const keyIsBig = meta && meta.key_endian && meta.key_endian.toLowerCase() === 'big';
      // prefer zero-copy view if file is little-endian (our writer now writes LE) and aligned
      if(!keyIsBig && (keysU8.byteOffset % 4 === 0)){
        try {
          keys32 = new Uint32Array(keysU8.buffer, keysU8.byteOffset, count);
        } catch(e){
          // fallback to safe conversion if view fails
          keys32 = new Uint32Array(count);
          const dvKeys = new DataView(keysU8.buffer, keysU8.byteOffset, keysU8.byteLength);
          for(let i=0;i<count;i++) keys32[i] = dvKeys.getUint32(i*4, !keyIsBig) >>> 0;
        }
      } else {
        // convert (only if file is big-endian or misaligned)
        keys32 = new Uint32Array(count);
        const dvKeys = new DataView(keysU8.buffer, keysU8.byteOffset, keysU8.byteLength);
        for(let i=0;i<count;i++) keys32[i] = dvKeys.getUint32(i*4, !keyIsBig) >>> 0;
      }
    }

    let vals32 = null;
    if(valBytes === 4){
      const valIsBig = meta && meta.value_endian && meta.value_endian.toLowerCase() === 'big';
      if(!valIsBig && (valsU8.byteOffset % 4 === 0)){
        try {
          vals32 = new Uint32Array(valsU8.buffer, valsU8.byteOffset, Math.floor(valsU8.length/4));
        } catch(e){
          vals32 = new Uint32Array(Math.floor(valsU8.length/4));
          const dvVals = new DataView(valsU8.buffer, valsU8.byteOffset, valsU8.byteLength);
          for(let i=0;i<vals32.length;i++) vals32[i] = dvVals.getUint32(i*4, !valIsBig) >>> 0;
        }
      } else {
        vals32 = new Uint32Array(Math.floor(valsU8.length/4));
        const dvVals = new DataView(valsU8.buffer, valsU8.byteOffset, valsU8.byteLength);
        for(let i=0;i<vals32.length;i++) vals32[i] = dvVals.getUint32(i*4, !valIsBig) >>> 0;
      }
    }

    // Build a direct-map (Direct Address Table) for fast block bounds by high 16 bits.
    let directMap = null;
    if (keys32 && keys32.length > 0) {
      const COUNT = keys32.length >>> 0;
      directMap = new Uint32Array(65537);
      // initialize all entries to COUNT (acts as sentinel / default)
      directMap.fill(COUNT);
      // set first occurrence (min index) per high-bits bucket
      for (let i = 0; i < COUNT; i++) {
        const hb = keys32[i] >>> 16;
        if (i < directMap[hb]) directMap[hb] = i;
      }
      // propagate minima backwards so directMap[hb] = first index with highBits >= hb
      for (let hb = 65535; hb >= 0; hb--) {
        if (directMap[hb] > directMap[hb + 1]) directMap[hb] = directMap[hb + 1];
      }
      // ensure tail sentinel
      directMap[65536] = COUNT;
    }

    const index = { keysU8, valsU8, sample32, meta, keyBytes, valBytes, count, sampleStride, keys32, vals32, directMap,
                    keysSize: keysU8.length, valsSize: valsU8.length, sampleSize: sampleU8.length };
    // record last load time (not cumulative)
    bench.loadMs = (now() - t0);
    return index;
  }

  /* ---------- fast search primitives (numeric views used in hot path) ---------- */
  function upperBoundSample(sample32, x){
    let lo = 0, hi = sample32.length;
    while(lo < hi){
      const mid = (lo + hi) >>> 1;
      if(sample32[mid] > x) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  }

  function binarySearchInBlock(keys32, key, left, right){
    let lo = left, hi = right;
    while(lo <= hi){
      const mid = (lo + hi) >>> 1;
      const k = keys32[mid];
      if(k === key) return mid;
      if(k < key) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  // Hot synchronous lookup: numeric-only fast path (no awaits, minimal overhead)
  function lookupPackedSync(key, idxObj){
    bench.numBinarySearches++;
    const keys32 = idxObj.keys32;
    const vals32 = idxObj.vals32;
    const sample32 = idxObj.sample32;
    const sampleStride = idxObj.sampleStride || idxObj.meta && idxObj.meta.sample_stride ? Number(idxObj.meta.sample_stride) : 64;
    const count = idxObj.count;

    if(!keys32 || count === 0) return -1;

    // Prefer Direct Map (fast): compute high 16 bits to get start/end
    if (idxObj.directMap && idxObj.directMap.length === 65537) {
      const dm = idxObj.directMap;
      const hb = key >>> 16;
      const left = dm[hb] >>> 0;
      const rightExclusive = dm[hb + 1] >>> 0;
      if (left >= rightExclusive) return -1; // empty bucket
      const right = rightExclusive - 1;
      const pos = binarySearchInBlock(keys32, key, left, right);
      if (pos >= 0) {
        bench.numValueReads++;
        return pos;
      }
      return -1;
    }

    // Fallback: sample-based block narrowing (original)
    const sb = upperBoundSample(sample32, key) - 1;
    let left = (sb >= 0) ? sb * sampleStride : 0;
    if(sb >= 0) bench.numSamplesUsed++;
    const right = Math.min(count - 1, Math.max(left, ((sb + 1) * sampleStride) - 1));

    // binary search within block using keys32 (fast)
    const pos = binarySearchInBlock(keys32, key, left, right);
    if(pos >= 0){
      bench.numValueReads++;
      return pos;
    }
    return -1;
  }

  /* ---------- public lookup wrappers ---------- */
  function lookupTmdbToWatchmodeSync(kind, tmdbId){
    const key = ((Number(tmdbId) << 1) | (kind === 'tv' ? 1 : 0)) >>> 0;
    const idx = (kind === 'movie') ? packedTmdb2wmMovie : packedTmdb2wmTv;
    if(!idx) return null;
    const pos = lookupPackedSync(key, idx);
    if(pos < 0) return null;
    // read value (fast if vals32 present)
    return idx.vals32 ? idx.vals32[pos] >>> 0 : (function(){ // fallback generic
      let off = pos * idx.valBytes, v = 0, mul = 1;
      const u8 = idx.valsU8;
      for(let i=0;i<idx.valBytes;i++){ v += (u8[off + i] & 0xFF) * mul; mul *= 256; }
      return v >>> 0;
    })();
  }

  function lookupWatchmodeToTmdbSync(wmid){
    const key = (Number(wmid) >>> 0);
    if(!packedWm2tmdb) return null;
    const pos = lookupPackedSync(key, packedWm2tmdb);
    if(pos < 0) return null;
    const packed = packedWm2tmdb.vals32 ? packedWm2tmdb.vals32[pos] >>> 0 : (function(){
      let off = pos * packedWm2tmdb.valBytes, v = 0, mul = 1;
      const u8 = packedWm2tmdb.valsU8;
      for(let i=0;i<packedWm2tmdb.valBytes;i++){ v += (u[off + i]&0xFF) * mul; mul *= 256; } // fallback will be fixed below
    })();
    // The above fallback has a small variable name issue in the original block; use robust fallback:
    let packedVal;
    if (packedWm2tmdb.vals32) packedVal = packedWm2tmdb.vals32[pos] >>> 0;
    else {
      let off = pos * packedWm2tmdb.valBytes, v = 0, mul = 1;
      const u8 = packedWm2tmdb.valsU8;
      for(let i=0;i<packedWm2tmdb.valBytes;i++){ v += (u8[off + i] & 0xFF) * mul; mul *= 256; }
      packedVal = v >>> 0;
    }
    return { tmdb_id: (packedVal >>> 1), tmdb_type: ((packedVal & 1) ? 'tv' : 'movie') };
  }

  /* ---------- ultra-fast batch resolver (25 ids) ---------- */
  function resolveWatchmodeIdsPackedBatchSync(watchmodeIds){
    const tStart = now();
    bench.numBinarySearches = 0; bench.numKeyReads = 0; bench.numValueReads = 0; bench.numSamplesUsed = 0;

    if(!Array.isArray(watchmodeIds) || watchmodeIds.length === 0) return [];

    const orig = watchmodeIds.map(x => Number(x));
    // dedupe preserving first occurrence
    const seen = Object.create(null);
    const unique = [];
    for(const v of orig){ if(!Number.isFinite(v)) continue; if(!(v in seen)){ seen[v] = true; unique.push(v); } }
    if(unique.length === 0) return [];

    unique.sort((a,b)=>a-b);

    if(!packedWm2tmdb) return orig.map(n=>({watchmode_id:n, tmdb_id:null, tmdb_type:null, found_local:false, source:'missing'}));

    // Heuristic: determine whether to use single-pass merge
    const sample = packedWm2tmdb.sample32;
    const sampleStride = packedWm2tmdb.sampleStride || packedWm2tmdb.meta && packedWm2tmdb.meta.sample_stride ? Number(packedWm2tmdb.meta.sample_stride) : 64;
    const firstBlock = Math.max(0, upperBoundSample(sample, unique[0]) - 1);
    const lastBlock = Math.max(0, upperBoundSample(sample, unique[unique.length - 1]) - 1);
    const blocksSpanned = Math.max(1, lastBlock - firstBlock + 1);
    const estScannedEntries = blocksSpanned * sampleStride;
    const mergeThreshold = Math.max(256, unique.length * sampleStride * 3); // tuned multiplier

    const results = Object.create(null);

    if(packedWm2tmdb.keys32 && estScannedEntries <= mergeThreshold){
      // clustered -> merge scan (fast when clustered)
      const keys = packedWm2tmdb.keys32;
      const vals = packedWm2tmdb.vals32;
      let qi = 0, ki = 0, Nq = unique.length, Nk = keys.length;
      while(qi < Nq && ki < Nk){
        const q = unique[qi];
        const k = keys[ki];
        if(k < q){ ki++; continue; }
        if(k === q){
          const v = vals ? vals[ki] : (function(){ let off = ki * packedWm2tmdb.valBytes, rv=0, mul=1, u=packedWm2tmdb.valsU8; for(let i=0;i<packedWm2tmdb.valBytes;i++){ rv += (u[off+i]&0xFF)*mul; mul*=256 } return rv>>>0; })();
          results[q] = { tmdb_id: (v >>> 1), tmdb_type: ((v & 1) ? 'tv' : 'movie'), found_local:true, source:'local' };
          qi++; ki++;
        } else {
          results[q] = { tmdb_id: null, tmdb_type:null, found_local:false, source:'not_found' };
          qi++;
        }
      }
      while(qi < Nq){ results[unique[qi++]] = { tmdb_id: null, tmdb_type:null, found_local:false, source:'not_found' }; }
    } else {
      // sparse -> do per-id synchronous lookups (fast for sparse queries)
      for(const id of unique){
        const r = lookupWatchmodeToTmdbSync(id);
        if(r) results[id] = { tmdb_id: r.tmdb_id, tmdb_type: r.tmdb_type, found_local:true, source:'local' };
        else results[id] = { tmdb_id: null, tmdb_type:null, found_local:false, source:'not_found' };
      }
    }

    const ordered = orig.map(n => {
      const r = results[n] || { tmdb_id:null, tmdb_type:null, found_local:false, source:'missing' };
      return { watchmode_id: n, tmdb_id: r.tmdb_id, tmdb_type: r.tmdb_type, found_local: r.found_local, source: r.source };
    });

    const tEnd = now();
    logMsg(`Batch resolution (fast) finished in ${(tEnd - tStart).toFixed(4)} ms`, true);
    logMsg(`Benchmark (packed): binarySearches=${bench.numBinarySearches}, valueReads=${bench.numValueReads}, samplesUsed=${bench.numSamplesUsed}, loadMs=${bench.loadMs.toFixed(2)}ms`, false);
    return ordered;
  }

  /* ---------- wrapper async loaders ---------- */
  async function loadAllPackedMappings(){
    const t0 = now();
    try {
      try { packedTmdb2wmMovie = await loadPackedMapping(TMDB2WM_MOVIE); } catch(e){ packedTmdb2wmMovie = null; console.warn(e); }
      try { packedTmdb2wmTv = await loadPackedMapping(TMDB2WM_TV); } catch(e){ packedTmdb2wmTv = null; console.warn(e); }
      try { packedWm2tmdb = await loadPackedMapping(WM2TMDB); } catch(e){ packedWm2tmdb = null; console.warn(e); }
      // bench.loadMs already set per mapping load; set it to the total load duration here
      bench.loadMs = (now() - t0);
      const parts = [];
      if(packedTmdb2wmMovie) parts.push(`tmdb2wm.movie(k=${humanBytes(packedTmdb2wmMovie.keysSize)},v=${humanBytes(packedTmdb2wmMovie.valsSize)},s=${humanBytes(packedTmdb2wmMovie.sampleSize)})`);
      if(packedTmdb2wmTv) parts.push(`tmdb2wm.tv(k=${humanBytes(packedTmdb2wmTv.keysSize)},v=${humanBytes(packedTmdb2wmTv.valsSize)},s=${humanBytes(packedTmdb2wmTv.sampleSize)})`);
      if(packedWm2tmdb) parts.push(`wm2tmdb(k=${humanBytes(packedWm2tmdb.keysSize)},v=${humanBytes(packedWm2tmdb.valsSize)},s=${humanBytes(packedWm2tmdb.sampleSize)})`);
      logMsg(`Loaded mappings into RAM: ${parts.join(' ; ')}`, false);
      return true;
    } catch(e){
      logMsg('loadAllPackedMappings error: ' + ((e&&e.message)? e.message : String(e)), true);
      return false;
    }
  }

  /* ---------- helpers & UI glue ---------- */
  function humanBytes(n){ if(n === null || n === undefined) return '-'; if(n < 1024) return `${n} B`; if(n < 1024*1024) return `${(n/1024).toFixed(1)} KB`; return `${(n/(1024*1024)).toFixed(2)} MB`; }
  async function openSettings(){ const current = await GM_getValue('watchmode_api_key',''); const key = prompt('Enter Watchmode API Key (saved to GM storage):', current || ''); if(key === null) return; await GM_setValue('watchmode_api_key', key.trim()); alert('API key saved.'); await ensureAllLocalFiles(); await loadAllPackedMappings(); }
  document.getElementById('wm-settings').onclick = openSettings;

  /* ---------- main run handler ---------- */
  async function runHandler(){
    const totalStart = now();
    logDiv.innerHTML = ''; statusEl.textContent = 'running…'; document.getElementById('wm-run').disabled = true;
    try {
      await ensureAllLocalFiles();
      // only load if not already loaded (avoid redoing work)
      if(!(packedTmdb2wmMovie && packedTmdb2wmTv && packedWm2tmdb)) {
        await loadAllPackedMappings();
      }

      const idInput = document.getElementById('wm-id').value.trim();
      if(idInput){
        logMsg(`Watchmode ID Entered: ${idInput}`, true);
        await callDetailsApi(idInput);
        return;
      }

      const path = location.pathname;
      let m = path.match(/\/movie\/(\d+)/i);
      let kind = 'movie';
      if(!m){ m = path.match(/\/tv\/(\d+)/i); kind = 'tv'; }
      if(!m){ alert('No Watchmode ID entered and could not detect a TMDB id from the current URL. Visit a TMDB movie/tv page.'); return; }
      const tmdbId = Number(m[1]);
      logMsg(`Attempting to resolve from TMDB id: ${tmdbId} (${kind})`, true);

      let wm = null;
      try {
        wm = lookupTmdbToWatchmodeSync(kind, tmdbId);
        if(wm !== null){ logMsg(`Local packed mapping resolved Watchmode ID: ${wm}`, true); }
        else { logMsg('Not found locally in packed tmdb->watchmode; trying API fallback if available', false); }
      } catch(e){ logMsg('Local packed lookup error: ' + ((e&&e.message) ? e.message : String(e)), true); }

      if(!wm){
        const apiKey = await GM_getValue('watchmode_api_key','');
        if(!apiKey){ logMsg('No Watchmode API key set — cannot perform API fallback.', true); return; }
        wm = await searchWatchmodeViaApi(tmdbId, kind);
      }

      if(wm){
        document.getElementById('wm-id').value = String(wm);
        const details = await callDetailsApi(wm);
        if(details && Array.isArray(details.similar_titles) && details.similar_titles.length){
          logMsg(`Found ${details.similar_titles.length} similar_titles — resolving to TMDB ids (preserving order)`, true);
          const resolved = resolveWatchmodeIdsPackedBatchSync(details.similar_titles); // sync fast path
          const minimal = resolved.map(r => ({ watchmode_id: r.watchmode_id, tmdb_id: r.tmdb_id, tmdb_type: r.tmdb_type } ));
          const minimalStr = JSON.stringify(minimal, null, 2);
          logMsg('===== RESOLVED SIMILAR TITLES (ordered) =====', true);
          logHtml(`<pre style="white-space:pre-wrap;max-height:300px;overflow:auto;">${escapeHtml(minimalStr)}</pre>`);
          try { GM_setClipboard(minimalStr); logMsg('Resolved JSON copied to clipboard', false); } catch(e){}
        } else {
          logMsg('No similar_titles present in details response', false);
        }
      } else {
        logMsg('Watchmode id could not be resolved.', true);
      }
    } finally {
      const totalEnd = now();
      logMsg('', false);
      logMsg(`TOTAL Time: ${(totalEnd - totalStart).toFixed(2)} ms`, true);
      statusEl.textContent = ''; document.getElementById('wm-run').disabled = false;
    }
  }
  document.getElementById('wm-run').onclick = runHandler;

  /* ---------- API helpers (unchanged) ---------- */
  async function searchWatchmodeViaApi(tmdbId, kind){
    const apiKey = await GM_getValue('watchmode_api_key', '');
    if(!apiKey) throw new Error('No Watchmode API key set');
    const field = (kind === 'tv') ? 'tmdb_series_id' : 'tmdb_movie_id';
    const url = `https://api.watchmode.com/v1/search/?apiKey=${encodeURIComponent(apiKey)}&search_field=${encodeURIComponent(field)}&search_value=${encodeURIComponent(tmdbId)}`;
    logMsg(`Calling Watchmode search API as fallback: ${url}`);
    const resp = await gmJsonWithTiming(url);
    if(resp && resp.status === 200 && resp.data && Array.isArray(resp.data.title_results) && resp.data.title_results.length){
      const id = Number(resp.data.title_results[0].id);
      logMsg(`Watchmode search returned id ${id}`);
      return id;
    }
    return null;
  }

  async function callDetailsApi(watchmodeId){
    const apiKey = await GM_getValue('watchmode_api_key','');
    if(!apiKey){ alert('Please set your Watchmode API key in Settings.'); return null; }
    const url = `https://api.watchmode.com/v1/title/${watchmodeId}/details/?apiKey=${encodeURIComponent(apiKey)}`;
    logHtml(`<b>API URL:</b> ${escapeHtml(url)}`);
    try {
      const res = await gmJsonWithTiming(url);
      logMsg(`Status: ${res.status}`);
      logMsg(`Fetch Time: ${res.fetchTimeMs.toFixed(2)} ms`);
      logMsg(`Parse Time: ${res.parseTimeMs.toFixed(2)} ms`);
      logMsg(`Total Time: ${(res.fetchTimeMs + res.parseTimeMs).toFixed(2)} ms`, true);
      logMsg('');
      logMsg('===== RAW RESPONSE (truncated) =====', true);
      try {
        const preview = JSON.stringify(res.data, (k,v)=> (typeof v === 'string' && v.length>200) ? v.slice(0,200)+'...[truncated]' : v, 2);
        logHtml(`<pre style="white-space:pre-wrap;max-height:160px;overflow:auto;">${escapeHtml(preview)}</pre>`);
      } catch(e){}
      return res.data;
    } catch(e){
      logMsg('Error calling Watchmode details: ' + ((e&&e.message) ? e.message : String(e)), true);
      return null;
    }
  }

  /* ---------- init ---------- */
  async function initAutoDownload(){
    try {
      metaStatusSpan.textContent = 'meta: checking…';
      metaStatusSpan.style.color = '#888';
      await ensureAllLocalFiles();
      await loadAllPackedMappings();
      metaStatusSpan.textContent = 'meta: packed mappings loaded';
      metaStatusSpan.style.color = '#666';
      logMsg('All packed mappings loaded into RAM (typed arrays / direct views).');
      logMsg(`Bench load time: ${bench.loadMs.toFixed(2)} ms`, false);
    } catch(e){
      metaStatusSpan.textContent = 'meta: missing';
      metaStatusSpan.style.color = '#b00';
      console.warn('initAutoDownload error', e);
    }
  }
  initAutoDownload().catch(e=>{ console.warn('auto-init fail', e); });

  panel.querySelector('b').addEventListener('dblclick', ()=>{ logDiv.innerHTML=''; logMsg('Log cleared.'); });
  logMsg('Ready. This userscript uses zero-copy views when possible and keeps the ultra-fast hot path.', false);

})();
