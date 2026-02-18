// ==UserScript==
// @name         Watchmode Local Mapping - EF + Values (full workflow) 006
// @namespace    http://tampermonkey.net/
// @version      2.0-ef-values
// @description  Load Elias–Fano blobs + companion values into RAM and perform full workflow: tmdb->watchmode -> details -> resolve similar_titles -> TMDB ids (preserves order). Copies final JSON to clipboard.
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
  const WM_VERSION = '2.0-ef-values';
  const BASE = "https://hcgiub001.github.io/FLAMWM"; // update to your host if needed
  const STORAGE_PREFIX = "wm_local_ef_";
  const REQUEST_TIMEOUT_MS = 30000;

  // prefixes/names produced by your Python EF script and the legacy values names (values files are required)
  const PREFIXES = {
    tmdb2wm_movie: "tmdb2wm.movie",
    tmdb2wm_tv:    "tmdb2wm.tv",
    wm2tmdb:       "wm2tmdb"
  };

  function efFilesFor(prefixName) {
    const base = prefixName;
    return {
      // EF blobs
      ef_meta: `${base}.ef.meta.json`,
      ef_low:  `${base}.ef.low`,
      ef_high: `${base}.ef.high`,
      ef_super: `${base}.ef.super.u32`,
      ef_blocks_u16: `${base}.ef.blocks.u16`,
      ef_blocks_u32: `${base}.ef.blocks.u32`,
      sample: `${base}.sample.u32`,
      // companion values (these must exist for full workflow)
      // naming follows your original script's legacy writer for values.
      // tmdb2wm.movie -> "tmdb2wm.movie.watchmode.movie.values.bin"
      // tmdb2wm.tv    -> "tmdb2wm.tv.watchmode.tv.values.bin"
      // wm2tmdb       -> "wm2tmdb.packed_tmdb.values.bin"
      values: (prefixName === PREFIXES.tmdb2wm_movie) ? `${base}.watchmode.movie.values.bin`
            : (prefixName === PREFIXES.tmdb2wm_tv)    ? `${base}.watchmode.tv.values.bin`
            : `${base}.packed_tmdb.values.bin`,
      // legacy meta.json (useful to know val bytes & sample stride; small, safe to load)
      meta: `${base}.meta.json`
    };
  }

  const FILESETS = {
    movie: efFilesFor(PREFIXES.tmdb2wm_movie),
    tv:    efFilesFor(PREFIXES.tmdb2wm_tv),
    wm2tmdb: efFilesFor(PREFIXES.wm2tmdb)
  };

  /* ---------- UI (same look) ---------- */
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
      <b>Watchmode EF+Values</b>
      <div style="margin-left:8px;color:#666;font-size:11px">Elias–Fano keys + values (full workflow)</div>
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

  /* ---------- stats & bench ---------- */
  let totalDownloadedBytes = 0; // counts bytes downloaded/read from storage
  let totalRamBytes = 0;        // bytes loaded into RAM
  let bench = { loadMs:0, numEfSelects:0, numBinarySearches:0, numSamplesUsed:0 };

  /* ---------- small utilities ---------- */
  function humanBytes(n){ if(n === null || n === undefined) return '-'; if(n < 1024) return `${n} B`; if(n < 1024*1024) return `${(n/1024).toFixed(1)} KB`; return `${(n/(1024*1024)).toFixed(2)} MB`; }
  function readU32ArrayFromBuffer(ab){
    const u8 = new Uint8Array(ab);
    const count = Math.floor(u8.length / 4);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const out = new Uint32Array(count);
    for(let i=0;i<count;i++) out[i] = dv.getUint32(i*4, true) >>> 0;
    return out;
  }
  function readSampleU32(ab){
    return readU32ArrayFromBuffer(ab);
  }
  // popcount for 32-bit unsigned ints (fast)
  function popcount32(x){
    x = x >>> 0;
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    x = (x + (x >>> 4)) & 0x0F0F0F0F;
    x = x + (x >>> 8);
    x = x + (x >>> 16);
    return x & 0xFF;
  }

  /* ---------- EF runtime builder & select (same robust implementation) ---------- */
  function buildEfIndexFromBlobs(meta, lowAb, highAb, superAb, blocksAb, sampleAb){
    const ef = {};
    ef.meta = meta;
    ef.n = Number(meta.n || 0);
    ef.L = Number(meta.L || 0);
    ef.low_bits_len_bits = Number(meta.low_bits_len_bits || 0);
    ef.low_bytes = lowAb ? lowAb.byteLength : 0;
    ef.lowU8 = lowAb ? new Uint8Array(lowAb) : new Uint8Array(0);
    ef.high_bits_len_bits = Number(meta.high_bits_len_bits || 0);
    ef.high_bytes = highAb ? highAb.byteLength : 0;
    ef.highU8 = highAb ? new Uint8Array(highAb) : new Uint8Array(0);

    // build u32 words for high
    ef.highU32 = new Uint32Array(Math.ceil(ef.highU8.length / 4));
    if(ef.highU8.length > 0){
      const dv = new DataView(ef.highU8.buffer, ef.highU8.byteOffset, ef.highU8.byteLength);
      for(let i=0;i<ef.highU32.length;i++){
        const offset = i*4;
        if(offset + 4 <= ef.highU8.length){
          ef.highU32[i] = dv.getUint32(offset, true) >>> 0;
        } else {
          let w = 0;
          for(let b=0;b<4 && (offset+b) < ef.highU8.length; b++){
            w |= (ef.highU8[offset + b]) << (8*b);
          }
          ef.highU32[i] = w >>> 0;
        }
      }
    }

    ef.superRanks = superAb ? readU32ArrayFromBuffer(superAb) : new Uint32Array(0);

    ef.blockBits = (meta.ef_aux && meta.ef_aux.block_bits) ? Number(meta.ef_aux.block_bits) : 64;
    ef.superblockBits = (meta.ef_aux && meta.ef_aux.superblock_bits) ? Number(meta.ef_aux.superblock_bits) : 2048;
    ef.numBlocks = Math.ceil(ef.high_bits_len_bits / ef.blockBits);

    const perBlockBytes = (meta.ef_aux && meta.ef_aux.per_block_bytes) ? Number(meta.ef_aux.per_block_bytes) : null;
    if(blocksAb && blocksAb.byteLength > 0){
      if(perBlockBytes === 2) {
        // read u16
        const u8 = new Uint8Array(blocksAb);
        const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
        const count = Math.floor(u8.length / 2);
        ef.blockOffsets = new Uint32Array(count);
        for(let i=0;i<count;i++) ef.blockOffsets[i] = dv.getUint16(i*2, true) >>> 0;
      } else {
        // u32
        ef.blockOffsets = readU32ArrayFromBuffer(blocksAb);
      }
    } else ef.blockOffsets = new Uint32Array(0);

    // low-bit reader
    ef.readLow = function(k){
      const L = ef.L;
      if(L === 0) return 0;
      const bitStart = k * L;
      let value = 0;
      let bitsRead = 0;
      let byteIdx = bitStart >>> 3;
      let bitOffset = bitStart & 7;
      let remaining = L;
      while(remaining > 0){
        const cur = ef.lowU8[byteIdx] >>> 0;
        const take = Math.min(remaining, 8 - bitOffset);
        const mask = ((1 << take) - 1);
        const bits = (cur >>> bitOffset) & mask;
        value |= (bits << bitsRead);
        bitsRead += take;
        remaining -= take;
        byteIdx++;
        bitOffset = 0;
      }
      return value >>> 0;
    };

    ef.selectPos = function(k){
      if(k < 0 || k >= ef.n) return -1;
      // find superblock
      let lo = 0, hi = ef.superRanks.length - 1;
      while(lo < hi){
        const mid = (lo + hi) >>> 1;
        if(ef.superRanks[mid] <= k) lo = mid + 1;
        else hi = mid;
      }
      let s = Math.max(0, lo - 1);
      if(s >= ef.superRanks.length - 1) s = ef.superRanks.length - 2;
      if(s < 0) s = 0;
      const superRankStart = ef.superRanks[s];
      const blocksPerSuper = Math.ceil(ef.superblockBits / ef.blockBits);
      const bStart = s * blocksPerSuper;
      const bEndExclusive = Math.min(ef.numBlocks, bStart + blocksPerSuper);
      const targetRel = k - superRankStart;

      // binary search block in that range
      let blo = bStart, bhi = Math.max(bStart, bEndExclusive - 1), bfound = bStart;
      while(blo <= bhi){
        const bmid = (blo + bhi) >>> 1;
        const offMid = ef.blockOffsets[bmid] || 0;
        if(offMid <= targetRel) { bfound = bmid; blo = bmid + 1; }
        else { bhi = bmid - 1; }
      }
      const blockIndex = bfound;
      const onesBeforeBlock = superRankStart + (ef.blockOffsets[blockIndex] || 0);

      const blockBitStart = blockIndex * ef.blockBits;
      const blockBitEnd = Math.min(ef.high_bits_len_bits, blockBitStart + ef.blockBits);

      let onesSoFar = onesBeforeBlock;
      const startWord = blockBitStart >>> 5;
      const endWord = ((blockBitEnd - 1) >>> 5);
      for(let wi = startWord; wi <= endWord; wi++){
        const wordVal = (wi < ef.highU32.length) ? ef.highU32[wi] : 0;
        const wordStartBit = wi * 32;
        const from = Math.max(blockBitStart, wordStartBit);
        const to = Math.min(blockBitEnd, wordStartBit + 32);
        const sh = from - wordStartBit;
        const lenBits = to - from;
        const mask = lenBits === 32 ? 0xFFFFFFFF : ((1 << lenBits) - 1);
        const segment = (wordVal >>> sh) & mask;
        const pop = popcount32(segment);
        if(onesSoFar + pop > k){
          let acc = onesSoFar;
          for(let b=0;b<lenBits;b++){
            if(((segment >>> b) & 1) !== 0){
              if(acc === k) return (wordStartBit + sh + b);
              acc++;
            }
          }
        }
        onesSoFar += pop;
      }
      // fallback
      let globalOnes = 0;
      for(let wi=0; wi<ef.highU32.length; wi++){
        const w = ef.highU32[wi];
        const pop = popcount32(w);
        if(globalOnes + pop > k){
          let acc = globalOnes;
          for(let b=0;b<32;b++){
            if(((w >>> b) & 1) !== 0){
              if(acc === k) return wi*32 + b;
              acc++;
            }
          }
        }
        globalOnes += pop;
      }
      return -1;
    };

    ef.selectValue = function(k){
      bench.numEfSelects++;
      const pos = ef.selectPos(k);
      if(pos < 0) return null;
      const high = pos - k;
      const low = ef.readLow(k);
      const valBig = (BigInt(high) << BigInt(ef.L)) | BigInt(low);
      if(valBig <= BigInt(Number.MAX_SAFE_INTEGER)) return Number(valBig);
      return valBig;
    };

    // sample array
    ef.sample = sampleAb ? readSampleU32(sampleAb) : new Uint32Array(0);
    ef.sampleStride = meta.sample_stride || meta.sampleStride || 64;

    return ef;
  }

  /* ---------- load EF blobs + values for one mapping ---------- */
  async function ensureEfAndValues(files, prefixKey){
    // getOrDownload: try storage then network; update totalDownloadedBytes
    async function getOrDownload(name, url){
      const stored = await loadLocalFileArrayBuffer(name);
      if(stored){
        totalDownloadedBytes += stored.byteLength;
        return stored;
      }
      try {
        const resp = await gmArrayBufferWithTiming(url);
        if(resp && resp.buffer && resp.status >= 200 && resp.status < 300){
          totalDownloadedBytes += resp.buffer.byteLength;
          await saveLocalFile(name, resp.buffer);
          return resp.buffer;
        }
      } catch(e){}
      return null;
    }

    const metaAb = await getOrDownload(files.ef_meta, `${BASE}/${files.ef_meta}`);
    if(!metaAb){ logMsg(`Missing EF meta: ${files.ef_meta}`, true); return null; }
    let meta = null;
    try { meta = JSON.parse(new TextDecoder().decode(metaAb)); } catch(e){ logMsg(`Bad EF meta JSON: ${files.ef_meta}`, true); return null; }

    // choose blocks file per meta if possible
    const perBlockBytes = (meta.ef_aux && meta.ef_aux.per_block_bytes) ? Number(meta.ef_aux.per_block_bytes) : null;
    const candidateBlocks = [];
    if(perBlockBytes === 4) candidateBlocks.push(files.ef_blocks_u32);
    else if(perBlockBytes === 2) candidateBlocks.push(files.ef_blocks_u16);
    else {
      // try u16 then u32
      candidateBlocks.push(files.ef_blocks_u16, files.ef_blocks_u32);
    }

    const lowAb = await getOrDownload(files.ef_low, `${BASE}/${files.ef_low}`);
    const highAb = await getOrDownload(files.ef_high, `${BASE}/${files.ef_high}`);
    const superAb = await getOrDownload(files.ef_super, `${BASE}/${files.ef_super}`);

    let blocksAb = null;
    for(const bn of candidateBlocks){
      if(!bn) continue;
      const buf = await getOrDownload(bn, `${BASE}/${bn}`);
      if(buf){ blocksAb = buf; break; }
    }

    const sampleAb = await getOrDownload(files.sample, `${BASE}/${files.sample}`);

    // values file (required for full workflow)
    const valuesAb = await getOrDownload(files.values, `${BASE}/${files.values}`);
    // optional legacy meta (for val bytes info)
    const legacyMetaAb = await getOrDownload(files.meta, `${BASE}/${files.meta}`);

    if(!lowAb || !highAb || !superAb || !blocksAb || !sampleAb){
      logMsg(`EF incomplete for ${prefixKey} - required blobs missing`, true);
      return null;
    }
    if(!valuesAb){
      logMsg(`Values file missing for ${prefixKey} -> ${files.values}. Without values the script can only return EF index position, not actual IDs.`, true);
      // We still build EF index so you can inspect positions, but full workflow will fail at mapping to IDs.
    }

    // Build ef runtime index
    const efIndex = buildEfIndexFromBlobs(meta, lowAb, highAb, superAb, blocksAb, sampleAb);
    // sizes
    efIndex._sizes = { low: lowAb.byteLength, high: highAb.byteLength, super: superAb.byteLength, blocks: blocksAb.byteLength, sample: sampleAb.byteLength, values: valuesAb ? valuesAb.byteLength : 0, meta: metaAb.byteLength || 0 };

    // load values into typed array (assume 4-byte u32 little-endian by writer)
    let values32 = null;
    if(valuesAb){
      values32 = readU32ArrayFromBuffer(valuesAb);
      totalRamBytes += valuesAb.byteLength;
    }

    // update RAM accounting for EF blobs (we consider we loaded them fully)
    totalRamBytes += efIndex._sizes.low + efIndex._sizes.high + efIndex._sizes.super + efIndex._sizes.blocks + efIndex._sizes.sample + efIndex._sizes.meta;
    return { ef: efIndex, values32: values32, metaLegacy: legacyMetaAb ? JSON.parse(new TextDecoder().decode(legacyMetaAb)) : null };
  }

  /* ---------- global mapping holders ---------- */
  let ef_tmdb2wm_movie = null;
  let ef_tmdb2wm_tv = null;
  let ef_wm2tmdb = null;

  /* ---------- load all mappings ---------- */
  async function loadAllEfMappings(){
    const t0 = now();
    totalDownloadedBytes = 0;
    totalRamBytes = 0;
    bench.loadMs = 0;
    try {
      const a = await ensureEfAndValues(FILESETS.movie, 'tmdb2wm.movie');
      ef_tmdb2wm_movie = a;
      const b = await ensureEfAndValues(FILESETS.tv, 'tmdb2wm.tv');
      ef_tmdb2wm_tv = b;
      const c = await ensureEfAndValues(FILESETS.wm2tmdb, 'wm2tmdb');
      ef_wm2tmdb = c;
      bench.loadMs = (now() - t0);

      // summary
      const parts = [];
      function describe(name, obj){
        if(!obj) return;
        const s = obj.ef._sizes;
        parts.push(`${name}(sample=${humanBytes(s.sample)},low=${humanBytes(s.low)},high=${humanBytes(s.high)},vals=${humanBytes(s.values)})`);
      }
      describe('tmdb2wm.movie', ef_tmdb2wm_movie);
      describe('tmdb2wm.tv', ef_tmdb2wm_tv);
      describe('wm2tmdb', ef_wm2tmdb);

      logMsg(`EF mappings loaded into RAM: ${parts.join(' ; ')}`, false);
      logMsg(`Total downloaded/read bytes: ${humanBytes(totalDownloadedBytes)}`, false);
      logMsg(`Total RAM used (approx): ${humanBytes(totalRamBytes)}`, false);
      logMsg(`Load time: ${bench.loadMs.toFixed(2)} ms`, false);
      return true;
    } catch(e){
      logMsg(`loadAllEfMappings error: ${e && e.message ? e.message : e}`, true);
      return false;
    }
  }

  /* ---------- EF-based lookup and value retrieval ---------- */

  // Use sample to narrow then binary search using ef.selectValue
  function lookupEfPackedSync(efContainer, packedKey){
    if(!efContainer || !efContainer.ef) return -1;
    const efIndex = efContainer.ef;
    const sampleArr = efIndex.sample;
    const S = efIndex.sampleStride || 64;
    // upperBound sample
    let lo = 0, hi = sampleArr.length;
    while(lo < hi){
      const mid = (lo + hi) >>> 1;
      if(sampleArr[mid] > packedKey) hi = mid;
      else lo = mid + 1;
    }
    const sb = lo - 1;
    let left = (sb >= 0) ? sb * S : 0;
    if(sb >= 0) bench.numSamplesUsed++;
    const right = Math.min(efIndex.n - 1, Math.max(left, ((sb + 1) * S) - 1));

    // binary search within [left, right]
    let l = left, r = right;
    bench.numBinarySearches++;
    while(l <= r){
      const mid = (l + r) >>> 1;
      const v = efIndex.selectValue(mid);
      const vv = (typeof v === 'bigint') ? Number(v) : v;
      if(vv === packedKey) return mid;
      if(vv < packedKey) l = mid + 1;
      else r = mid - 1;
    }
    return -1;
  }

  // tmdb->watchmode: packed = (tmdb<<1)|is_tv
  function lookupTmdbToWatchmodeSync(kind, tmdbId){
    const packed = (BigInt(tmdbId) << 1n) | (kind === 'tv' ? 1n : 0n);
    const packedKey = (packed <= BigInt(Number.MAX_SAFE_INTEGER)) ? Number(packed) : packed;
    const efContainer = (kind === 'movie') ? ef_tmdb2wm_movie : ef_tmdb2wm_tv;
    if(!efContainer) return null;
    const pos = lookupEfPackedSync(efContainer, packedKey);
    if(pos < 0) return null;
    // read value
    if(efContainer.values32){
      const val = efContainer.values32[pos] >>> 0;
      return val; // watchmode id
    } else {
      // values missing
      return { pos };
    }
  }

  // watchmode -> packed_tmdb (values are packed_tmdb)
  function lookupWatchmodeToTmdbSync(wmid){
    const packedKey = Number(wmid) >>> 0;
    if(!ef_wm2tmdb) return null;
    const pos = lookupEfPackedSync(ef_wm2tmdb, packedKey);
    if(pos < 0) return null;
    if(ef_wm2tmdb.values32){
      const packedVal = ef_wm2tmdb.values32[pos] >>> 0;
      return { tmdb_id: (packedVal >>> 1), tmdb_type: ((packedVal & 1) ? 'tv' : 'movie') };
    } else {
      return { pos };
    }
  }

  /* ---------- batch resolver (keeping same semantics as original) ---------- */
  function resolveWatchmodeIdsPackedBatchSync(watchmodeIds){
    const tStart = now();
    bench.numBinarySearches = 0; bench.numEfSelects = 0; bench.numSamplesUsed = 0;

    if(!Array.isArray(watchmodeIds) || watchmodeIds.length === 0) return [];

    const orig = watchmodeIds.map(x => Number(x));
    // dedupe preserving first occurrence
    const seen = Object.create(null);
    const unique = [];
    for(const v of orig){ if(!Number.isFinite(v)) continue; if(!(v in seen)){ seen[v] = true; unique.push(v); } }
    if(unique.length === 0) return [];

    unique.sort((a,b)=>a-b);

    if(!ef_wm2tmdb) return orig.map(n=>({watchmode_id:n, tmdb_id:null, tmdb_type:null, found_local:false, source:'missing'}));

    const results = Object.create(null);
    // simple path: per-id EF lookup (ok for typical counts like 25)
    for(const id of unique){
      const r = lookupWatchmodeToTmdbSync(id);
      if(r && r.tmdb_id !== undefined) results[id] = { tmdb_id: r.tmdb_id, tmdb_type: r.tmdb_type, found_local:true, source:'local' };
      else results[id] = { tmdb_id: null, tmdb_type:null, found_local:false, source:'not_found' };
    }

    const ordered = orig.map(n => {
      const r = results[n] || { tmdb_id:null, tmdb_type:null, found_local:false, source:'missing' };
      return { watchmode_id: n, tmdb_id: r.tmdb_id, tmdb_type: r.tmdb_type, found_local: r.found_local, source: r.source };
    });

    const tEnd = now();
    logMsg(`Batch resolution (EF) finished in ${(tEnd - tStart).toFixed(4)} ms`, true);
    logMsg(`Benchmark (EF): binarySearches=${bench.numBinarySearches}, efSelects=${bench.numEfSelects}, samplesUsed=${bench.numSamplesUsed}, loadMs=${bench.loadMs.toFixed(2)}ms`, false);
    return ordered;
  }

  /* ---------- UI glue & main run flow (same behavior as original) ---------- */
  async function openSettings(){
    const current = await GM_getValue('watchmode_api_key','');
    const key = prompt('Enter Watchmode API Key (saved to GM storage):', current || '');
    if(key === null) return;
    await GM_setValue('watchmode_api_key', (key || '').trim());
    alert('API key saved.');
    await loadAllEfMappings();
  }
  document.getElementById('wm-settings').onclick = openSettings;

  async function runHandler(){
    const totalStart = now();
    logDiv.innerHTML = ''; statusEl.textContent = 'running…'; document.getElementById('wm-run').disabled = true;
    try {
      metaStatusSpan.textContent = 'meta: loading…';
      metaStatusSpan.style.color = '#888';
      await loadAllEfMappings();
      metaStatusSpan.textContent = 'meta: loaded';
      metaStatusSpan.style.color = '#666';

      const idInput = document.getElementById('wm-id').value.trim();
      if(idInput){
        logMsg(`Watchmode ID Entered: ${idInput}`, true);
        const details = await callDetailsApi(idInput);
        logMsg('Details fetched (see log).', false);
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
        if(wm !== null && typeof wm === 'number'){ logMsg(`Local EF mapping resolved Watchmode ID: ${wm}`, true); }
        else if(wm && wm.pos !== undefined){ logMsg(`EF index found: pos=${wm.pos} (values missing)`, true); }
        else { logMsg('Not found locally in EF; trying API fallback if available', false); }
      } catch(e){ logMsg('Local EF lookup error: ' + ((e&&e.message) ? e.message : String(e)), true); }

      if(!wm || (wm && wm.pos !== undefined)){
        const apiKey = await GM_getValue('watchmode_api_key','');
        if(!apiKey){ logMsg('No Watchmode API key set — cannot perform API fallback.', true); return; }
        // fallback: search by tmdb via Watchmode API
        const found = await searchWatchmodeViaApi(tmdbId, kind);
        if(found){
          wm = found;
          logMsg(`Watchmode search returned id ${wm}`, false);
        }
      }

      if(wm && typeof wm === 'number'){
        document.getElementById('wm-id').value = String(wm);
        const details = await callDetailsApi(wm);
        if(details && Array.isArray(details.similar_titles) && details.similar_titles.length){
          logMsg(`Found ${details.similar_titles.length} similar_titles — resolving to TMDB ids (preserving order)`, true);
          const resolved = resolveWatchmodeIdsPackedBatchSync(details.similar_titles); // EF-based batch
          const minimal = resolved.map(r => ({ watchmode_id: r.watchmode_id, tmdb_id: r.tmdb_id, tmdb_type: r.tmdb_type } ));
          const minimalStr = JSON.stringify(minimal, null, 2);
          logMsg('===== RESOLVED SIMILAR TITLES (ordered) =====', true);
          logHtml(`<pre style="white-space:pre-wrap;max-height:300px;overflow:auto;">${escapeHtml(minimalStr)}</pre>`);
          try { GM_setClipboard(minimalStr); logMsg('Resolved JSON copied to clipboard', false); } catch(e){}
        } else {
          logMsg('No similar_titles present in details response', false);
        }
      } else {
        logMsg('Watchmode id could not be resolved (EF values missing and API fallback unavailable).', true);
      }
    } finally {
      const totalEnd = now();
      logMsg('', false);
      logMsg(`TOTAL Time: ${(totalEnd - totalStart).toFixed(2)} ms`, true);
      statusEl.textContent = ''; document.getElementById('wm-run').disabled = false;
    }
  }
  document.getElementById('wm-run').onclick = runHandler;

  /* ---------- Watchmode API helpers (unchanged) ---------- */
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

  /* ---------- init (auto-load EF files + values) ---------- */
  async function initAutoLoad(){
    try {
      metaStatusSpan.textContent = 'meta: checking…';
      metaStatusSpan.style.color = '#888';
      await loadAllEfMappings();
      metaStatusSpan.textContent = 'meta: loaded';
      metaStatusSpan.style.color = '#666';
      logMsg('EF+values mappings loaded into RAM.', false);
      logMsg(`Total downloaded/read bytes: ${humanBytes(totalDownloadedBytes)}`, false);
      logMsg(`Total RAM used (approx): ${humanBytes(totalRamBytes)}`, false);
    } catch(e){
      metaStatusSpan.textContent = 'meta: missing';
      metaStatusSpan.style.color = '#b00';
      console.warn('initAutoLoad error', e);
    }
  }
  initAutoLoad().catch(e=>{ console.warn('auto-init fail', e); });

  panel.querySelector('b').addEventListener('dblclick', ()=>{ logDiv.innerHTML=''; logMsg('Log cleared.'); });
  logMsg('Ready. This script uses EF keys + companion values (values must exist) and reproduces your original workflow.', false);

})();
