#!/usr/bin/env python3
"""
build_all_bundles.py

Downloads Watchmode CSV (or reads local CSV) and builds three bundle sets:
 - wm2tmdb.{bin,index,meta.json}   (watchmode -> tmdb)  [index_spacing default 100]
 - movie.{bin,index,meta.json}     (tmdb -> watchmode, keep duplicates) [index_spacing default 100]
 - tv.{bin,index,meta.json}        (tmdb -> watchmode, keep duplicates) [index_spacing default 100]

Writes gzipped assets and latest.json manifest.

Usage:
  python build_all_bundles.py --out ./out --csv /path/to/title_id_map.csv
or
  python build_all_bundles.py --out ./out
(then it will download from WATCHMODE_CSV_URL or from the default watchmode URL)
"""
from __future__ import annotations
import os
import sys
import json
import struct
import argparse
from pathlib import Path
import urllib.request
import hashlib
import gzip
import time
from datetime import datetime
from typing import Tuple, List
import tempfile

# use pandas for CSV handling
import pandas as pd

# Defaults
WATCHMODE_CSV_URL = "https://api.watchmode.com/datasets/title_id_map.csv"
DEFAULT_INDEX_SPACING = 100  # as requested
# Index struct for 32-bit offsets: first:uint32, offset:uint32
INDEX_STRUCT_32 = '<II'
INDEX_ENTRY_SIZE_32 = struct.calcsize(INDEX_STRUCT_32)


# ---------------- utilities ----------------

def download_url_to_path(url: str, out_path: Path):
    print(f"Downloading {url} -> {out_path}")
    with urllib.request.urlopen(url) as r:
        with open(out_path, 'wb') as f:
            f.write(r.read())

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def gzip_file(src: Path, dest: Path):
    with open(src, 'rb') as sf, gzip.open(dest, 'wb', mtime=0) as gf:
        while True:
            chunk = sf.read(1 << 20)
            if not chunk:
                break
            gf.write(chunk)

def uleb128_encode(n: int) -> bytes:
    if n < 0:
        raise ValueError("uleb128_encode expects non-negative integer")
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


# ---------------- builder: wm2tmdb (watchmode->tmdb) ----------------
def build_wm2tmdb(df: pd.DataFrame, out_prefix: Path, index_spacing: int = DEFAULT_INDEX_SPACING) -> Tuple[Path, Path, Path]:
    """
    df expected columns: ['watchmode_id','tmdb_id','tmdb_type_norm'] (watchmode sorted ascending)
    Produces out_prefix.bin, out_prefix.index, out_prefix.meta.json
    Index entries: '<II' (first_watchmode:uint32, offset:uint32)
    """
    bin_path = out_prefix.with_suffix('.bin')
    index_path = out_prefix.with_suffix('.index')
    meta_path = out_prefix.with_suffix('.meta.json')

    total = len(df)
    if total == 0:
        bin_path.write_bytes(b'')
        index_path.write_bytes(b'')
        meta = {"index_spacing": index_spacing, "index_entry_size": INDEX_ENTRY_SIZE_32,
                "tmdb_packed_bits": 1, "total_records": 0, "files": {"bin": bin_path.name, "index": index_path.name}}
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return bin_path, index_path, meta_path

    print(f"Building wm2tmdb bundle: {total} records, spacing={index_spacing}")
    index_entries: List[Tuple[int, int]] = []
    with open(bin_path, 'wb') as bf:
        prev_wm = None
        for i, row in df.iterrows():
            wm = int(row['watchmode_id'])
            tmdb = int(row['tmdb_id'])
            is_tv = 1 if row['tmdb_type_norm'] == 'tv' else 0
            if (i % index_spacing) == 0:
                offset = bf.tell()
                index_entries.append((wm, offset))
                prev_wm = wm
                bf.write(uleb128_encode(0))
            else:
                if prev_wm is None:
                    prev_wm = wm
                    bf.write(uleb128_encode(0))
                else:
                    delta = wm - prev_wm
                    if delta < 0:
                        raise RuntimeError("Input must be sorted ascending by watchmode id")
                    bf.write(uleb128_encode(delta))
                    prev_wm = wm
            packed = (tmdb << 1) | is_tv
            bf.write(uleb128_encode(packed))

    # sanity check: 32-bit offsets
    bin_size = bin_path.stat().st_size
    if bin_size >= 0x1_0000_0000:
        raise RuntimeError("bin too large for 32-bit offsets")

    # write index (32-bit offsets)
    with open(index_path, 'wb') as idxf:
        for first_wm, offset in index_entries:
            idxf.write(struct.pack(INDEX_STRUCT_32, int(first_wm), int(offset)))

    meta = {
        "index_spacing": index_spacing,
        "index_entry_size": INDEX_ENTRY_SIZE_32,
        "tmdb_packed_bits": 1,
        "total_records": total,
        "max_watchmode": int(df['watchmode_id'].max()),
        "max_tmdb": int(df['tmdb_id'].max()),
        "files": {"bin": bin_path.name, "index": index_path.name}
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(f"  -> wm2tmdb: bin {bin_size} bytes, index entries {len(index_entries)}")
    return bin_path, index_path, meta_path


# ---------------- builder: tmdb -> watchmode (movie/tv), KEEP duplicates ----------------
def write_tmdb_to_wm_bundle(df: pd.DataFrame, out_prefix: Path, index_spacing: int = DEFAULT_INDEX_SPACING) -> Tuple[Path, Path, Path]:
    """
    df sorted ascending by tmdb_id. We keep duplicates (multiple watchmode IDs per tmdb).
    Writes fixed-width watchmode ids (min bytes required), index entries '<II' (first_tmdb:uint32, offset:uint32).
    """
    bin_path = out_prefix.with_suffix('.bin')
    index_path = out_prefix.with_suffix('.index')
    meta_path = out_prefix.with_suffix('.meta.json')

    total = len(df)
    if total == 0:
        bin_path.write_bytes(b'')
        index_path.write_bytes(b'')
        meta = {"index_spacing": index_spacing, "index_entry_size": INDEX_ENTRY_SIZE_32,
                "tmdb_varint": True, "watchmode_bytes": 0, "total_records": 0,
                "files": {"bin": bin_path.name, "index": index_path.name}}
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return bin_path, index_path, meta_path

    max_wm = int(df['watchmode_id'].max())
    # choose minimal fixed bytes
    if max_wm < (1 << 8):
        watchmode_bytes = 1
    elif max_wm < (1 << 16):
        watchmode_bytes = 2
    elif max_wm < (1 << 24):
        watchmode_bytes = 3
    elif max_wm < (1 << 32):
        watchmode_bytes = 4
    else:
        watchmode_bytes = 8

    print(f"Writing {out_prefix.name} bundle: {total} records, watchmode_bytes={watchmode_bytes}, spacing={index_spacing}")

    index_entries: List[Tuple[int, int]] = []
    with open(bin_path, 'wb') as bf:
        prev_tmdb = None
        for i, row in df.iterrows():
            tmdb = int(row['tmdb_id'])
            wm = int(row['watchmode_id'])
            if (i % index_spacing) == 0:
                offset = bf.tell()
                index_entries.append((tmdb, offset))
                prev_tmdb = tmdb
                # write varint delta 0
                bf.write(uleb128_encode(0))
            else:
                if prev_tmdb is None:
                    prev_tmdb = tmdb
                    bf.write(uleb128_encode(0))
                else:
                    delta = tmdb - prev_tmdb
                    if delta < 0:
                        raise RuntimeError("Input CSV must be sorted ascending by TMDB id.")
                    bf.write(uleb128_encode(delta))
                    prev_tmdb = tmdb
            # write watchmode fixed-width little-endian
            bf.write(wm.to_bytes(watchmode_bytes, byteorder='little', signed=False))

    # write index with 32-bit offsets
    with open(index_path, 'wb') as idxf:
        for first_tmdb, offset in index_entries:
            idxf.write(struct.pack(INDEX_STRUCT_32, int(first_tmdb), int(offset)))

    meta = {
        "index_spacing": index_spacing,
        "index_entry_size": INDEX_ENTRY_SIZE_32,
        "tmdb_varint": True,
        "watchmode_bytes": watchmode_bytes,
        "total_records": total,
        "files": {"bin": bin_path.name, "index": index_path.name},
        "version": 1
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(f"  -> {out_prefix.name}: bin {bin_path.stat().st_size} bytes, index entries {len(index_entries)}")
    return bin_path, index_path, meta_path


# ---------------- main orchestration ----------------
def main():
    parser = argparse.ArgumentParser(description="Build all mapping bundles (wm2tmdb and tmdb->watchmode movie/tv)")
    parser.add_argument('--csv', help='Path to local CSV (if omitted, downloads the default WATCHMODE CSV)', default=None)
    parser.add_argument('--out', help='Output directory', default='out')
    parser.add_argument('--index-spacing', type=int, default=DEFAULT_INDEX_SPACING, help='Index spacing (default 100)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
    else:
        # download to temporary file
        tmp = out_dir / 'title_id_map.csv'
        download_url_to_path(WATCHMODE_CSV_URL, tmp)
        csv_path = tmp

    print(f"Reading CSV {csv_path} ... (this may take a few seconds)")
    # read needed columns: watchmode id, tmdb id, tmdb type
    sample = pd.read_csv(csv_path, nrows=0)
    header = list(sample.columns)
    # attempt to detect column names
    def find_col(possible):
        for p in possible:
            for h in header:
                if h.strip().lower() == p:
                    return h
        return None
    # common names (flexible)
    watchmode_col = find_col(['watchmode id','watchmode_id','watchmodeid','watchmode'])
    tmdb_col = find_col(['tmdb id','tmdb_id','tmdbid','tmdb'])
    type_col = find_col(['tmdb type','tmdb_type','type','tmdbtype'])
    if not (watchmode_col and tmdb_col and type_col):
        print("CSV header detection failed, columns found:", header)
        raise SystemExit(1)

    usecols = [watchmode_col, tmdb_col, type_col]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=str, low_memory=False)
    df.columns = ['watchmode_id', 'tmdb_id', 'tmdb_type']
    # normalize types
    df['tmdb_type_norm'] = df['tmdb_type'].fillna('').astype(str).str.strip().str.lower().replace({
        'tv_series':'tv','tvseries':'tv','television':'tv','series':'tv','tv_show':'tv','tv show':'tv',
        'film':'movie','films':'movie','movies':'movie'
    })
    # convert numeric fields
    df['watchmode_id'] = pd.to_numeric(df['watchmode_id'], errors='coerce').astype('Int64')
    df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['watchmode_id','tmdb_id'])
    df['watchmode_id'] = df['watchmode_id'].astype(int)
    df['tmdb_id'] = df['tmdb_id'].astype(int)

    # Build unified wm2tmdb using watchmode sort order
    df_wm_sorted = df[['watchmode_id','tmdb_id','tmdb_type_norm']].drop_duplicates(subset=['watchmode_id']).sort_values('watchmode_id').reset_index(drop=True)
    wm_bin, wm_index, wm_meta = build_wm2tmdb(df_wm_sorted, out_dir / 'wm2tmdb', index_spacing=args.index_spacing)

    # Build tmdb->watchmode bundles for movie and tv separately (keep duplicates)
    df_movies = df[df['tmdb_type_norm'] == 'movie'][['tmdb_id','watchmode_id']].reset_index(drop=True).sort_values('tmdb_id').reset_index(drop=True)
    df_movies.columns = ['tmdb_id','watchmode_id']
    df_tv = df[df['tmdb_type_norm'] == 'tv'][['tmdb_id','watchmode_id']].reset_index(drop=True).sort_values('tmdb_id').reset_index(drop=True)
    df_tv.columns = ['tmdb_id','watchmode_id']

    movie_bin, movie_index, movie_meta = write_tmdb_to_wm_bundle(df_movies, out_dir / 'movie', index_spacing=args.index_spacing)
    tv_bin, tv_index, tv_meta = write_tmdb_to_wm_bundle(df_tv, out_dir / 'tv', index_spacing=args.index_spacing)

    # gzip everything and compute sha256
    generated_files = [wm_bin, wm_index, wm_meta, movie_bin, movie_index, movie_meta, tv_bin, tv_index, tv_meta]
    gz_files = []
    manifest_files = []
    for f in generated_files:
        gz = Path(str(f) + '.gz')
        gzip_file(f, gz)
        h = sha256_of_file(gz)
        sz = gz.stat().st_size
        gz_files.append(gz)
        manifest_files.append({"name": gz.name, "path": str(gz), "size": sz, "sha256": h})

    # produce latest.json manifest
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    version = datetime.utcnow().strftime('v%Y%m%d%H%M%S')
    latest = {
        "version": version,
        "date": now_iso,
        "files": manifest_files,
        "keep": 5
    }
    latest_path = out_dir / 'latest.json'
    latest_path.write_text(json.dumps(latest, indent=2), encoding='utf-8')
    print("Wrote latest.json with version", version)

    print("All done. Generated gzipped files:")
    for m in manifest_files:
        print(" -", m['name'], m['size'], "bytes", m['sha256'])

    print("Latest manifest:", latest_path)
    print("Done.")

if __name__ == '__main__':
    main()
