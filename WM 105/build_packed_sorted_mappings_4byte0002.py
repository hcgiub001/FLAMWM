#!/usr/bin/env python3
"""
build_packed_sorted_mappings_4byte.py

Same as before but:
 - Force 4-byte keys and 4-byte values on-disk (no 1/2/3/8-byte choices)
 - If run without --csv, attempt to open a file picker (tkinter) for the CSV
 - Raise an error if any key/value exceeds 32-bit unsigned range
"""
from __future__ import annotations
import os
import json
import struct
import argparse
from pathlib import Path
from typing import List, Optional
import sys

import pandas as pd

# ---------------- constants ----------------

FORCED_KEY_BYTES = 4
FORCED_VALUE_BYTES = 4

# ---------------- utils ----------------

def pack_to_bytes_le(n: int, length: int) -> bytes:
    return int(n).to_bytes(length, byteorder='little', signed=False)

def pack_to_bytes_be(n: int, length: int) -> bytes:
    return int(n).to_bytes(length, byteorder='big', signed=False)

# ---------------- CSV parsing / normalization ----------------

def detect_columns(header_names):
    mapping = {}
    for name in header_names:
        nm = name.strip().lower()
        if nm in ("watchmode id", "watchmode_id", "watchmodeid", "watchmode"):
            mapping['watchmode_id'] = name
        elif nm in ("tmdb id", "tmdb_id", "tmdbid", "tmdb"):
            mapping['tmdb_id'] = name
        elif nm in ("tmdb type", "tmdb_type", "type", "tmdbtype"):
            mapping['tmdb_type'] = name
    return mapping

def normalize_tmdb_type_column(s: pd.Series) -> pd.Series:
    out = s.fillna('').astype(str).str.strip().str.lower()
    out = out.replace({
        'tv_series': 'tv', 'tvseries': 'tv', 'television': 'tv', 'series': 'tv',
        'tv_show': 'tv', 'tv show': 'tv',
        'film': 'movie', 'films': 'movie', 'movies': 'movie'
    })
    return out

def read_and_prepare_df(csv_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    print(f"Reading CSV: {csv_path}")
    sample = pd.read_csv(csv_path, nrows=0, encoding=encoding)
    header = list(sample.columns)
    col_map = detect_columns(header)
    required = ['watchmode_id', 'tmdb_id', 'tmdb_type']
    for r in required:
        if r not in col_map:
            raise RuntimeError(f"CSV header missing required column (need {r}). Detected columns: {header}")

    usecols = [col_map['watchmode_id'], col_map['tmdb_id'], col_map['tmdb_type']]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=str, encoding=encoding, quotechar='"', low_memory=False)
    df.columns = ['watchmode_id', 'tmdb_id', 'tmdb_type']

    df['tmdb_type'] = normalize_tmdb_type_column(df['tmdb_type'])

    # Convert numeric fields robustly
    df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce').astype('Int64')
    df['watchmode_id'] = pd.to_numeric(df['watchmode_id'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['tmdb_id', 'watchmode_id'])
    df['tmdb_id'] = df['tmdb_id'].astype(int)
    df['watchmode_id'] = df['watchmode_id'].astype(int)

    # Normalize type to only 'movie' or 'tv' (others dropped)
    df['tmdb_type_norm'] = df['tmdb_type'].where(df['tmdb_type'].isin(['movie', 'tv']), None)
    unknown = df[df['tmdb_type_norm'].isna()]
    if len(unknown) > 0:
        print(f"Warning: {len(unknown)} rows with unknown/unsupported tmdb_type were dropped (sample below):")
        print(unknown.head(10).to_string(index=False))
        df = df.dropna(subset=['tmdb_type_norm'])

    return df

# ---------------- file writer helpers ----------------

def make_sample_array(keys_int: List[int], sample_stride: int) -> bytes:
    out = bytearray()
    n = len(keys_int)
    for i in range(0, n, sample_stride):
        k = int(keys_int[i]) & 0xFFFFFFFF
        out += struct.pack('<I', k)
    return bytes(out)

def write_packed_pair(prefix: str,
                      keys_int: List[int],
                      values_int: List[int],
                      sample_stride: int,
                      key_name: str,
                      value_name: str):
    assert len(keys_int) == len(values_int), "keys/values length mismatch"
    n = len(keys_int)
    # ensure directory exists (handle prefix with no dirname)
    dirn = os.path.dirname(prefix) or '.'
    os.makedirs(dirn, exist_ok=True)

    # Force 4 bytes, but check ranges
    max_key = max(keys_int) if keys_int else 0
    max_val = max(values_int) if values_int else 0
    if max_key > 0xFFFFFFFF:
        raise RuntimeError(f"Key value exceeds 32-bit unsigned range (max_key={max_key}). Use 8-byte writer or reduce IDs.")
    if max_val > 0xFFFFFFFF:
        raise RuntimeError(f"Value exceeds 32-bit unsigned range (max_val={max_val}). Use 8-byte writer or reduce IDs.")

    key_bytes = FORCED_KEY_BYTES
    val_bytes = FORCED_VALUE_BYTES

    keys_path = f"{prefix}.{key_name}.keys.bin"
    vals_path = f"{prefix}.{value_name}.values.bin"
    sample_path = f"{prefix}.sample.u32"
    meta_path = f"{prefix}.meta.json"

    # keys big-endian, values little-endian (same semantics as before)
    with open(keys_path, 'wb') as kf:
        for k in keys_int:
            kf.write(pack_to_bytes_le(k, key_bytes))
    with open(vals_path, 'wb') as vf:
        for v in values_int:
            vf.write(pack_to_bytes_le(v, val_bytes))

    sample_bytes = make_sample_array(keys_int, sample_stride)
    with open(sample_path, 'wb') as sf:
        sf.write(sample_bytes)

    meta = {
        "count": n,
        "key_bytes": key_bytes,
        "value_bytes": val_bytes,
        "key_endian": "little",
        "value_endian": "little",
        "sample_stride": sample_stride,
        "key_name": key_name,
        "value_name": value_name,
        "files": {
            "keys": os.path.basename(keys_path),
            "values": os.path.basename(vals_path),
            "sample": os.path.basename(sample_path)
        }
    }
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, indent=2)

    ks = os.path.getsize(keys_path)
    vs = os.path.getsize(vals_path)
    ss = os.path.getsize(sample_path)
    print(f"Wrote: {keys_path} ({ks} bytes), {vals_path} ({vs} bytes), {sample_path} ({ss} bytes)")
    print(f"  meta -> {meta_path}")
    return keys_path, vals_path, sample_path, meta_path

# ---------------- high-level builders (fixed numeric ops) ----------------

def build_tmdb_to_watchmode(df: pd.DataFrame, out_dir: str, sample_stride: int):
    out_dir_p = Path(out_dir)
    for kind in ('movie', 'tv'):
        dsub = df[df['tmdb_type_norm'] == kind].copy()
        if dsub.empty:
            print(f"No rows for {kind}, skipping.")
            continue

        # convert to numpy arrays for safe bitwise shifts
        is_tv_bit = 1 if kind == 'tv' else 0
        tmdb_arr = dsub['tmdb_id'].to_numpy(dtype='int64')
        # packed = (tmdb << 1) | is_tv_bit
        packed_arr = (tmdb_arr.astype('uint64') << 1) | (is_tv_bit & 0x1)

        dsub = dsub.assign(packed_tmdb=packed_arr)
        dsub = dsub.sort_values(['packed_tmdb', 'watchmode_id']).reset_index(drop=True)

        keys = dsub['packed_tmdb'].astype(int).tolist()
        values = dsub['watchmode_id'].astype(int).tolist()

        prefix = str(out_dir_p / f"tmdb2wm.{kind}")
        print(f"Building tmdb->watchmode ({kind}): {len(keys)} entries")
        write_packed_pair(prefix, keys, values, sample_stride, key_name=f"tmdb2wm.{kind}", value_name=f"watchmode.{kind}")

def build_watchmode_to_tmdb(df: pd.DataFrame, out_dir: str, sample_stride: int):
    out_dir_p = Path(out_dir)
    # create numpy arrays for packing
    tmdb_arr = df['tmdb_id'].to_numpy(dtype='int64')
    is_tv_arr = (df['tmdb_type_norm'] == 'tv').to_numpy(dtype='int64')
    packed_arr = (tmdb_arr.astype('uint64') << 1) | (is_tv_arr.astype('uint64') & 0x1)

    df = df.assign(packed_tmdb=packed_arr)

    before = len(df)
    df = df.drop_duplicates(subset=['watchmode_id'], keep='first').reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before-after} duplicate watchmode_id rows (keep=first).")

    df = df.sort_values('watchmode_id').reset_index(drop=True)

    keys = df['watchmode_id'].astype(int).tolist()
    values = df['packed_tmdb'].astype(int).tolist()

    prefix = str(out_dir_p / "wm2tmdb")
    print(f"Building watchmode->tmdb: {len(keys)} entries")
    write_packed_pair(prefix, keys, values, sample_stride, key_name="wm2tmdb", value_name="packed_tmdb")

# ---------------- CLI + file picker helper ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Build packed sorted arrays + sample index for tmdb<->watchmode mappings (4-byte forced).")
    p.add_argument('--csv', '-c', help='Path to CSV file (optional). If omitted, a file picker will be opened when possible.', default=None)
    p.add_argument('--out', '-o', help='Output directory (defaults to CSV dir)', default=None)
    p.add_argument('--sample-stride', '-s', help='Sample stride S (default 64)', type=int, default=64)
    p.add_argument('--encoding', help='CSV encoding (default autodetect)', default=None)
    return p.parse_args()

def pick_file_with_tkinter(initialdir: Optional[str] = None) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    filename = filedialog.askopenfilename(title="Select CSV file", initialdir=initialdir or os.getcwd(),
                                          filetypes=[("CSV files","*.csv"),("All files","*.*")])
    root.destroy()
    return filename or None

def main():
    args = parse_args()
    csv_path = args.csv
    # If csv not provided, attempt file picker
    if not csv_path:
        picked = pick_file_with_tkinter()
        if picked:
            csv_path = picked
            print(f"Picked CSV: {csv_path}")
        else:
            print("No CSV provided and tkinter file picker unavailable or cancelled.")
            print("Please provide --csv <path> on the command line.")
            sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"CSV path does not exist: {csv_path}")
        sys.exit(1)

    out_dir = args.out if args.out else str(Path(csv_path).parent)
    sample_stride = int(args.sample_stride)

    df = read_and_prepare_df(csv_path, encoding=args.encoding)
    total = len(df)
    print(f"Total rows after cleanup: {total}")
    if total == 0:
        print("No records. Exiting.")
        return

    build_tmdb_to_watchmode(df, out_dir, sample_stride)
    build_watchmode_to_tmdb(df, out_dir, sample_stride)

    print("\nAll done. Files written into:", out_dir)

if __name__ == '__main__':
    main()
