#!/usr/bin/env python3
"""
build_packed_sorted_mappings_ef.py

Build Elias–Fano blobs + companion values.u32 arrays for:
 - tmdb2wm.movie.*
 - tmdb2wm.tv.*
 - wm2tmdb.*

This script emits ONLY the EF files (prefix.ef.low, prefix.ef.high, prefix.ef.meta.json,
prefix.ef.super.u32, prefix.ef.blocks.u16|u32, prefix.sample.u32) AND the companion
prefix.values.u32 (u32 LE) for each mapping. No legacy files are produced.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# ---------------- defaults / constants ----------------

DEFAULT_SAMPLE_STRIDE = 64
DEFAULT_EF_SUPERBLOCK_BITS = 2048
DEFAULT_EF_BLOCK_BITS = 64

# ---------------- helper utils ----------------

def pack_to_bytes_le(n: int, length: int) -> bytes:
    return int(n).to_bytes(length, byteorder='little', signed=False)

def u32_le_bytes(x: int) -> bytes:
    return struct.pack('<I', int(x) & 0xFFFFFFFF)

def u16_le_bytes(x: int) -> bytes:
    return struct.pack('<H', int(x) & 0xFFFF)

# ---------------- CSV parsing ----------------

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

# ---------------- sample array ----------------

def make_sample_array(keys_int: np.ndarray, sample_stride: int) -> bytes:
    out = bytearray()
    n = len(keys_int)
    for i in range(0, n, sample_stride):
        k = int(keys_int[i]) & 0xFFFFFFFF
        out += struct.pack('<I', k)
    return bytes(out)

# ---------------- Elias-Fano builder ----------------

def choose_ef_L(max_val: int, n: int) -> int:
    # choose number of low bits
    if n <= 0:
        return 0
    if max_val == 0:
        return 0
    ratio = max_val / n
    if ratio <= 1:
        return 0
    return int(math.floor(math.log2(ratio)))

def build_elias_fano(sorted_arr: np.ndarray, L: Optional[int] = None) -> Dict:
    """
    Build Elias-Fano components from sorted unsigned integer numpy array.
    Returns dict: n, L, low_bits_bytes, low_bits_len_bits, high_bits_bytes, high_bits_len_bits, h_last
    """
    n = int(sorted_arr.size)
    if n == 0:
        return {'n': 0, 'L': 0, 'low_bits_len_bits': 0, 'low_bits_bytes': b'', 'high_bits_len_bits': 0, 'high_bits_bytes': b'', 'h_last': 0}

    max_val = int(sorted_arr[-1])
    if L is None:
        L = choose_ef_L(max_val, n)
    L = int(L)

    if L == 0:
        lows = sorted_arr.astype(np.uint64)
        highs = sorted_arr.astype(np.uint64)
    else:
        lows = sorted_arr & np.uint64((1 << L) - 1)
        highs = sorted_arr >> np.uint64(L)

    # low bits packed L * n bits, little-endian within bytes
    total_low_bits = n * L
    low_bytes_len = (total_low_bits + 7) // 8
    low_buf = bytearray(low_bytes_len)
    if L > 0:
        bitpos = 0
        for v in lows.astype(np.uint64):
            val = int(v)
            # write L bits of val
            for b in range(L):
                if (val >> b) & 1:
                    byte_idx = bitpos >> 3
                    bit_idx = bitpos & 7
                    low_buf[byte_idx] |= (1 << bit_idx)
                bitpos += 1

    # high bits: set bit at position (high[i] + i)
    highs_int = highs.astype(np.uint64)
    h_last = int(highs_int[-1])
    high_len = h_last + n
    high_bytes_len = (high_len + 7) // 8
    high_buf = bytearray(high_bytes_len)
    for i in range(n):
        pos = int(highs_int[i]) + i
        byte_idx = pos >> 3
        bit_idx = pos & 7
        high_buf[byte_idx] |= (1 << bit_idx)

    return {
        'n': n,
        'L': L,
        'low_bits_len_bits': int(total_low_bits),
        'low_bits_bytes': bytes(low_buf),
        'high_bits_len_bits': int(high_len),
        'high_bits_bytes': bytes(high_buf),
        'h_last': int(h_last)
    }

# ---------------- EF aux index builder ----------------

def ef_aux_sizes_exact(high_len_bits: int, superblock_bits: int = DEFAULT_EF_SUPERBLOCK_BITS, block_bits: int = DEFAULT_EF_BLOCK_BITS) -> Dict:
    num_superblocks = math.ceil(high_len_bits / superblock_bits) if superblock_bits > 0 else 0
    superblock_bytes = num_superblocks * 4
    num_blocks = math.ceil(high_len_bits / block_bits)
    per_block_bytes = 2 if superblock_bits <= 65535 else 4
    block_bytes = num_blocks * per_block_bytes
    return {
        'num_superblocks': int(num_superblocks),
        'superblock_bytes': int(superblock_bytes),
        'num_blocks': int(num_blocks),
        'per_block_bytes': int(per_block_bytes),
        'block_bytes': int(block_bytes)
    }

def ef_select_build_indexes(high_bytes: bytes, high_bits_len: int, superblock_bits: int = DEFAULT_EF_SUPERBLOCK_BITS, block_bits: int = DEFAULT_EF_BLOCK_BITS):
    """
    Build super_ranks (u32 per superblock cumulative counts) and block_offsets (per-block rank offsets relative to superblock)
    Returns: (super_ranks: list[int], block_offsets: list[int], block_bits, superblock_bits)
    """
    total_bits = high_bits_len
    if total_bits == 0:
        return [0], [], block_bits, superblock_bits

    # convert to 64-bit words (little-endian chunks)
    num_words = (len(high_bytes) + 7) // 8
    words = [0] * num_words
    for i in range(num_words):
        start = i * 8
        chunk = high_bytes[start:start+8]
        words[i] = int.from_bytes(chunk, 'little')

    bits_per_word = 64
    blocks = math.ceil(total_bits / block_bits)
    superblocks = math.ceil(total_bits / superblock_bits)
    super_ranks = [0] * (superblocks + 1)
    block_offsets = [0] * blocks

    rank = 0
    current_block = 0
    next_block_bit = 0
    sb_index = 0
    sb_next_bit = (sb_index + 1) * superblock_bits

    bit_index = 0
    word_index = 0

    while bit_index < total_bits:
        w = words[word_index]
        bits_in_this_word = min(bits_per_word, total_bits - bit_index)
        mask = (1 << bits_in_this_word) - 1
        pop = w & mask
        popcount = pop.bit_count()
        prev_rank = rank
        rank += popcount

        while next_block_bit < bit_index + bits_in_this_word and current_block < blocks:
            pos = next_block_bit
            within = pos - bit_index
            if within <= 0:
                pop_until = 0
            else:
                mask2 = (1 << within) - 1
                pop_until = (w & mask2).bit_count()
            block_rank = prev_rank + pop_until
            block_offsets[current_block] = block_rank
            current_block += 1
            next_block_bit += block_bits

        if bit_index + bits_in_this_word >= sb_next_bit:
            super_ranks[sb_index + 1] = rank
            sb_index += 1
            sb_next_bit = (sb_index + 1) * superblock_bits

        bit_index += bits_in_this_word
        word_index += 1

    # convert block_offsets to offsets relative to superblock rank
    for b in range(blocks):
        block_start_bit = b * block_bits
        sb = block_start_bit // superblock_bits
        super_rank = super_ranks[sb]
        block_offsets[b] = int(block_offsets[b] - super_rank)

    # finalize
    super_ranks = [int(x) for x in super_ranks]
    return super_ranks, block_offsets, block_bits, superblock_bits

# ---------------- write helpers ----------------

def write_bytes_file(path: str, data: bytes):
    with open(path, 'wb') as f:
        f.write(data)

def write_u32_array_le(path: str, arr: List[int]):
    with open(path, 'wb') as f:
        for v in arr:
            f.write(u32_le_bytes(v))

def write_block_offsets_le(path: str, arr: List[int], per_block_bytes: int):
    with open(path, 'wb') as f:
        if per_block_bytes == 2:
            for v in arr:
                if v > 0xFFFF:
                    raise RuntimeError(f"block offset {v} doesn't fit in u16")
                f.write(u16_le_bytes(v))
        else:
            for v in arr:
                f.write(u32_le_bytes(v))

# ---------------- EF writer (writes EF blobs + companion values.u32) ----------------

def write_elias_fano_files(prefix: str, keys_sorted_np: np.ndarray, values_np: np.ndarray, sample_stride: int, ef_superblock_bits: int, ef_block_bits: int):
    """
    Given sorted keys (np.uint64 array) and values (np.uint64 array, one-per-key),
    build EF and write:
      - prefix.ef.low (bytes)
      - prefix.ef.high (bytes)
      - prefix.ef.meta.json (json metadata)
      - prefix.ef.super.u32 (u32 cumulative super ranks)
      - prefix.ef.blocks.u16|u32 (block offsets relative to super rank)
      - prefix.sample.u32 (u32 sample)
      - prefix.values.u32 (u32 LE array of corresponding values)
    Returns dict of file paths and sizes.
    """
    p = Path(prefix)
    out_dir = p.parent
    name = p.name

    os.makedirs(out_dir, exist_ok=True)

    n = int(keys_sorted_np.size)
    if n != int(values_np.size):
        raise RuntimeError("keys and values length mismatch")

    # build ef
    ef = build_elias_fano(keys_sorted_np)
    ef_aux = ef_aux_sizes_exact(ef['high_bits_len_bits'], superblock_bits=ef_superblock_bits, block_bits=ef_block_bits)
    per_block_bytes = ef_aux['per_block_bytes']

    # write low/high
    low_path = str(out_dir / f"{name}.ef.low")
    high_path = str(out_dir / f"{name}.ef.high")
    write_bytes_file(low_path, ef['low_bits_bytes'])
    write_bytes_file(high_path, ef['high_bits_bytes'])

    # build aux indexes
    super_ranks, block_offsets, _, _ = ef_select_build_indexes(ef['high_bits_bytes'], ef['high_bits_len_bits'], superblock_bits=ef_superblock_bits, block_bits=ef_block_bits)

    super_path = str(out_dir / f"{name}.ef.super.u32")
    blocks_path = str(out_dir / f"{name}.ef.blocks.u{per_block_bytes*8}")

    write_u32_array_le(super_path, super_ranks)
    write_block_offsets_le(blocks_path, block_offsets, per_block_bytes)

    # write sample array for hybrid usage (u32)
    sample_path = str(out_dir / f"{name}.sample.u32")
    sample_bytes = make_sample_array(keys_sorted_np, sample_stride)
    write_bytes_file(sample_path, sample_bytes)

    # write companion values as u32 (little-endian)
    values_path = str(out_dir / f"{name}.values.u32")
    # verify ranges
    if values_np.size > 0:
        max_val = int(values_np.max())
        if max_val > 0xFFFFFFFF:
            raise RuntimeError(f"Value exceeds 32-bit unsigned range (max_val={max_val}). Reduce IDs or use a different format.")
    write_u32_array_le(values_path, [int(x) & 0xFFFFFFFF for x in values_np.tolist()])

    meta = {
        "n": n,
        "L": int(ef['L']),
        "low_bits_len_bits": int(ef['low_bits_len_bits']),
        "low_bytes": len(ef['low_bits_bytes']),
        "high_bits_len_bits": int(ef['high_bits_len_bits']),
        "high_bytes": len(ef['high_bits_bytes']),
        "ef_aux": {
            "superblock_bits": int(ef_superblock_bits),
            "block_bits": int(ef_block_bits),
            "num_superblocks": ef_aux['num_superblocks'],
            "num_blocks": ef_aux['num_blocks'],
            "per_block_bytes": ef_aux['per_block_bytes']
        },
        "sample_stride": sample_stride,
        "files": {
            "low": os.path.basename(low_path),
            "high": os.path.basename(high_path),
            "super": os.path.basename(super_path),
            "blocks": os.path.basename(blocks_path),
            "sample": os.path.basename(sample_path),
            "values": os.path.basename(values_path)
        }
    }

    meta_path = str(out_dir / f"{name}.ef.meta.json")
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, indent=2)

    info = {
        'low_path': low_path,
        'high_path': high_path,
        'super_path': super_path,
        'blocks_path': blocks_path,
        'sample_path': sample_path,
        'values_path': values_path,
        'meta_path': meta_path,
        'low_bytes': len(ef['low_bits_bytes']),
        'high_bytes': len(ef['high_bits_bytes']),
        'super_blocks': len(super_ranks),
        'num_blocks': len(block_offsets),
        'values_bytes': os.path.getsize(values_path)
    }
    print(f"Wrote EF+values for {name}: low={info['low_bytes']} bytes, high={info['high_bytes']} bytes, values={human_bytes(info['values_bytes'])}, superblocks={info['super_blocks']}, blocks={info['num_blocks']}")
    print(f"  meta -> {meta_path}")
    return info

# ---------------- helpers ----------------

def human_bytes(n):
    if n < 1024: return f"{n} B"
    if n < 1024*1024: return f"{(n/1024):.1f} KB"
    return f"{(n/(1024*1024)):.2f} MB"

# ---------------- high-level mapping builders ----------------

def build_tmdb_to_watchmode(df: pd.DataFrame, out_dir: str, sample_stride: int, ef_superblock_bits: int, ef_block_bits: int):
    out_dir_p = Path(out_dir)
    for kind in ('movie', 'tv'):
        dsub = df[df['tmdb_type_norm'] == kind].copy()
        if dsub.empty:
            print(f"No rows for {kind}, skipping.")
            continue

        is_tv_bit = 1 if kind == 'tv' else 0
        tmdb_arr = dsub['tmdb_id'].to_numpy(dtype=np.uint64)
        packed_arr = (tmdb_arr.astype(np.uint64) << 1) | (is_tv_bit & 0x1)

        dsub = dsub.assign(packed_tmdb=packed_arr)
        dsub = dsub.sort_values(['packed_tmdb', 'watchmode_id']).reset_index(drop=True)

        keys_np = dsub['packed_tmdb'].astype(np.uint64).to_numpy()
        vals_np = dsub['watchmode_id'].astype(np.uint64).to_numpy()

        prefix = str(out_dir_p / f"tmdb2wm.{kind}")
        print(f"Building tmdb->watchmode ({kind}): {len(keys_np)} entries")

        # EF blobs + companion values.u32
        write_elias_fano_files(prefix, keys_np, vals_np, sample_stride, ef_superblock_bits, ef_block_bits)

def build_watchmode_to_tmdb(df: pd.DataFrame, out_dir: str, sample_stride: int, ef_superblock_bits: int, ef_block_bits: int):
    out_dir_p = Path(out_dir)
    tmdb_arr = df['tmdb_id'].to_numpy(dtype=np.uint64)
    is_tv_arr = (df['tmdb_type_norm'] == 'tv').to_numpy(dtype=np.uint64)
    packed_arr = (tmdb_arr.astype(np.uint64) << 1) | (is_tv_arr.astype(np.uint64) & 0x1)

    df = df.assign(packed_tmdb=packed_arr)

    before = len(df)
    df = df.drop_duplicates(subset=['watchmode_id'], keep='first').reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before-after} duplicate watchmode_id rows (keep=first).")

    df = df.sort_values('watchmode_id').reset_index(drop=True)

    keys_np = df['watchmode_id'].astype(np.uint64).to_numpy()
    vals_np = df['packed_tmdb'].astype(np.uint64).to_numpy()

    prefix = str(out_dir_p / "wm2tmdb")
    print(f"Building watchmode->tmdb: {len(keys_np)} entries")

    write_elias_fano_files(prefix, keys_np, vals_np, sample_stride, ef_superblock_bits, ef_block_bits)

# ---------------- CLI and main ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Build EF-packed mappings + companion values.u32 for tmdb<->watchmode.")
    p.add_argument('--csv', '-c', help='Path to CSV file (optional). If omitted, Tkinter file picker will be opened if available.', default=None)
    p.add_argument('--out', '-o', help='Output directory (defaults to CSV dir)', default=None)
    p.add_argument('--sample-stride', '-s', type=int, default=DEFAULT_SAMPLE_STRIDE, help='Sample stride S (default 64)')
    p.add_argument('--ef-superblock-bits', type=int, default=DEFAULT_EF_SUPERBLOCK_BITS, help='EF superblock size in bits (default 2048)')
    p.add_argument('--ef-block-bits', type=int, default=DEFAULT_EF_BLOCK_BITS, help='EF block size in bits (default 64)')
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
    if not csv_path:
        picked = pick_file_with_tkinter()
        if picked:
            csv_path = picked
            print(f"Picked CSV: {csv_path}")
        else:
            print("No CSV provided and tkinter file picker unavailable or cancelled.")
            print("Please provide --csv <path> on the command line.")
            sys.exit(1)

    csv_path = str(csv_path)
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

    # Build both mapping types (EF + values.u32)
    build_tmdb_to_watchmode(df, out_dir, sample_stride, args.ef_superblock_bits, args.ef_block_bits)
    build_watchmode_to_tmdb(df, out_dir, sample_stride, args.ef_superblock_bits, args.ef_block_bits)

    print("\nAll done. Files written into:", out_dir)

if __name__ == '__main__':
    main()
