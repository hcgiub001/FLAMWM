# -*- coding: utf-8 -*-
"""
watchmode_api.py — Fenlight adapter for Watchmode packed mapping files + API.

Resolves bidirectional mappings between TMDB IDs and Watchmode IDs using
compact binary files (keys.bin, values.bin, sample.u32, meta.json) downloaded
once and cached locally.  Also provides Watchmode API integration for details
and similar titles lookups.

Binary file format per mapping:
    keys.bin   — sorted uint32 keys (packed or plain IDs)
    values.bin — uint32 values (packed or plain IDs)
    sample.u32 — every Nth key for fast block narrowing
    meta.json  — metadata (key_bytes, value_bytes, sample_stride, endianness)

Packed TMDB ID format:  (tmdb_id << 1) | type_bit   (0=movie, 1=tv)

Loading modes: 'RAM' (full read), 'mmap' (on-demand pages), 'auto' (pick by avail mem).

Query results are cached in main_cache with a version token for cheap bulk invalidation.

Public API:
    ensure_loaded / reload_dataset / close_dataset
    get_setting_mode / set_setting_mode / get_runtime_mode
    set_api_key / get_api_key
    lookup_tmdb_to_watchmode / lookup_watchmode_to_tmdb
    resolve_watchmode_ids_batch
    search_watchmode_via_api / call_details_api / get_similar_titles_resolved
    clear_watchmode_cache / dataset_info
"""

from __future__ import annotations

import http.client
import json
import mmap
import os
import platform
import ctypes
import shutil
import ssl
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
from array import array
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Tuple

from caches.main_cache import main_cache
from caches.settings_cache import get_setting, set_setting
from modules.kodi_utils import translate_path, kodi_dialog, kodi_log  # noqa: F401

# =====================================================================
# Configuration
# =====================================================================
_BASE_URL = "https://hcgiub001.github.io/FLAMWM"

_MAPPING_SPECS = {
    "tmdb2wm_movie": {
        "keys":   "tmdb2wm.movie.tmdb2wm.movie.keys.bin",
        "values": "tmdb2wm.movie.watchmode.movie.values.bin",
        "sample": "tmdb2wm.movie.sample.u32",
        "meta":   "tmdb2wm.movie.meta.json",
    },
    "tmdb2wm_tv": {
        "keys":   "tmdb2wm.tv.tmdb2wm.tv.keys.bin",
        "values": "tmdb2wm.tv.watchmode.tv.values.bin",
        "sample": "tmdb2wm.tv.sample.u32",
        "meta":   "tmdb2wm.tv.meta.json",
    },
    "wm2tmdb": {
        "keys":   "wm2tmdb.wm2tmdb.keys.bin",
        "values": "wm2tmdb.packed_tmdb.values.bin",
        "sample": "wm2tmdb.sample.u32",
        "meta":   "wm2tmdb.meta.json",
    },
}

_ALL_FILES: List[str] = []
for _spec in _MAPPING_SPECS.values():
    for _f in (_spec["meta"], _spec["sample"], _spec["keys"], _spec["values"]):
        if _f not in _ALL_FILES:
            _ALL_FILES.append(_f)

_DEFAULT_CACHE_SUBDIR      = "watchmode"
_SETTING_KEY_MODE          = "fenlight.watchmode.mode"
_SETTING_KEY_API_KEY       = "fenlight.watchmode.api_key"
_CACHE_VERSION_KEY         = "watchmode:cache_version"
_DEFAULT_EXPIRATION_HOURS  = 24
_REQUEST_TIMEOUT_S         = 30

# Auto-mode memory thresholds
_AUTO_THRESHOLD_DEFAULT = 200 * 1024 * 1024
_SAFETY_MARGIN_MIN      = 64  * 1024 * 1024
_SAFETY_MARGIN_FRAC     = 0.05

# =====================================================================
# Memory detection (multi-platform)
# =====================================================================
def _parse_kodi_memory_mb(label_value: str) -> int:
    try:
        s = label_value.strip().upper()
        multiplier = 1
        if "GB" in s:
            multiplier = 1024
            s = s.replace("GB", "")
        elif "MB" in s:
            s = s.replace("MB", "")
        elif "KB" in s:
            multiplier = 1.0 / 1024.0
            s = s.replace("KB", "")
        return int(float(s.strip()) * multiplier)
    except Exception:
        return 0


def _get_mem_via_kodi() -> Optional[Tuple[int, int, str]]:
    try:
        import xbmc
    except ImportError:
        return None
    try:
        total_str = xbmc.getInfoLabel("System.Memory(total)")
        free_str  = (xbmc.getInfoLabel("System.Memory(available)")
                     or xbmc.getInfoLabel("System.Memory(free)")
                     or xbmc.getInfoLabel("System.FreeMemory"))
        if not total_str or not free_str:
            return None
        total_mb = _parse_kodi_memory_mb(total_str)
        free_mb  = _parse_kodi_memory_mb(free_str)
        if total_mb <= 0 or free_mb <= 0:
            return None
        return (free_mb * 1024 * 1024, total_mb * 1024 * 1024, "kodi/InfoLabel")
    except Exception:
        return None


def _get_mem_via_psutil() -> Optional[Tuple[int, int, str]]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total), "psutil"
    except Exception:
        return None


def _get_mem_via_proc_meminfo() -> Optional[Tuple[int, int, str]]:
    try:
        if not os.path.exists("/proc/meminfo"):
            return None
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                try:
                    info[key] = int(val)
                except Exception:
                    pass
        if "MemAvailable" in info:
            avail = info["MemAvailable"] * 1024
        else:
            avail = int((info.get("MemFree", 0) + info.get("Cached", 0)
                         + info.get("Buffers", 0)) * 1024 * 0.7)
        total = info.get("MemTotal", 0) * 1024
        return int(avail), int(total), "/proc/meminfo"
    except Exception:
        return None


def _get_mem_via_windows() -> Optional[Tuple[int, int, str]]:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",              ctypes.c_ulong),
                ("dwMemoryLoad",          ctypes.c_ulong),
                ("ullTotalPhys",          ctypes.c_ulonglong),
                ("ullAvailPhys",          ctypes.c_ulonglong),
                ("ullTotalPageFile",      ctypes.c_ulonglong),
                ("ullAvailPageFile",      ctypes.c_ulonglong),
                ("ullTotalVirtual",       ctypes.c_ulonglong),
                ("ullAvailVirtual",       ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullAvailPhys), int(stat.ullTotalPhys), "GlobalMemoryStatusEx"
        return None
    except Exception:
        return None


def _get_available_memory() -> Tuple[int, int, str]:
    for fn in (_get_mem_via_kodi, _get_mem_via_psutil):
        r = fn()
        if r:
            return r
    sysname = platform.system().lower()
    if sysname in ("linux", "android"):
        r = _get_mem_via_proc_meminfo()
        if r:
            return r
    elif sysname == "windows":
        r = _get_mem_via_windows()
        if r:
            return r
    return 0, 0, "unknown"


def _compute_safety_margin(total_bytes: int) -> int:
    return max(_SAFETY_MARGIN_MIN, int(total_bytes * _SAFETY_MARGIN_FRAC))


# =====================================================================
# Packing helpers
# =====================================================================
def _packed_tmdb_value(tmdb_id: int, kind: str) -> int:
    return (int(tmdb_id) << 1) | (1 if str(kind).lower().startswith("tv") else 0)


def _unpack_tmdb_value(packed: int) -> Tuple[int, str]:
    return (packed >> 1), ("tv" if (packed & 1) else "movie")


# =====================================================================
# Atomic file write
# =====================================================================
def _atomic_write_temp(target_path: str, data_stream) -> None:
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".tmp_wm", suffix=".bin")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(data_stream, out_f)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _atomic_write_bytes(target_path: str, data: bytes) -> None:
    import io
    _atomic_write_temp(target_path, io.BytesIO(data))


# =====================================================================
# HTTP Connection Pool (keep-alive, TLS session reuse)
# =====================================================================
class _ConnectionPool:
    """Thread-safe per-host HTTP(S) connection pool with keep-alive.

    Eliminates repeated TCP + TLS handshakes when making sequential
    requests to the same host (~100-200 ms savings per reused call),
    matching the browser keep-alive behaviour that Tampermonkey gets
    for free via GM_xmlhttpRequest.

    Connections are keyed by (scheme, host, port).  A connection is
    *taken* from the pool before use and *returned* only if the
    server allows keep-alive — so two concurrent requests to the
    same host each get their own socket (no data races).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conns: Dict[Tuple[str, str, int], http.client.HTTPConnection] = {}

    # ---- public -------------------------------------------------

    def fetch(self, url: str, timeout: int = _REQUEST_TIMEOUT_S) -> bytes:
        """GET *url*, follow redirects, return body bytes."""
        current_url = url
        for _ in range(6):                       # 1 original + up to 5 redirects
            status, body, location = self._single_get(current_url, timeout)
            if 300 <= status < 400 and location:
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            return body
        raise IOError(f"Too many redirects for {url}")

    def close_all(self) -> None:
        """Close every pooled connection."""
        with self._lock:
            for c in self._conns.values():
                self._quiet_close(c)
            self._conns.clear()

    # ---- internals ----------------------------------------------

    def _single_get(
        self, url: str, timeout: int
    ) -> Tuple[int, bytes, Optional[str]]:
        """One GET (no redirect following).

        Returns (status, body, location_header_or_None).
        """
        parsed = urllib.parse.urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        host   = parsed.hostname or ""
        port   = parsed.port or (443 if scheme == "https" else 80)
        path   = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        key = (scheme, host, port)
        hdrs = {
            "Host":       host,
            "User-Agent": "watchmode_api/1.0",
            "Accept":     "*/*",
            "Connection": "keep-alive",
        }

        # 1) try the pooled connection first
        with self._lock:
            conn = self._conns.pop(key, None)

        if conn is not None:
            try:
                conn.request("GET", path, headers=hdrs)
                resp = conn.getresponse()
                body = resp.read()
                self._maybe_return(key, conn, resp)
                return resp.status, body, resp.getheader("Location")
            except Exception:
                self._quiet_close(conn)
                # fall through → fresh connection

        # 2) fresh connection
        conn = self._make_conn(scheme, host, port, timeout)
        try:
            conn.request("GET", path, headers=hdrs)
            resp = conn.getresponse()
            body = resp.read()
            self._maybe_return(key, conn, resp)
            return resp.status, body, resp.getheader("Location")
        except Exception:
            self._quiet_close(conn)
            raise

    def _maybe_return(
        self,
        key: Tuple[str, str, int],
        conn: http.client.HTTPConnection,
        resp: http.client.HTTPResponse,
    ) -> None:
        """Put *conn* back in the pool unless the server said 'close'."""
        conn_hdr = (resp.getheader("Connection") or "").lower()
        if conn_hdr == "close" or getattr(resp, "will_close", False):
            self._quiet_close(conn)
            return
        with self._lock:
            old = self._conns.pop(key, None)
            self._conns[key] = conn
        if old is not None and old is not conn:
            self._quiet_close(old)

    @staticmethod
    def _make_conn(
        scheme: str, host: str, port: int, timeout: int
    ) -> http.client.HTTPConnection:
        if scheme == "https":
            ctx = ssl.create_default_context()
            return http.client.HTTPSConnection(
                host, port, timeout=timeout, context=ctx
            )
        return http.client.HTTPConnection(host, port, timeout=timeout)

    @staticmethod
    def _quiet_close(conn: http.client.HTTPConnection) -> None:
        try:
            conn.close()
        except Exception:
            pass


# module-level singleton
_http_pool = _ConnectionPool()

# =====================================================================
# Network helpers (use the pool)
# =====================================================================
def _fetch_url_bytes(url: str, timeout: int = _REQUEST_TIMEOUT_S) -> bytes:
    """Download URL and return raw bytes.  Connection is kept alive."""
    return _http_pool.fetch(url, timeout=timeout)


def _fetch_url_json(url: str, timeout: int = _REQUEST_TIMEOUT_S) -> Any:
    """Download URL, parse JSON, return parsed object."""
    data = _fetch_url_bytes(url, timeout=timeout)
    return json.loads(data.decode("utf-8"))


# =====================================================================
# Path helpers
# =====================================================================
def _ensure_cache_dir() -> str:
    folder = translate_path("special://profile/addon_data/plugin.video.fenlight/")
    data_folder = os.path.join(folder, "cache", _DEFAULT_CACHE_SUBDIR)
    try:
        os.makedirs(data_folder, exist_ok=True)
    except Exception:
        pass
    return data_folder


def _file_cache_path(filename: str) -> str:
    return os.path.join(_ensure_cache_dir(), filename)


# =====================================================================
# Ensure all binary files are downloaded
# =====================================================================
def _ensure_all_files_downloaded() -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for fname in _ALL_FILES:
        path = _file_cache_path(fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            results[fname] = False
            continue
        url = f"{_BASE_URL}/{fname}"
        try:
            data = _fetch_url_bytes(url)
            _atomic_write_bytes(path, data)
            results[fname] = True
        except Exception as e:
            kodi_log(f"watchmode_api: failed to download {fname}: {e}", 1)
            results[fname] = False
    return results


# =====================================================================
# Binary mapping parser
# =====================================================================
class PackedMapping:
    __slots__ = (
        "_keys32", "_vals32", "_sample32", "_meta",
        "_count", "_sample_stride", "_direct_map",
        "_keys_size", "_vals_size", "_sample_size",
        "_fileobjs", "_mmaps", "_is_mmap",
    )

    def __init__(self):
        self._keys32: Optional[array] = None
        self._vals32: Optional[array] = None
        self._sample32: Optional[array] = None
        self._meta: Optional[Dict[str, Any]] = None
        self._count: int = 0
        self._sample_stride: int = 64
        self._direct_map: Optional[array] = None
        self._keys_size: int = 0
        self._vals_size: int = 0
        self._sample_size: int = 0
        self._fileobjs: List[Any] = []
        self._mmaps: List[Any] = []
        self._is_mmap: bool = False

    @property
    def loaded(self) -> bool:
        return self._keys32 is not None and self._vals32 is not None

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_mmap(self) -> bool:
        return self._is_mmap

    @property
    def sizes(self) -> Dict[str, int]:
        return {
            "keys_bytes": self._keys_size,
            "values_bytes": self._vals_size,
            "sample_bytes": self._sample_size,
        }

    def close(self) -> None:
        for mm in self._mmaps:
            try:
                mm.close()
            except Exception:
                pass
        self._mmaps.clear()
        for fo in self._fileobjs:
            try:
                fo.close()
            except Exception:
                pass
        self._fileobjs.clear()
        self._keys32 = None
        self._vals32 = None
        self._sample32 = None
        self._direct_map = None
        self._meta = None
        self._count = 0
        self._is_mmap = False

    def load_ram(self, spec: Dict[str, str]) -> None:
        self.close()
        self._is_mmap = False

        meta_path = _file_cache_path(spec["meta"])
        with open(meta_path, "r", encoding="utf-8") as fh:
            self._meta = json.load(fh)

        key_bytes = int(self._meta.get("key_bytes", 4))
        val_bytes = int(self._meta.get("value_bytes", 4))
        self._sample_stride = int(self._meta.get("sample_stride", 64))

        sample_path = _file_cache_path(spec["sample"])
        sample_data = self._read_file_bytes(sample_path)
        self._sample_size = len(sample_data)
        sample_count = len(sample_data) // 4
        self._sample32 = array("I")
        self._sample32.frombytes(sample_data[:sample_count * 4])
        if sys.byteorder != "little":
            self._sample32.byteswap()

        keys_path = _file_cache_path(spec["keys"])
        keys_data = self._read_file_bytes(keys_path)
        self._keys_size = len(keys_data)
        count = len(keys_data) // key_bytes
        self._count = count

        key_is_big = self._meta.get("key_endian", "").lower() == "big"
        if key_bytes == 4:
            self._keys32 = array("I")
            self._keys32.frombytes(keys_data[:count * 4])
            if key_is_big:
                self._keys32.byteswap()
            elif sys.byteorder != "little":
                self._keys32.byteswap()
        else:
            raise ValueError(f"Unsupported key_bytes={key_bytes}")

        vals_path = _file_cache_path(spec["values"])
        vals_data = self._read_file_bytes(vals_path)
        self._vals_size = len(vals_data)

        val_is_big = self._meta.get("value_endian", "").lower() == "big"
        if val_bytes == 4:
            val_count = len(vals_data) // 4
            self._vals32 = array("I")
            self._vals32.frombytes(vals_data[:val_count * 4])
            if val_is_big:
                self._vals32.byteswap()
            elif sys.byteorder != "little":
                self._vals32.byteswap()
        else:
            raise ValueError(f"Unsupported value_bytes={val_bytes}")

        # RAM mode: build direct-map for O(1) block lookup
        self._build_direct_map()

    def load_mmap(self, spec: Dict[str, str]) -> None:
        self.close()
        self._is_mmap = True

        meta_path = _file_cache_path(spec["meta"])
        with open(meta_path, "r", encoding="utf-8") as fh:
            self._meta = json.load(fh)

        key_bytes = int(self._meta.get("key_bytes", 4))
        val_bytes = int(self._meta.get("value_bytes", 4))
        self._sample_stride = int(self._meta.get("sample_stride", 64))

        # sample.u32 is small — always load fully into RAM
        sample_path = _file_cache_path(spec["sample"])
        sample_data = self._read_file_bytes(sample_path)
        self._sample_size = len(sample_data)
        sample_count = len(sample_data) // 4
        self._sample32 = array("I")
        self._sample32.frombytes(sample_data[:sample_count * 4])
        if sys.byteorder != "little":
            self._sample32.byteswap()

        key_is_big = self._meta.get("key_endian", "").lower() == "big"
        val_is_big = self._meta.get("value_endian", "").lower() == "big"

        need_swap_keys = key_is_big or (sys.byteorder != "little")
        need_swap_vals = val_is_big or (sys.byteorder != "little")

        # ---- keys: mmap if no byte-swap needed, else fall back to RAM ----
        keys_path = _file_cache_path(spec["keys"])
        self._keys_size = os.path.getsize(keys_path)
        count = self._keys_size // key_bytes
        self._count = count

        # FIX: On little-endian systems we assume the binary files are native
        # (little-endian) regardless of what the metadata claims. This prevents
        # the entire keys file from being pulled into RAM just because the
        # JSON incorrectly labelled it as big-endian.
        if key_bytes == 4 and (sys.byteorder == "little" or not need_swap_keys):
            fk = open(keys_path, "rb")
            self._fileobjs.append(fk)
            mmk = mmap.mmap(fk.fileno(), length=0, access=mmap.ACCESS_READ)
            self._mmaps.append(mmk)
            mv = memoryview(mmk)
            self._keys32 = mv[:count * 4].cast("I")
        else:
            keys_data = self._read_file_bytes(keys_path)
            self._keys32 = array("I")
            self._keys32.frombytes(keys_data[:count * 4])
            if key_is_big:
                self._keys32.byteswap()
            elif sys.byteorder != "little":
                self._keys32.byteswap()

        # ---- values: mmap if no byte-swap needed, else fall back to RAM ----
        vals_path = _file_cache_path(spec["values"])
        self._vals_size = os.path.getsize(vals_path)
        val_count = self._vals_size // val_bytes

        if val_bytes == 4 and not need_swap_vals:
            fv = open(vals_path, "rb")
            self._fileobjs.append(fv)
            mmv = mmap.mmap(fv.fileno(), length=0, access=mmap.ACCESS_READ)
            self._mmaps.append(mmv)
            mv = memoryview(mmv)
            self._vals32 = mv[:val_count * 4].cast("I")
        else:
            vals_data = self._read_file_bytes(vals_path)
            self._vals32 = array("I")
            self._vals32.frombytes(vals_data[:val_count * 4])
            if val_is_big:
                self._vals32.byteswap()
            elif sys.byteorder != "little":
                self._vals32.byteswap()

        # *** DO NOT build _direct_map in mmap mode ***
        # _build_direct_map iterates every key (keys[i] >> 16) which would
        # page-fault the entire mmap'd keys file into physical RAM,
        # completely defeating the purpose of mmap.  The sample-based
        # binary search is used instead — it only touches the small
        # sample.u32 (already in RAM) plus one narrow block of keys
        # per lookup.
        self._direct_map = None

    def _build_direct_map(self) -> None:
        """Build a 64 K bucket index over the top 16 bits of each key.

        Only called in RAM mode where all keys are already resident.
        """
        keys = self._keys32
        if keys is None or len(keys) == 0:
            self._direct_map = None
            return

        count = len(keys)
        dm = array("I", [count] * 65537)

        for i in range(count):
            hb = keys[i] >> 16
            if i < dm[hb]:
                dm[hb] = i

        for hb in range(65535, -1, -1):
            if dm[hb] > dm[hb + 1]:
                dm[hb] = dm[hb + 1]

        dm[65536] = count
        self._direct_map = dm

    @staticmethod
    def _read_file_bytes(path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    def _upper_bound_sample(self, x: int) -> int:
        sample = self._sample32
        if sample is None:
            return 0
        lo, hi = 0, len(sample)
        while lo < hi:
            mid = (lo + hi) >> 1
            if sample[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _binary_search_block(self, key: int, left: int, right: int) -> int:
        keys = self._keys32
        lo, hi = left, right
        while lo <= hi:
            mid = (lo + hi) >> 1
            k = keys[mid]
            if k == key:
                return mid
            if k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def lookup(self, key: int) -> Optional[int]:
        if not self.loaded or self._count == 0:
            return None

        keys = self._keys32
        vals = self._vals32

        # Fast path: direct-map (RAM mode only)
        if self._direct_map is not None:
            dm = self._direct_map
            hb = key >> 16
            left = dm[hb]
            right_excl = dm[hb + 1]
            if left >= right_excl:
                return None
            pos = self._binary_search_block(key, left, right_excl - 1)
            if pos >= 0:
                return vals[pos]
            return None

        # Fallback: sample-based block narrowing (mmap-friendly)
        sb = self._upper_bound_sample(key) - 1
        stride = self._sample_stride
        left = max(0, sb * stride) if sb >= 0 else 0
        right = min(self._count - 1, max(left, ((sb + 1) * stride) - 1))
        pos = self._binary_search_block(key, left, right)
        if pos >= 0:
            return vals[pos]
        return None

    def lookup_range(self, key: int) -> Tuple[int, int]:
        if self._direct_map is not None:
            hb = key >> 16
            return (self._direct_map[hb], self._direct_map[hb + 1])
        return (0, self._count)


# =====================================================================
# Module state (thread-safe)
# =====================================================================
_lock = threading.RLock()
_mappings: Dict[str, Optional[PackedMapping]] = {
    "tmdb2wm_movie": None,
    "tmdb2wm_tv":    None,
    "wm2tmdb":       None,
}
_load_meta: Dict[str, Any] = {}
_runtime_mode: Optional[str] = None
_dataset_id: Optional[str] = None


# =====================================================================
# Cache version management
# =====================================================================
def _get_cache_version() -> str:
    v = main_cache.get(_CACHE_VERSION_KEY)
    if v is None:
        v = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, v, expiration=0)
        except Exception:
            pass
    return str(v)


def clear_watchmode_cache() -> None:
    with _lock:
        newv = str(int(time.time()))
        try:
            main_cache.set(_CACHE_VERSION_KEY, newv, expiration=0)
        except Exception:
            try:
                main_cache.delete(_CACHE_VERSION_KEY)
            except Exception:
                pass


# =====================================================================
# Dataset identity
# =====================================================================
def _compute_dataset_id() -> str:
    parts = []
    for fname in _ALL_FILES:
        path = _file_cache_path(fname)
        try:
            st = os.stat(path)
            parts.append(f"{fname}:{int(st.st_mtime)}:{int(st.st_size)}")
        except Exception:
            parts.append(f"{fname}:missing")
    return "|".join(parts)


# =====================================================================
# User settings
# =====================================================================
def get_setting_mode() -> str:
    v = get_setting(_SETTING_KEY_MODE, "auto")
    v = (v or "auto").strip()
    if v.lower() == "ram":
        return "RAM"
    if v.lower() == "mmap":
        return "mmap"
    return "auto"


def set_setting_mode(mode: str) -> bool:
    m = (mode or "auto").strip()
    if m.lower() not in ("auto", "ram", "mmap"):
        return False
    store = "RAM" if m.lower() == "ram" else ("mmap" if m.lower() == "mmap" else "auto")
    try:
        set_setting(_SETTING_KEY_MODE, store)
        return True
    except Exception:
        return False


def get_runtime_mode() -> Optional[str]:
    return _runtime_mode


def set_api_key(key: str) -> bool:
    try:
        set_setting(_SETTING_KEY_API_KEY, key.strip())
        return True
    except Exception:
        return False


def get_api_key() -> str:
    try:
        return get_setting(_SETTING_KEY_API_KEY, "")
    except Exception:
        return ""


# =====================================================================
# Lifecycle (load / reload / close)
# =====================================================================
def ensure_loaded(url: Optional[str] = None, mode: Optional[str] = None, force: bool = False) -> None:
    global _runtime_mode, _dataset_id, _load_meta

    with _lock:
        all_loaded = all(m is not None and m.loaded for m in _mappings.values())
        if all_loaded and not force:
            return

        download_results = _ensure_all_files_downloaded()

        mode_setting = get_setting_mode() if mode is None else mode
        chosen_mode = mode_setting

        if chosen_mode == "auto":
            total_size = 0
            for fname in _ALL_FILES:
                path = _file_cache_path(fname)
                try:
                    total_size += os.path.getsize(path)
                except Exception:
                    pass

            avail, total, _src = _get_available_memory()
            safety = _compute_safety_margin(total) if total else _SAFETY_MARGIN_MIN
            dataset_need = total_size + safety
            if avail and avail >= _AUTO_THRESHOLD_DEFAULT and avail >= dataset_need:
                chosen_mode = "RAM"
            else:
                chosen_mode = "mmap"

        for name, spec in _MAPPING_SPECS.items():
            mapping = _mappings.get(name)
            if mapping is not None:
                mapping.close()

            mapping = PackedMapping()
            if chosen_mode == "RAM":
                mapping.load_ram(spec)
            else:
                mapping.load_mmap(spec)
            _mappings[name] = mapping

        _runtime_mode = chosen_mode
        _dataset_id = _compute_dataset_id()

        meta: Dict[str, Any] = {
            "mode_chosen": chosen_mode,
            "files_downloaded": download_results,
        }
        for name, mapping in _mappings.items():
            if mapping is not None and mapping.loaded:
                meta[name] = {
                    "count": mapping.count,
                    "sizes": mapping.sizes,
                }
        _load_meta = meta


def reload_dataset(url: Optional[str] = None, mode: Optional[str] = None) -> None:
    global _runtime_mode, _dataset_id, _load_meta

    with _lock:
        for name, mapping in _mappings.items():
            if mapping is not None:
                try:
                    mapping.close()
                except Exception:
                    pass
            _mappings[name] = None
        _runtime_mode = None
        _dataset_id = None
        _load_meta = {}
        clear_watchmode_cache()
        ensure_loaded(url=url, mode=mode, force=True)


def close_dataset() -> None:
    """Release all resources including pooled HTTP connections."""
    global _runtime_mode, _dataset_id, _load_meta

    with _lock:
        for name, mapping in _mappings.items():
            if mapping is not None:
                try:
                    mapping.close()
                except Exception:
                    pass
            _mappings[name] = None
        _runtime_mode = None
        _dataset_id = None
        _load_meta = {}

    _http_pool.close_all()


# =====================================================================
# Query functions — local lookups
# =====================================================================
def _query_cache_key(prefix: str, key_id: int, kind: str = "") -> str:
    v = _get_cache_version()
    did = _dataset_id or "nodata"
    return f"wm:{prefix}:{v}:{did}:{key_id}:{kind}"


def lookup_tmdb_to_watchmode(
    tmdb_id: int,
    kind: str,
    use_cache: bool = True,
    expiration_hours: int = _DEFAULT_EXPIRATION_HOURS,
) -> Optional[int]:
    if _mappings.get("tmdb2wm_movie") is None:
        ensure_loaded()

    cache_key = _query_cache_key("t2w", tmdb_id, kind)
    if use_cache:
        try:
            cached = main_cache.get(cache_key)
            if cached is not None:
                return cached if cached != -1 else None
        except Exception:
            pass

    packed_key = _packed_tmdb_value(tmdb_id, kind)
    mapping_name = "tmdb2wm_movie" if kind.lower().startswith("movie") else "tmdb2wm_tv"
    mapping = _mappings.get(mapping_name)
    if mapping is None or not mapping.loaded:
        return None

    result = mapping.lookup(packed_key)

    try:
        main_cache.set(cache_key, result if result is not None else -1,
                       expiration=expiration_hours)
    except Exception:
        pass

    return result


def lookup_watchmode_to_tmdb(
    watchmode_id: int,
    use_cache: bool = True,
    expiration_hours: int = _DEFAULT_EXPIRATION_HOURS,
) -> Optional[Dict[str, Any]]:
    if _mappings.get("wm2tmdb") is None:
        ensure_loaded()

    cache_key = _query_cache_key("w2t", watchmode_id)
    if use_cache:
        try:
            cached = main_cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, dict):
                    return cached
                return None
        except Exception:
            pass

    mapping = _mappings.get("wm2tmdb")
    if mapping is None or not mapping.loaded:
        return None

    packed = mapping.lookup(watchmode_id)
    if packed is None:
        try:
            main_cache.set(cache_key, -1, expiration=expiration_hours)
        except Exception:
            pass
        return None

    tmdb_id, media_type = _unpack_tmdb_value(packed)
    result = {"tmdb_id": tmdb_id, "media_type": media_type}

    try:
        main_cache.set(cache_key, result, expiration=expiration_hours)
    except Exception:
        pass

    return result


def resolve_watchmode_ids_batch(
    watchmode_ids: List[int],
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    if _mappings.get("wm2tmdb") is None:
        ensure_loaded()

    if not watchmode_ids:
        return []

    mapping = _mappings.get("wm2tmdb")
    if mapping is None or not mapping.loaded:
        return [{"watchmode_id": wid, "tmdb_id": None, "media_type": None, "found": False}
                for wid in watchmode_ids]

    orig = [int(x) for x in watchmode_ids]

    seen = set()
    unique = []
    for v in orig:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    unique_sorted = sorted(unique)
    results_map: Dict[int, Dict[str, Any]] = {}

    keys = mapping._keys32
    vals = mapping._vals32
    if keys is not None and vals is not None and len(unique_sorted) > 0:
        stride = mapping._sample_stride

        # ---- choose batch strategy ----
        # The merge-scan (linear walk from index 0) is fast when data
        # is fully resident in RAM, but in mmap mode it would page-fault
        # the entire keys file into physical memory.  Skip it for mmap.
        use_merge = False
        if not mapping.is_mmap and mapping._direct_map is None:
            first_sb = max(0, mapping._upper_bound_sample(unique_sorted[0]) - 1)
            last_sb  = max(0, mapping._upper_bound_sample(unique_sorted[-1]) - 1)
            blocks_spanned = max(1, last_sb - first_sb + 1)
            est_scanned = blocks_spanned * stride
            merge_threshold = max(256, len(unique_sorted) * stride * 3)
            if est_scanned <= merge_threshold:
                use_merge = True

        if use_merge:
            qi = 0
            ki = 0
            nq = len(unique_sorted)
            nk = len(keys)
            while qi < nq and ki < nk:
                q = unique_sorted[qi]
                k = keys[ki]
                if k < q:
                    ki += 1
                elif k == q:
                    packed = vals[ki]
                    tid, mtype = _unpack_tmdb_value(packed)
                    results_map[q] = {"tmdb_id": tid, "media_type": mtype, "found": True}
                    qi += 1
                    ki += 1
                else:
                    results_map[q] = {"tmdb_id": None, "media_type": None, "found": False}
                    qi += 1
            while qi < nq:
                results_map[unique_sorted[qi]] = {"tmdb_id": None, "media_type": None, "found": False}
                qi += 1
        else:
            for wid in unique_sorted:
                packed = mapping.lookup(wid)
                if packed is not None:
                    tid, mtype = _unpack_tmdb_value(packed)
                    results_map[wid] = {"tmdb_id": tid, "media_type": mtype, "found": True}
                else:
                    results_map[wid] = {"tmdb_id": None, "media_type": None, "found": False}
    else:
        for wid in unique_sorted:
            packed = mapping.lookup(wid)
            if packed is not None:
                tid, mtype = _unpack_tmdb_value(packed)
                results_map[wid] = {"tmdb_id": tid, "media_type": mtype, "found": True}
            else:
                results_map[wid] = {"tmdb_id": None, "media_type": None, "found": False}

    output = []
    for wid in orig:
        r = results_map.get(wid, {"tmdb_id": None, "media_type": None, "found": False})
        output.append({
            "watchmode_id": wid,
            "tmdb_id":      r["tmdb_id"],
            "media_type":   r["media_type"],
            "found":        r["found"],
        })

    return output


# =====================================================================
# Watchmode API functions
# =====================================================================
def search_watchmode_via_api(
    tmdb_id: int,
    kind: str,
    api_key: Optional[str] = None,
) -> Optional[int]:
    key = api_key or get_api_key()
    if not key:
        raise ValueError("No Watchmode API key set")

    field = "tmdb_series_id" if kind.lower().startswith("tv") else "tmdb_movie_id"
    url = (
        f"https://api.watchmode.com/v1/search/"
        f"?apiKey={urllib.parse.quote(key)}"
        f"&search_field={urllib.parse.quote(field)}"
        f"&search_value={urllib.parse.quote(str(tmdb_id))}"
    )

    data = _fetch_url_json(url)
    if (isinstance(data, dict) and
            isinstance(data.get("title_results"), list) and
            len(data["title_results"]) > 0):
        return int(data["title_results"][0]["id"])
    return None


def call_details_api(
    watchmode_id: int,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    key = api_key or get_api_key()
    if not key:
        raise ValueError("No Watchmode API key set")

    url = (
        f"https://api.watchmode.com/v1/title/{int(watchmode_id)}/details/"
        f"?apiKey={urllib.parse.quote(key)}"
    )

    data = _fetch_url_json(url)
    return data if isinstance(data, dict) else None


def get_similar_titles_resolved(
    tmdb_id: int,
    kind: str,
    api_key: Optional[str] = None,
    use_local_first: bool = True,
    timing: bool = False,
) -> Any:
    t0 = time.perf_counter()
    key = api_key or get_api_key()

    wm_id = None
    if use_local_first:
        wm_id = lookup_tmdb_to_watchmode(tmdb_id, kind)

    if wm_id is None and key:
        try:
            wm_id = search_watchmode_via_api(tmdb_id, kind, api_key=key)
        except Exception:
            pass

    if wm_id is None:
        out = {"watchmode_id": None, "similar_count": 0, "results": [],
               "error": "Could not resolve TMDB ID to Watchmode ID"}
        total_ms = (time.perf_counter() - t0) * 1000.0
        return (out, total_ms) if timing else out

    if not key:
        out = {"watchmode_id": wm_id, "similar_count": 0, "results": [],
               "error": "No API key — cannot fetch details"}
        total_ms = (time.perf_counter() - t0) * 1000.0
        return (out, total_ms) if timing else out

    details = call_details_api(wm_id, api_key=key)
    if details is None:
        out = {"watchmode_id": wm_id, "similar_count": 0, "results": [],
               "error": "Details API returned no data"}
        total_ms = (time.perf_counter() - t0) * 1000.0
        return (out, total_ms) if timing else out

    similar_ids = details.get("similar_titles", [])
    if not isinstance(similar_ids, list):
        similar_ids = []

    resolved = resolve_watchmode_ids_batch([int(x) for x in similar_ids])

    out = {
        "watchmode_id":  wm_id,
        "similar_count": len(resolved),
        "results":       resolved,
        "details_title": details.get("title"),
        "details_year":  details.get("year"),
    }
    total_ms = (time.perf_counter() - t0) * 1000.0
    return (out, total_ms) if timing else out


# =====================================================================
# Debug / status
# =====================================================================
def dataset_info() -> Dict[str, Any]:
    mappings_status = {}
    for name, m in _mappings.items():
        if m is not None and m.loaded:
            mappings_status[name] = {
                "loaded":   True,
                "count":    m.count,
                "sizes":    m.sizes,
                "is_mmap":  m.is_mmap,
            }
        else:
            mappings_status[name] = {"loaded": False}

    return {
        "loaded":       all(m is not None and m.loaded for m in _mappings.values()),
        "dataset_id":   _dataset_id,
        "runtime_mode": _runtime_mode,
        "api_key_set":  bool(get_api_key()),
        "mappings":     mappings_status,
        "meta":         _load_meta or {},
    }