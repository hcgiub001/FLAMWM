# -*- coding: utf-8 -*-
"""
watchnext_api.py — Fetch "similar titles" + extras via Watchmode API
using local TMDB↔Watchmode binary mappings.

Flow:
  ① Convert TMDB ID → Watchmode ID (local binary mapping)
  ② Call Watchmode API /title/{id}/details/ to get the full response
     — If local mapping found a numeric Watchmode ID, use it (1 credit).
     — Otherwise, use TMDB format "movie-{id}" or "tv-{id}" (2 credits).
  ③ Extract similar_titles AND extras from that single response
  ④ Convert returned Watchmode IDs → TMDB IDs (local binary mapping)

Packed TMDB format: (tmdb_id << 1) | type_bit   (0 = movie, 1 = tv)

Public API (see usage notes at bottom of file):
  get_watchnext_bundle_for_addon(tmdb_id, kind)
      → {"results": [...], "total_results": N, "watchmode_extras": {...}}
  get_watchnext_for_addon(tmdb_id, kind)
      → {"results": [...], "total_results": N}
  get_watchmode_extras(tmdb_id, kind)
      → {"will_you_like_this": ..., "review_summary": ..., ...}
  query_watchnext_packed(tmdb_id, kind)
      → [packed_int, ...]
  query_watchnext_pairs(tmdb_id, kind)
      → {"count": N, "results": [[id, type_bit], ...]}
"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import http.client
import json
import mmap
import os
import platform
import ssl
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from array import array
from typing import Any, Dict, List, Optional, Tuple, Union

from caches.main_cache import main_cache
from caches.settings_cache import get_setting, set_setting
from modules.kodi_utils import translate_path, kodi_log


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
_BASE_URL          = "https://hcgiub001.github.io/FLAMWM"
_WATCHMODE_API_BASE = "https://api.watchmode.com/v1"

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

_ALL_FILES: Tuple[str, ...] = tuple({
    f for spec in _MAPPING_SPECS.values()
    for f in (spec["meta"], spec["sample"], spec["keys"], spec["values"])
})

_SETTING_KEY_MODE    = "fenlight.watchmode.mode"
_SETTING_KEY_API_KEY = "fenlight.watchmode.api_key"

_SETTING_KEY_CACHE_DETAILS_HOURS  = "fenlight.watchmode.cache.details_hours"
_SETTING_KEY_CACHE_SIMILAR_HOURS  = "fenlight.watchmode.cache.similar_hours"
_SETTING_KEY_CACHE_NEGATIVE_HOURS = "fenlight.watchmode.cache.negative_hours"

_DEFAULT_DETAILS_CACHE_HOURS  = 168   # 7 days
_DEFAULT_SIMILAR_CACHE_HOURS  = 24    # 1 day
_DEFAULT_NEGATIVE_CACHE_HOURS = 24    # 1 day

_CACHE_VERSION_KEY   = "watchmode:cache_version"
_REQUEST_TIMEOUT_S   = 30
_NEGATIVE_SENTINEL   = "WATCHMODE_NEGATIVE"

# Idle timeout for pooled connections (seconds)
_CONN_IDLE_TIMEOUT_S = 65

# Memory auto-detection thresholds
_AUTO_THRESHOLD_DEFAULT = 200 * 1024 * 1024
_SAFETY_MARGIN_MIN      = 64 * 1024 * 1024
_SAFETY_MARGIN_FRAC     = 0.05

_IS_LITTLE = sys.byteorder == "little"

# Fields extracted from the Watchmode /details/ response as "extras"
_EXTRAS_FIELDS = (
    "will_you_like_this",
    "review_summary",
    "user_rating",
    "critic_score",
    "relevance_percentile",
    "popularity_percentile",
)


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════
def _log(msg: str, level: int = 0) -> None:
    try:
        kodi_log(msg, level)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Pack / Unpack helpers
# ═══════════════════════════════════════════════════════════════════════════
def _is_tv(kind: str) -> bool:
    return kind.lower().startswith("tv")

def _pack_tmdb(tmdb_id: int, kind: str) -> int:
    """Encode TMDB ID + type into a single int: (tmdb_id << 1) | type_bit."""
    return (tmdb_id << 1) | (1 if _is_tv(kind) else 0)

def _unpack_tmdb(packed: int) -> Tuple[int, str]:
    """Decode packed int back to (tmdb_id, media_type)."""
    return packed >> 1, ("tv" if packed & 1 else "movie")

def _tmdb_format_id(tmdb_id: int, kind: str) -> str:
    """Build the TMDB-format ID accepted by Watchmode: 'movie-278' or 'tv-1396'."""
    return f"tv-{tmdb_id}" if _is_tv(kind) else f"movie-{tmdb_id}"


# ═══════════════════════════════════════════════════════════════════════════
# Cache version tracking (bump version to invalidate all cached results)
# ═══════════════════════════════════════════════════════════════════════════
_cache_version: Optional[str] = None
_cache_version_lock = threading.Lock()

def _get_cache_version() -> str:
    global _cache_version
    if _cache_version is not None:
        return _cache_version
    with _cache_version_lock:
        if _cache_version is not None:
            return _cache_version
        v = main_cache.get(_CACHE_VERSION_KEY)
        if v is None:
            v = str(int(time.time()))
            try:
                main_cache.set(_CACHE_VERSION_KEY, v, expiration=0)
            except Exception:
                pass
        _cache_version = str(v)
        return _cache_version

def clear_watchmode_cache() -> None:
    global _cache_version
    new_version = str(int(time.time()))
    with _cache_version_lock:
        _cache_version = new_version
    try:
        main_cache.set(_CACHE_VERSION_KEY, new_version, expiration=0)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Cache TTL settings
# ═══════════════════════════════════════════════════════════════════════════
def _get_ttl_hours(setting_key: str, default: int) -> int:
    try:
        h = int(get_setting(setting_key, str(default)))
        return h if h >= 1 else default
    except Exception:
        return default

def _details_cache_hours() -> int:
    return _get_ttl_hours(_SETTING_KEY_CACHE_DETAILS_HOURS, _DEFAULT_DETAILS_CACHE_HOURS)

def _similar_cache_hours() -> int:
    return _get_ttl_hours(_SETTING_KEY_CACHE_SIMILAR_HOURS, _DEFAULT_SIMILAR_CACHE_HOURS)

def _negative_cache_hours() -> int:
    return _get_ttl_hours(_SETTING_KEY_CACHE_NEGATIVE_HOURS, _DEFAULT_NEGATIVE_CACHE_HOURS)


# ═══════════════════════════════════════════════════════════════════════════
# Memory detection (used to choose RAM vs mmap loading strategy)
# ═══════════════════════════════════════════════════════════════════════════
def _parse_kodi_memory_mb(label: str) -> int:
    try:
        s = label.strip().upper()
        if "GB" in s:
            return int(float(s.replace("GB", "").strip()) * 1024)
        if "MB" in s:
            return int(float(s.replace("MB", "").strip()))
        if "KB" in s:
            return max(1, int(float(s.replace("KB", "").strip()) / 1024))
        return int(float(s))
    except Exception:
        return 0

def _get_mem_kodi() -> Optional[Tuple[int, int]]:
    try:
        import xbmc
    except ImportError:
        return None
    try:
        total_str = xbmc.getInfoLabel("System.Memory(total)")
        free_str = (
            xbmc.getInfoLabel("System.Memory(available)")
            or xbmc.getInfoLabel("System.Memory(free)")
            or xbmc.getInfoLabel("System.FreeMemory")
        )
        if not total_str or not free_str:
            return None
        total_mb = _parse_kodi_memory_mb(total_str)
        free_mb  = _parse_kodi_memory_mb(free_str)
        if total_mb <= 0 or free_mb <= 0:
            return None
        return free_mb << 20, total_mb << 20
    except Exception:
        return None

def _get_mem_psutil() -> Optional[Tuple[int, int]]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    except Exception:
        return None

def _get_mem_proc() -> Optional[Tuple[int, int]]:
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        info[parts[0].strip()] = int(parts[1].strip().split(None, 1)[0])
                    except (ValueError, IndexError):
                        pass
        avail_kb = info.get("MemAvailable")
        if avail_kb is None:
            avail_kb = int(
                (info.get("MemFree", 0) + info.get("Cached", 0) + info.get("Buffers", 0)) * 0.7
            )
        total_kb = info.get("MemTotal", 0)
        if avail_kb <= 0 or total_kb <= 0:
            return None
        return avail_kb << 10, total_kb << 10
    except Exception:
        return None

def _get_mem_windows() -> Optional[Tuple[int, int]]:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys), int(stat.ullTotalPhys)
    except Exception:
        pass
    return None

def _get_available_memory() -> Tuple[int, int]:
    """Returns (available_bytes, total_bytes). Either may be 0 if unknown."""
    for fn in (_get_mem_kodi, _get_mem_psutil):
        result = fn()
        if result:
            return result
    sysname = platform.system().lower()
    if sysname in ("linux", "android"):
        result = _get_mem_proc()
        if result:
            return result
    elif sysname == "windows":
        result = _get_mem_windows()
        if result:
            return result
    return 0, 0


# ═══════════════════════════════════════════════════════════════════════════
# SHA-1 short hash (for anonymising API key in cache keys)
# ═══════════════════════════════════════════════════════════════════════════
_sha1_cache: Dict[str, str] = {}

def _sha1_short(s: str) -> str:
    h = _sha1_cache.get(s)
    if h is None:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        _sha1_cache[s] = h
    return h


# ═══════════════════════════════════════════════════════════════════════════
# Atomic file write
# ═══════════════════════════════════════════════════════════════════════════
def _atomic_write_bytes(target_path: str, data: bytes) -> None:
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".tmp_wm", suffix=".bin")
    try:
        os.write(fd, data)
        os.close(fd)
        fd = -1
        os.replace(tmp, target_path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Proxy detection (reads env vars / system settings — no HTTP)
# Uses urllib.request.getproxies() which inspects:
#   • Environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY, etc.)
#   • Windows registry (Internet Settings)
#   • macOS System Configuration framework
# Result is cached after the first call.  proxy_bypass() performs simple
# string / pattern matching against the no_proxy list — no network I/O.
# ═══════════════════════════════════════════════════════════════════════════
_proxy_map: Optional[Dict[str, str]] = None
_proxy_map_lock = threading.Lock()

def _get_proxies() -> Dict[str, str]:
    """Return the system proxy dict, cached after the first call."""
    global _proxy_map
    if _proxy_map is not None:
        return _proxy_map
    with _proxy_map_lock:
        if _proxy_map is not None:
            return _proxy_map
        try:
            _proxy_map = urllib.request.getproxies()
        except Exception:
            _proxy_map = {}
        return _proxy_map

def _proxy_for(scheme: str, host: str) -> Optional[urllib.parse.ParseResult]:
    """Return the parsed proxy URL for scheme targeting host, or None.

    Returns None when no proxy is configured for the scheme, or when the
    host appears in the no_proxy / bypass list.
    """
    try:
        if urllib.request.proxy_bypass(host):
            return None
    except Exception:
        pass
    proxies = _get_proxies()
    url = proxies.get(scheme)
    if not url:
        return None
    # Ensure the URL has a scheme so urlparse splits it correctly
    # (e.g. a bare "proxy.local:8080" would otherwise land in path).
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.hostname:
        return None
    return parsed

def _proxy_auth_header(proxy: urllib.parse.ParseResult) -> Optional[str]:
    """Build a Basic Proxy-Authorization value, or None if no credentials."""
    if not proxy.username:
        return None
    cred = "{}:{}".format(
        urllib.parse.unquote(proxy.username),
        urllib.parse.unquote(proxy.password or ""),
    )
    return "Basic " + base64.b64encode(cred.encode()).decode()


# ═══════════════════════════════════════════════════════════════════════════
# HTTP Connection Pool (keep-alive, thread-safe, follows redirects,
# drops idle connections after _CONN_IDLE_TIMEOUT_S, respects system
# proxy settings)
# ═══════════════════════════════════════════════════════════════════════════
class _ConnectionPool:
    """Thread-safe HTTP/HTTPS connection pool with keep-alive, redirect
    support, system proxy detection, and automatic eviction of connections
    idle longer than _CONN_IDLE_TIMEOUT_S seconds.

    Proxy behaviour
    ───────────────
    • HTTP target through an HTTP proxy:
      TCP connects to the proxy; the full URL is sent as the request
      path so the proxy can forward to the origin server.
    • HTTPS target through an HTTP proxy:
      TCP connects to the proxy, issues a CONNECT request to establish
      a tunnel, then wraps the socket in TLS to the target
      (http.client.HTTPSConnection.set_tunnel).
    • No proxy configured (the common case):
      Direct connection — identical to the previous behaviour, zero
      extra overhead.

    Proxy credentials embedded in the proxy URL
    (http://user:pass@proxy:port) are forwarded as a
    Proxy-Authorization: Basic … header automatically.
    """

    _MAX_REDIRECTS = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Maps (scheme, host, port) → (connection, last_used_monotonic)
        self._conns: Dict[Tuple[str, str, int], Tuple[http.client.HTTPConnection, float]] = {}

    def fetch(self, url: str, timeout: int = _REQUEST_TIMEOUT_S) -> Tuple[int, bytes]:
        """GET a URL, following redirects. Returns (status_code, response_body)."""
        current = url
        for _ in range(self._MAX_REDIRECTS + 1):
            status, body, location = self._get_once(current, timeout)
            if 300 <= status < 400 and location:
                current = urllib.parse.urljoin(current, location)
                continue
            return status, body
        raise IOError(f"Too many redirects: {url}")

    def close_all(self) -> None:
        with self._lock:
            for conn, _ts in self._conns.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()

    # --- internals ---
    def _get_once(self, url: str, timeout: int) -> Tuple[int, bytes, Optional[str]]:
        parsed = urllib.parse.urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        host   = parsed.hostname or ""
        port   = parsed.port or (443 if scheme == "https" else 80)
        path   = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        key = (scheme, host, port)

        # ── Proxy detection ──────────────────────────────────────────
        proxy = _proxy_for(scheme, host)
        # For HTTP through a proxy the request line must carry the full
        # URL so the proxy knows where to forward.  HTTPS tunnels (via
        # CONNECT) are transparent — use the normal path.
        is_http_proxy = proxy is not None and scheme != "https"
        request_path  = url if is_http_proxy else path

        headers = {
            "Host":       host,
            "User-Agent": "watchmode_api/2.0",
            "Accept":     "application/json,*/*",
            "Connection": "keep-alive",
        }

        # Proxy-Authorization for plain-HTTP proxy requests.
        # (For HTTPS tunnels the auth header is passed via set_tunnel
        #  inside _new_conn instead.)
        if is_http_proxy and proxy is not None:
            auth = _proxy_auth_header(proxy)
            if auth:
                headers["Proxy-Authorization"] = auth

        # Try reusing an existing connection (only if not idle too long).
        conn: Optional[http.client.HTTPConnection] = None
        with self._lock:
            entry = self._conns.pop(key, None)
        if entry is not None:
            stored_conn, stored_ts = entry
            if (time.monotonic() - stored_ts) <= _CONN_IDLE_TIMEOUT_S:
                conn = stored_conn
            else:
                try:
                    stored_conn.close()
                except Exception:
                    pass

        if conn is not None:
            try:
                conn.request("GET", request_path, headers=headers)
                resp = conn.getresponse()
                body = resp.read()
                self._return_conn(key, conn, resp)
                return resp.status, body, resp.getheader("Location")
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

        # Open a new connection (direct or through proxy).
        conn = self._new_conn(scheme, host, port, timeout, proxy)
        try:
            conn.request("GET", request_path, headers=headers)
            resp = conn.getresponse()
            body = resp.read()
            self._return_conn(key, conn, resp)
            return resp.status, body, resp.getheader("Location")
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

    def _return_conn(self, key, conn, resp) -> None:
        """Return connection to pool unless the server signalled close."""
        conn_header = (resp.getheader("Connection") or "").lower()
        if conn_header == "close" or getattr(resp, "will_close", False):
            try:
                conn.close()
            except Exception:
                pass
            return
        now = time.monotonic()
        with self._lock:
            old_entry = self._conns.pop(key, None)
            self._conns[key] = (conn, now)
        if old_entry is not None and old_entry[0] is not conn:
            try:
                old_entry[0].close()
            except Exception:
                pass

    @staticmethod
    def _new_conn(scheme: str, host: str, port: int, timeout: int,
                  proxy: Optional[urllib.parse.ParseResult] = None):
        """Create a new HTTP(S) connection, optionally through proxy.

        • HTTP  + proxy → HTTPConnection to proxy (caller uses full URL
          as request path).
        • HTTPS + proxy → HTTPSConnection to proxy with CONNECT tunnel
          to *host:port*; TLS is negotiated end-to-end through the
          tunnel.
        • No proxy      → direct connection (previous behaviour).
        """
        if proxy is not None:
            p_host = proxy.hostname or ""
            p_port = proxy.port or 8080

            if scheme == "https":
                # CONNECT tunnel: TCP → proxy, then TLS → target
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(
                    p_host, p_port, timeout=timeout, context=ctx,
                )
                tunnel_hdrs: Optional[Dict[str, str]] = None
                auth = _proxy_auth_header(proxy)
                if auth:
                    tunnel_hdrs = {"Proxy-Authorization": auth}
                conn.set_tunnel(host, port, headers=tunnel_hdrs)
                return conn
            else:
                # Plain HTTP: connect to proxy, full URL as path
                return http.client.HTTPConnection(
                    p_host, p_port, timeout=timeout,
                )

        # No proxy — direct connection.
        if scheme == "https":
            ctx = ssl.create_default_context()
            return http.client.HTTPSConnection(host, port, timeout=timeout, context=ctx)
        return http.client.HTTPConnection(host, port, timeout=timeout)


_http_pool = _ConnectionPool()


# ═══════════════════════════════════════════════════════════════════════════
# File paths
# ═══════════════════════════════════════════════════════════════════════════
_cache_dir: Optional[str] = None

def _ensure_cache_dir() -> str:
    global _cache_dir
    if _cache_dir is not None:
        return _cache_dir
    base = translate_path("special://profile/addon_data/plugin.video.fenlight/")
    d = os.path.join(base, "cache", "watchmode")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    _cache_dir = d
    return d

def _file_path(filename: str) -> str:
    return os.path.join(_ensure_cache_dir(), filename)


# ═══════════════════════════════════════════════════════════════════════════
# Download mapping files from GitHub
# ═══════════════════════════════════════════════════════════════════════════
def _ensure_files_downloaded() -> bool:
    """Download any missing mapping files. Returns True if all present."""
    all_ok = True
    for fname in _ALL_FILES:
        path = _file_path(fname)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        url = f"{_BASE_URL}/{fname}"
        try:
            status, data = _http_pool.fetch(url)
            if status >= 400:
                raise IOError(f"HTTP {status}")
            _atomic_write_bytes(path, data)
        except Exception as e:
            _log(f"watchnext_api: download failed {fname}: {e}", 1)
            all_ok = False
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Manual binary search (safe on both array and mmap-backed memoryview)
# ═══════════════════════════════════════════════════════════════════════════
def _binary_search(keys, key: int, lo: int, hi: int) -> int:
    """
    Return the index of *key* inside keys[lo:hi], or -1 if absent.
    Uses a plain while-loop so that each iteration only does a single
    keys[mid] element access — safe on mmap-backed memoryview objects
    where the C bisect module's PySequence_GetItem can fault pages
    unexpectedly on some CPython/Android builds.
    Works identically on array('I') objects in RAM mode.
    """
    while lo < hi:
        mid = (lo + hi) >> 1
        v = keys[mid]
        if v < key:
            lo = mid + 1
        elif v > key:
            hi = mid
        else:
            return mid
    return -1

def _upper_bound(sample, key: int) -> int:
    """
    Return the index of the first element in *sample* that is > key,
    or len(sample) if all elements are <= key.
    Equivalent to bisect.bisect_right(sample, key) but safe on memoryview.
    """
    lo, hi = 0, len(sample)
    while lo < hi:
        mid = (lo + hi) >> 1
        if sample[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ═══════════════════════════════════════════════════════════════════════════
# madvise helper — tell the kernel our access pattern is random so it
# does NOT prefetch sequential pages around every fault.
# Available since Python 3.8 on Unix.  Silently skipped elsewhere.
# ═══════════════════════════════════════════════════════════════════════════
def _madvise_random(mm: mmap.mmap) -> None:
    try:
        mm.madvise(mmap.MADV_RANDOM)   # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Binary mapping (sorted u32 key→value files, O(log n) lookup)
# ═══════════════════════════════════════════════════════════════════════════
class PackedMapping:
    """
    Lookup table backed by sorted binary files of u32 keys and u32 values.

    RAM mode:  keys & values are array('I') in memory.  A 64 K direct-map
               bucket index over the top 16 bits of each key gives O(1)
               range narrowing — only a tiny final binary search remains.
    mmap mode: keys & values are mmap-backed memoryview objects.  A small
               sample array (every Nth key) narrows the bisect range so
               lookups touch only one stride-sized block of pages.
               MADV_RANDOM is set on every mmap to prevent the kernel from
               prefetching sequential pages on each random-access fault.
    """
    __slots__ = (
        "_keys", "_vals", "_sample", "_meta",
        "_count", "_stride", "_direct_map",
        "_fileobjs", "_mmaps", "_is_mmap",
    )

    def __init__(self) -> None:
        self._keys: Any                    = None
        self._vals: Any                    = None
        self._sample: Optional[array]      = None
        self._meta: Optional[Dict[str, Any]] = None
        self._count: int                   = 0
        self._stride: int                  = 64
        self._direct_map: Optional[array]  = None
        self._fileobjs: List[Any]          = []
        self._mmaps: List[Any]             = []
        self._is_mmap: bool                = False

    @property
    def loaded(self) -> bool:
        return self._keys is not None

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_mmap(self) -> bool:
        return self._is_mmap

    def close(self) -> None:
        for mm in self._mmaps:
            try:
                mm.close()
            except Exception:
                pass
        for fo in self._fileobjs:
            try:
                fo.close()
            except Exception:
                pass
        self._mmaps.clear()
        self._fileobjs.clear()
        self._keys = self._vals = self._sample = self._meta = None
        self._direct_map = None
        self._count = 0

    def load_ram(self, spec: Dict[str, str]) -> None:
        """Load all files fully into RAM arrays, then build the direct-map."""
        self.close()
        self._is_mmap = False
        self._load_meta(spec)
        self._sample = self._load_u32_file(_file_path(spec["sample"]))
        self._keys   = self._load_u32_file(_file_path(spec["keys"]))
        self._vals   = self._load_u32_file(_file_path(spec["values"]))
        self._count  = len(self._keys)
        self._build_direct_map()

    def load_mmap(self, spec: Dict[str, str]) -> None:
        """Load sample into RAM; memory-map keys and values from disk.

        Each mmap is marked MADV_RANDOM so the kernel does not prefetch
        surrounding pages on every random-access fault.  Without this
        hint the default readahead (typically 128-256 KB per fault) can
        pull the entire file into physical RAM after only a few dozen
        scattered lookups — exactly the symptom of "mmap faulting
        everything into RAM".

        The direct-map is intentionally NOT built here: constructing it
        requires iterating every key (keys[i] >> 16), which would
        page-fault the entire mmap'd file into physical RAM and defeat
        the purpose of mmap.  The sample-based binary search is used
        instead — it only touches the small sample array (already in
        RAM) plus one narrow block of keys per lookup.
        """
        self.close()
        self._is_mmap = True
        self._load_meta(spec)
        self._sample     = self._load_u32_file(_file_path(spec["sample"]))
        self._direct_map = None          # explicitly skip

        for attr, key in (("_keys", "keys"), ("_vals", "values")):
            path = _file_path(spec[key])
            size = os.path.getsize(path)
            # Align to 4-byte boundary so .cast("I") is safe.
            aligned = (size // 4) * 4
            if aligned == 0:
                raise ValueError(f"Mapping file too small: {path}")
            fo = open(path, "rb")
            self._fileobjs.append(fo)
            mm = mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ)
            _madvise_random(mm)
            self._mmaps.append(mm)
            setattr(self, attr, memoryview(mm)[:aligned].cast("I"))

        self._count = len(self._keys)

    def lookup(self, key: int) -> Optional[int]:
        """Find the value for *key*, or None."""
        keys = self._keys
        if keys is None:
            return None

        # ---- Fast path: direct-map bucket index (RAM mode only) ----
        dm = self._direct_map
        if dm is not None:
            hb = key >> 16
            lo = dm[hb]
            hi = dm[hb + 1]
        else:
            # ---- Fallback: sample-based range narrowing (mmap-safe) ----
            sample = self._sample
            if sample is not None and len(sample) > 0:
                bucket = _upper_bound(sample, key) - 1
                lo = max(0, bucket * self._stride)
                hi = min(lo + self._stride, self._count)
            else:
                lo, hi = 0, self._count

        if lo >= hi:
            return None

        i = _binary_search(keys, key, lo, hi)
        return self._vals[i] if i >= 0 else None

    # --- internals ---
    def _build_direct_map(self) -> None:
        """Build a 64 K bucket index over the top 16 bits of each key.

        Only called in RAM mode where all keys are already resident.
        """
        keys = self._keys
        if keys is None or len(keys) == 0:
            self._direct_map = None
            return

        count = len(keys)
        dm = array("I", [count] * 65537)

        for i in range(count):
            hb = keys[i] >> 16
            if i < dm[hb]:
                dm[hb] = i

        # Backward fill: empty buckets inherit the next bucket's start.
        for hb in range(65535, -1, -1):
            if dm[hb] > dm[hb + 1]:
                dm[hb] = dm[hb + 1]

        dm[65536] = count
        self._direct_map = dm

    def _load_meta(self, spec: Dict[str, str]) -> None:
        with open(_file_path(spec["meta"]), "r", encoding="utf-8") as fh:
            self._meta = json.load(fh)
        if (int(self._meta.get("key_bytes", 4)) != 4
                or int(self._meta.get("value_bytes", 4)) != 4):
            raise ValueError("Only 32-bit key/value files supported")
        self._stride = int(self._meta.get("sample_stride", 64))

    @staticmethod
    def _load_u32_file(path: str) -> array:
        with open(path, "rb") as fh:
            data = fh.read()
        a = array("I")
        a.frombytes(data[:len(data) // 4 * 4])
        if not _IS_LITTLE:
            a.byteswap()
        return a


# ═══════════════════════════════════════════════════════════════════════════
# Module state (loaded mappings, runtime mode)
# ═══════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()

_mappings: Dict[str, Optional[PackedMapping]] = {
    "tmdb2wm_movie": None,
    "tmdb2wm_tv":    None,
    "wm2tmdb":       None,
}
_runtime_mode: Optional[str] = None
_dataset_id:   Optional[str] = None

def _is_loaded() -> bool:
    return all(m is not None and m.loaded for m in _mappings.values())

def _compute_dataset_id() -> str:
    parts: List[str] = []
    for fname in _ALL_FILES:
        path = _file_path(fname)
        try:
            st = os.stat(path)
            parts.append(f"{fname}:{int(st.st_mtime)}:{st.st_size}")
        except Exception:
            parts.append(f"{fname}:-")
    return "|".join(parts)

def _close_all_mappings() -> None:
    global _runtime_mode, _dataset_id
    for name in _mappings:
        if _mappings[name] is not None:
            _mappings[name].close()
        _mappings[name] = None
    _runtime_mode = _dataset_id = None


# ═══════════════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════════════
def get_setting_mode() -> str:
    v = (get_setting(_SETTING_KEY_MODE, "auto") or "auto").strip().lower()
    if v == "ram":
        return "RAM"
    if v == "mmap":
        return "mmap"
    return "auto"

def set_setting_mode(mode: str) -> bool:
    m = (mode or "auto").strip().lower()
    if m not in ("ram", "mmap", "auto"):
        return False
    try:
        set_setting(_SETTING_KEY_MODE, m)
        return True
    except Exception:
        return False

def get_runtime_mode() -> Optional[str]:
    return _runtime_mode

def get_api_key() -> str:
    try:
        return get_setting(_SETTING_KEY_API_KEY, "") or ""
    except Exception:
        return ""

def set_api_key(key: str) -> bool:
    try:
        set_setting(_SETTING_KEY_API_KEY, (key or "").strip())
        clear_watchmode_cache()
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle (load / reload / close)
# ═══════════════════════════════════════════════════════════════════════════
def ensure_loaded(mode: Optional[str] = None, force: bool = False) -> None:
    """Make sure all three mapping tables are loaded. Thread-safe, idempotent."""
    global _runtime_mode, _dataset_id

    if _is_loaded() and not force:
        return

    with _lock:
        if _is_loaded() and not force:
            return

        _ensure_files_downloaded()

        chosen = mode if mode else get_setting_mode()

        # Auto-select RAM vs mmap based on available memory.
        if chosen == "auto":
            total_file_size = sum(
                os.path.getsize(_file_path(f))
                for f in _ALL_FILES
                if os.path.isfile(_file_path(f))
            )
            avail, total = _get_available_memory()
            safety = (max(_SAFETY_MARGIN_MIN, int(total * _SAFETY_MARGIN_FRAC))
                      if total else _SAFETY_MARGIN_MIN)
            chosen = ("RAM"
                      if (avail >= _AUTO_THRESHOLD_DEFAULT
                          and avail >= total_file_size + safety)
                      else "mmap")

        for name, spec in _MAPPING_SPECS.items():
            if _mappings[name] is not None:
                _mappings[name].close()
            m = PackedMapping()
            if chosen == "RAM":
                m.load_ram(spec)
            else:
                m.load_mmap(spec)
            _mappings[name] = m

        _runtime_mode = chosen
        _dataset_id   = _compute_dataset_id()

def reload_dataset(mode: Optional[str] = None) -> None:
    with _lock:
        _close_all_mappings()
    clear_watchmode_cache()
    ensure_loaded(mode=mode, force=True)

def close_dataset() -> None:
    with _lock:
        _close_all_mappings()
    _http_pool.close_all()


# ═══════════════════════════════════════════════════════════════════════════
# Local mapping lookups
# ═══════════════════════════════════════════════════════════════════════════
def lookup_tmdb_to_watchmode(tmdb_id: int, kind: str) -> Optional[int]:
    """TMDB ID + kind → Watchmode ID, or None."""
    if not _is_loaded():
        ensure_loaded()
    table = _mappings["tmdb2wm_tv" if _is_tv(kind) else "tmdb2wm_movie"]
    return table.lookup(_pack_tmdb(tmdb_id, kind)) if table else None

def lookup_watchmode_to_tmdb_packed(watchmode_id: int) -> Optional[int]:
    """Watchmode ID → packed TMDB int, or None."""
    if not _is_loaded():
        ensure_loaded()
    table = _mappings.get("wm2tmdb")
    return table.lookup(watchmode_id) if table else None


# ═══════════════════════════════════════════════════════════════════════════
# Watchmode API calls (cached HTTP with negative-caching & dedup)
# ═══════════════════════════════════════════════════════════════════════════
def _sorted_query_string(params: Dict[str, Any]) -> str:
    return urllib.parse.urlencode(
        sorted((str(k), str(v)) for k, v in params.items() if v is not None),
        doseq=True,
    )

def _make_cache_key(namespace: str, endpoint: str,
                    params: Dict[str, Any], api_key: str) -> str:
    v = _get_cache_version()
    return (f"wnapi:{namespace}:{v}:{_sha1_short(api_key)}"
            f":{endpoint}:{_sorted_query_string(params)}")

# In-flight dedup: one lock per cache key prevents duplicate parallel API calls.
_inflight_lock = threading.Lock()
_inflight: Dict[str, threading.Lock] = {}

def _cached_api_get(url: str, cache_key: str, ttl_hours: int,
                    neg_hours: int, bypass: bool = False) -> Any:
    """Fetch JSON from url with SQLite caching and negative-caching."""

    # Check cache.
    if not bypass:
        try:
            cached = main_cache.get(cache_key)
            if cached is not None:
                return None if cached == _NEGATIVE_SENTINEL else cached
        except Exception:
            pass

    # Acquire per-key lock so identical requests don't fire in parallel.
    with _inflight_lock:
        lk = _inflight.get(cache_key)
        if lk is None:
            lk = threading.Lock()
            _inflight[cache_key] = lk

    with lk:
        # Re-check cache (another thread may have just filled it).
        if not bypass:
            try:
                cached = main_cache.get(cache_key)
                if cached is not None:
                    return None if cached == _NEGATIVE_SENTINEL else cached
            except Exception:
                pass

        try:
            status, body = _http_pool.fetch(url, timeout=_REQUEST_TIMEOUT_S)
        except Exception:
            return None
        finally:
            with _inflight_lock:
                _inflight.pop(cache_key, None)

        # Simple retry on rate-limit.
        if status == 429:
            time.sleep(2)
            try:
                status, body = _http_pool.fetch(url, timeout=_REQUEST_TIMEOUT_S)
            except Exception:
                return None

        if status in (401, 403):
            raise PermissionError(f"Watchmode API auth error HTTP {status}")

        if status >= 400:
            try:
                main_cache.set(cache_key, _NEGATIVE_SENTINEL, expiration=neg_hours)
            except Exception:
                pass
            return None

        try:
            data = json.loads(body)
        except Exception:
            try:
                main_cache.set(cache_key, _NEGATIVE_SENTINEL, expiration=neg_hours)
            except Exception:
                pass
            return None

        try:
            main_cache.set(cache_key, data, expiration=ttl_hours)
        except Exception:
            pass
        return data

def _call_api(endpoint: str, params: Dict[str, Any], api_key: str,
              namespace: str, ttl_hours: int,
              bypass: bool = False) -> Optional[Dict[str, Any]]:
    """Low-level Watchmode API GET."""
    ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    url_params = dict(params)
    url_params["apiKey"] = api_key
    url = f"{_WATCHMODE_API_BASE}{ep}?{_sorted_query_string(url_params)}"

    ck = _make_cache_key(namespace, ep, params, api_key)
    data = _cached_api_get(url, ck, ttl_hours, _negative_cache_hours(),
                           bypass=bypass)
    return data if isinstance(data, dict) else None

def _fetch_details(title_id: Union[int, str], api_key: str,
                   bypass_cache: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch /title/{id}/details/ from Watchmode API.

    title_id may be a numeric Watchmode ID (e.g. 345534, 1 credit),
    an IMDB ID (e.g. "tt0903747", 2 credits), or a TMDB-format ID
    (e.g. "movie-278" / "tv-1396", 2 credits).
    """
    return _call_api(f"/title/{title_id}/details/", {}, api_key,
                     namespace="details", ttl_hours=_details_cache_hours(),
                     bypass=bypass_cache)


# ═══════════════════════════════════════════════════════════════════════════
# Core: bundle function  (similar titles + extras in one call)
#
# This is the single workhorse.  Every public query function below
# delegates to get_watchnext_bundle_for_addon and then returns whichever
# slice of the result the caller asked for.  Because the bundle is
# cached, subsequent calls for the same title are instant dict lookups.
# ═══════════════════════════════════════════════════════════════════════════
def _bundle_cache_key(tmdb_id: int, kind: str, api_key: str) -> str:
    v   = _get_cache_version()
    did = _dataset_id or "-"
    return f"wn:bundle:{v}:{did}:{_sha1_short(api_key)}:{tmdb_id}:{kind}"

def _empty_bundle() -> Dict[str, Any]:
    """Return a bundle with no results and None for every extras field."""
    return {
        "results":          [],
        "total_results":    0,
        "watchmode_extras": {f: None for f in _EXTRAS_FIELDS},
    }

def get_watchnext_bundle_for_addon(
    tmdb_id: int,
    kind: str,
    api_key: Optional[str] = None,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    One-stop call: returns similar titles AND watchmode extras.

    Returns
    -------
    {
        "results": [
            {"id": 278, "media_type": "movie"},
            {"id": 1396, "media_type": "tv"},
            ...
        ],
        "total_results": 25,
        "watchmode_extras": {
            "will_you_like_this":    "You'll enjoy this if ...",
            "review_summary":        "Pros: ... | Cons: ...",
            "user_rating":           7.4,
            "critic_score":          71,
            "relevance_percentile":  99.58,
            "popularity_percentile": 98.703
        }
    }

    If the API call fails, "results" will be [] and every extras field
    will be None.
    """
    if not _is_loaded():
        ensure_loaded()

    key = api_key or get_api_key()
    if not key:
        return _empty_bundle()

    cache_key = _bundle_cache_key(tmdb_id, kind, key)

    # ── Check bundle cache ────────────────────────────────────────
    if not bypass_cache:
        try:
            cached = main_cache.get(cache_key)
            if cached is not None:
                return cached if isinstance(cached, dict) else _empty_bundle()
        except Exception:
            pass

    neg_hours = _negative_cache_hours()
    ttl       = _similar_cache_hours()

    # ── Step 1: TMDB → Watchmode title identifier (local, instant) ─
    #    Try the fast local binary mapping first (numeric Watchmode ID
    #    — costs 1 API credit).  If missing, fall back to the TMDB-
    #    format ID the API accepts directly (e.g. "movie-278", 2 credits).
    wm_id: Optional[int] = lookup_tmdb_to_watchmode(tmdb_id, kind)
    title_id: Union[int, str] = wm_id if wm_id is not None else _tmdb_format_id(tmdb_id, kind)

    # ── Step 2: Call Watchmode API (the only network wait) ─────────
    try:
        details = _fetch_details(title_id, key, bypass_cache=bypass_cache)
    except Exception:
        details = None

    if not isinstance(details, dict):
        empty = _empty_bundle()
        try:
            main_cache.set(cache_key, empty, expiration=neg_hours)
        except Exception:
            pass
        return empty

    # ── Step 3: Extract extras (instant — just reading dict keys) ──
    extras: Dict[str, Any] = {}
    for field in _EXTRAS_FIELDS:
        extras[field] = details.get(field)

    # ── Step 4: Extract similar_titles & convert to TMDB IDs ───────
    similar_ids = details.get("similar_titles")
    results: List[Dict[str, Any]] = []

    if isinstance(similar_ids, list) and similar_ids:
        wm2tmdb = _mappings.get("wm2tmdb")
        if wm2tmdb is not None and wm2tmdb.loaded:
            lookup = wm2tmdb.lookup          # local ref avoids repeated attr lookup
            for wid in similar_ids:
                try:
                    packed = lookup(int(wid))
                except (TypeError, ValueError):
                    continue
                if packed is not None:
                    results.append({
                        "id":         packed >> 1,
                        "media_type": "tv" if packed & 1 else "movie",
                    })

    # ── Package & cache ───────────────────────────────────────────
    bundle: Dict[str, Any] = {
        "results":          results,
        "total_results":    len(results),
        "watchmode_extras": extras,
    }

    try:
        main_cache.set(cache_key, bundle, expiration=ttl)
    except Exception:
        pass

    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrappers (all backed by the cached bundle — zero extra
# network calls; the first call for a given title pays the API cost,
# every subsequent call for the same title is an instant cache hit)
# ═══════════════════════════════════════════════════════════════════════════
def get_watchnext_for_addon(
    tmdb_id: int, kind: str, **kwargs
) -> Dict[str, Any]:
    """Similar titles only (no extras).

    Returns {"results": [{"id": …, "media_type": …}, ...], "total_results": N}
    """
    b = get_watchnext_bundle_for_addon(tmdb_id, kind, **kwargs)
    return {"results": b["results"], "total_results": b["total_results"]}


def get_watchmode_extras(
    tmdb_id: int, kind: str, **kwargs
) -> Dict[str, Any]:
    """Extras only (no similar titles list).

    Returns
    -------
    {
        "will_you_like_this":    str or None,
        "review_summary":        str or None,
        "user_rating":           float or None,
        "critic_score":          int or None,
        "relevance_percentile":  float or None,
        "popularity_percentile": float or None,
    }
    """
    return get_watchnext_bundle_for_addon(tmdb_id, kind, **kwargs)["watchmode_extras"]


def query_watchnext_packed(
    tmdb_id: int, kind: str, **kwargs
) -> List[int]:
    """Similar titles as packed ints: (tmdb_id << 1) | type_bit.

    This is the fastest output format — no dicts allocated per item.
    type_bit: 0 = movie, 1 = tv.
    """
    b = get_watchnext_bundle_for_addon(tmdb_id, kind, **kwargs)
    return [
        (r["id"] << 1) | (1 if r["media_type"] == "tv" else 0)
        for r in b["results"]
    ]


def query_watchnext_pairs(
    tmdb_id: int, kind: str, **kwargs
) -> Dict[str, Any]:
    """Similar titles as [tmdb_id, type_bit] pairs.

    Returns {"count": N, "results": [[tmdb_id, type_bit], ...]}
    type_bit: 0 = movie, 1 = tv.
    """
    packed = query_watchnext_packed(tmdb_id, kind, **kwargs)
    return {"count": len(packed), "results": [[p >> 1, p & 1] for p in packed]}


# ═══════════════════════════════════════════════════════════════════════════
# Debug / status
# ═══════════════════════════════════════════════════════════════════════════
def dataset_info() -> Dict[str, Any]:
    mapping_info = {}
    for name, m in _mappings.items():
        if m is not None and m.loaded:
            mapping_info[name] = {
                "loaded": True, "count": m.count, "is_mmap": m.is_mmap,
            }
        else:
            mapping_info[name] = {"loaded": False}
    return {
        "loaded":           _is_loaded(),
        "dataset_id":       _dataset_id,
        "runtime_mode":     _runtime_mode,
        "api_key_set":      bool(get_api_key()),
        "proxy_configured": bool(
            _get_proxies().get("http") or _get_proxies().get("https")
        ),
        "mappings":         mapping_info,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API REFERENCE & USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════
#
# All five public query functions ultimately call
# get_watchnext_bundle_for_addon(), which:
#   1. Looks up TMDB→Watchmode ID via local binary mapping  (instant)
#   2. Calls Watchmode /title/{id}/details/                 (one HTTP call, cached)
#   3. Extracts similar_titles + extras from the response   (instant)
#   4. Converts Watchmode IDs → TMDB IDs via local mapping  (instant)
#   5. Caches the assembled bundle                          (instant)
#
# The FIRST call for a given (tmdb_id, kind) pays the network cost.
# Every subsequent call hits the cache and returns instantly.
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 1:  get_watchnext_bundle_for_addon  (RECOMMENDED — everything)
# ───────────────────────────────────────────────────────────────────────
#
#   from modules.watchnext_api import get_watchnext_bundle_for_addon
#
#   bundle = get_watchnext_bundle_for_addon(8337, "movie")
#
#   # ── Extras: available IMMEDIATELY, same response ──────────────
#   extras = bundle["watchmode_extras"]
#
#   extras["will_you_like_this"]     # "You'll enjoy this if ..."   (str or None)
#   extras["review_summary"]         # "Pros: ... | Cons: ..."      (str or None)
#   extras["user_rating"]            # 7.4                          (float or None)
#   extras["critic_score"]           # 71                           (int or None)
#   extras["relevance_percentile"]   # 99.58                        (float or None)
#   extras["popularity_percentile"]  # 98.703                       (float or None)
#
#   # ── Similar titles: also available immediately ────────────────
#   bundle["total_results"]          # 25
#   bundle["results"]               # [{"id": 278, "media_type": "movie"}, ...]
#
#   # ── Kodi window property example ──────────────────────────────
#   # my_window.setProperty("will_you_like_this",
#   #     extras.get("will_you_like_this") or "")
#   # my_window.setProperty("review_summary",
#   #     extras.get("review_summary") or "")
#   # my_window.setProperty("user_rating",
#   #     str(extras.get("user_rating") or ""))
#   # my_window.setProperty("critic_score",
#   #     str(extras.get("critic_score") or ""))
#   # my_window.setProperty("relevance_percentile",
#   #     str(extras.get("relevance_percentile") or ""))
#   # my_window.setProperty("popularity_percentile",
#   #     str(extras.get("popularity_percentile") or ""))
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 2:  get_watchnext_for_addon  (similar titles only, no extras)
# ───────────────────────────────────────────────────────────────────────
#
#   from modules.watchnext_api import get_watchnext_for_addon
#
#   data = get_watchnext_for_addon(8337, "movie")
#   data["total_results"]   # 25
#   data["results"]         # [{"id": 278, "media_type": "movie"}, ...]
#
#   # Same as bundle["results"] — just without the extras dict.
#   # Internally calls the bundle (so extras are fetched & cached too,
#   # they just aren't returned here).
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 3:  get_watchmode_extras  (extras only, no similar list)
# ───────────────────────────────────────────────────────────────────────
#
#   from modules.watchnext_api import get_watchmode_extras
#
#   extras = get_watchmode_extras(8337, "movie")
#
#   extras["will_you_like_this"]     # str or None
#   extras["review_summary"]         # str or None
#   extras["user_rating"]            # float or None
#   extras["critic_score"]           # int or None
#   extras["relevance_percentile"]   # float or None
#   extras["popularity_percentile"]  # float or None
#
#   # Useful when you only need the text/scores for a title info screen,
#   # without building the similar-titles grid.
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 4:  query_watchnext_packed  (similar titles as packed ints)
# ───────────────────────────────────────────────────────────────────────
#
#   from modules.watchnext_api import query_watchnext_packed
#
#   packed_list = query_watchnext_packed(8337, "movie")
#   # [556, 2793, ...]   ← each int is (tmdb_id << 1) | type_bit
#
#   # Decode one:
#   for p in packed_list:
#       tmdb_id    = p >> 1
#       media_type = "tv" if (p & 1) else "movie"
#
#   # Fastest format — no dicts allocated per result item.
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 5:  query_watchnext_pairs  (similar titles as [id, bit] pairs)
# ───────────────────────────────────────────────────────────────────────
#
#   from modules.watchnext_api import query_watchnext_pairs
#
#   data = query_watchnext_pairs(8337, "movie")
#   data["count"]    # 25
#   data["results"]  # [[278, 0], [1396, 1], ...]
#   #                     ^   ^       ^    ^
#   #                  tmdb  movie  tmdb   tv
#
# ───────────────────────────────────────────────────────────────────────
# OTHER PUBLIC FUNCTIONS (settings, lifecycle, debug)
# ───────────────────────────────────────────────────────────────────────
#
#   get_api_key()              → str         current Watchmode API key
#   set_api_key(key)           → bool        set key & clear cache
#   get_setting_mode()         → str         "RAM" | "mmap" | "auto"
#   set_setting_mode(mode)     → bool        persist mode preference
#   get_runtime_mode()         → str|None    actual mode after loading
#
#   ensure_loaded(mode, force) → None        load binary mappings
#   reload_dataset(mode)       → None        close + reload + clear cache
#   close_dataset()            → None        release all resources
#   clear_watchmode_cache()    → None        invalidate all cached results
#
#   dataset_info()             → dict        debug: loaded/mode/counts
#
#   lookup_tmdb_to_watchmode(tmdb_id, kind)      → int|None
#   lookup_watchmode_to_tmdb_packed(watchmode_id) → int|None
#
# ═══════════════════════════════════════════════════════════════════════════