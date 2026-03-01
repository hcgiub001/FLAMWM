# -*- coding: utf-8 -*-
"""
watchmode_api.py — Fetch "similar titles" + extras via Watchmode API
using local TMDB↔Watchmode binary mappings.

=== HIGH-LEVEL OVERVIEW ===

This module answers the question: "Given a movie or TV show I like,
what else should I watch?" It does this by:

1. Taking a TMDB ID (the ID used by The Movie Database) and converting
   it to a Watchmode ID using a pre-built binary lookup table stored
   on disk.

2. Calling the Watchmode API to get details about that title, which
   includes a list of similar titles (as Watchmode IDs) and extra
   metadata like user ratings and review summaries.

3. Converting those Watchmode IDs back to TMDB IDs (again using local
   binary lookup tables) so the rest of the addon can use them.

4. Caching everything so repeated lookups are instant.

=== THE FLOW (step by step) ===

  ① Convert TMDB ID → Watchmode ID (local binary mapping on disk)
  ② Call Watchmode API /title/{id}/details/ to get the full response
     — If local mapping found a numeric Watchmode ID, use it (1 credit).
     — Otherwise, use TMDB format "movie-{id}" or "tv-{id}" (2 credits).
  ③ Extract similar_titles AND extras from that single API response
  ④ Convert returned Watchmode IDs → TMDB IDs (local binary mapping)
  ⑤ Cache the assembled bundle so future calls skip the network entirely

=== ID PACKING FORMAT ===

To store both a TMDB ID and its media type (movie vs TV) in a single
integer, we use this encoding:

    packed = (tmdb_id << 1) | type_bit

Where type_bit is 0 for movies and 1 for TV shows.

Example: TMDB movie ID 278 → packed as (278 << 1) | 0 = 556
Example: TMDB TV ID 1396   → packed as (1396 << 1) | 1 = 2793

To decode: tmdb_id = packed >> 1, type = "tv" if (packed & 1) else "movie"

=== BINARY MAPPING FILES ===

The lookup tables are stored as sorted arrays of unsigned 32-bit integers.
There are three mapping tables:

  - tmdb2wm_movie: TMDB movie IDs → Watchmode IDs
  - tmdb2wm_tv:    TMDB TV IDs    → Watchmode IDs
  - wm2tmdb:       Watchmode IDs  → packed TMDB IDs

Each table has four files:
  - .keys.bin:    Sorted array of lookup keys (u32, little-endian)
  - .values.bin:  Corresponding values at the same index (u32, little-endian)
  - .sample.u32:  Every Nth key, used to narrow the binary search range
  - .meta.json:   Metadata (byte widths, sample stride, etc.)

Lookups work by binary-searching the keys array and returning the value
at the same index. The sample file provides "guideposts" so we don't
have to search the entire array — just a small window around the
guidepost.

=== LOADING MODES: RAM vs MMAP ===

RAM mode:  All files are read entirely into memory as Python arrays.
           A 64K "direct map" index is built over the top 16 bits of
           each key, giving near-O(1) lookups. Fast but uses more memory.

mmap mode: Files are memory-mapped from disk. The OS loads pages on
           demand as they're accessed. Uses very little memory but
           individual lookups may be slightly slower due to page faults.
           The sample file narrows the search range so only ~1 page
           gets faulted per lookup.

"auto" mode (default): Checks available system memory and picks RAM
           if there's enough, otherwise falls back to mmap.

=== CACHING STRATEGY ===

Three levels of caching:

1. Bundle cache: The fully assembled result (similar titles + extras)
   is cached in SQLite. Default TTL: 24 hours for successful results,
   24 hours for negative (no-data) results.

2. API response cache: Raw Watchmode API responses are cached separately.
   Default TTL: 7 days (details change infrequently).

3. Cache versioning: A version string is stored in the cache. Bumping
   it (via clear_watchmode_cache()) invalidates all cached results
   without deleting them — they simply won't match the new version.

=== NETWORK / PROXY ===

All HTTP is done via a keep-alive connection pool that:
- Reuses TCP connections across requests (saves handshake time)
- Follows redirects (up to 5 hops)
- Detects system proxy settings automatically
- Supports HTTP proxies with CONNECT tunneling for HTTPS
- Evicts idle connections after 65 seconds

=== PUBLIC API ===

Main functions (all return dicts or lists):
    get_watchmode_bundle_for_addon(tmdb_id, kind)
        → {"results": [...], "total_results": N, "watchmode_extras": {...}}
    get_watchmode_for_addon(tmdb_id, kind)
        → {"results": [...], "total_results": N}
    get_watchmode_extras(tmdb_id, kind)
        → {"will_you_like_this": ..., "review_summary": ..., ...}
    query_watchmode_packed(tmdb_id, kind)
        → [packed_int, ...]
    query_watchmode_pairs(tmdb_id, kind)
        → {"count": N, "results": [[id, type_bit], ...]}

Configuration:
    get_api_key() / set_api_key(key)
    get_setting_mode() / set_setting_mode(mode)
    get_runtime_mode()

Lifecycle:
    ensure_loaded(mode, force)   — Load mapping tables (idempotent)
    reload_dataset(mode)         — Force-reload from disk
    close_dataset()              — Free all resources
    clear_watchmode_cache()      — Invalidate all cached results

Debug:
    dataset_info()               — Status of all mappings and settings
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

# Where the pre-built binary mapping files are hosted.
# These files are downloaded once and cached locally on disk.
_BASE_URL = "https://hcgiub001.github.io/FLAMWM"

# The Watchmode REST API base URL. All API calls go through here.
_WATCHMODE_API_BASE = "https://api.watchmode.com/v1"

# Defines the filenames for each of the three mapping tables.
# Each table needs four files: keys, values, sample (guideposts),
# and meta (JSON with format info like stride and byte widths).
_MAPPING_SPECS = {
    # Maps TMDB movie IDs → Watchmode IDs
    "tmdb2wm_movie": {
        "keys":   "tmdb2wm.movie.tmdb2wm.movie.keys.bin",
        "values": "tmdb2wm.movie.watchmode.movie.values.bin",
        "sample": "tmdb2wm.movie.sample.u32",
        "meta":   "tmdb2wm.movie.meta.json",
    },
    # Maps TMDB TV IDs → Watchmode IDs
    "tmdb2wm_tv": {
        "keys":   "tmdb2wm.tv.tmdb2wm.tv.keys.bin",
        "values": "tmdb2wm.tv.watchmode.tv.values.bin",
        "sample": "tmdb2wm.tv.sample.u32",
        "meta":   "tmdb2wm.tv.meta.json",
    },
    # Maps Watchmode IDs → packed TMDB IDs (both movie and TV in one table)
    "wm2tmdb": {
        "keys":   "wm2tmdb.wm2tmdb.keys.bin",
        "values": "wm2tmdb.packed_tmdb.values.bin",
        "sample": "wm2tmdb.sample.u32",
        "meta":   "wm2tmdb.meta.json",
    },
}

# Collect all unique filenames across all mapping specs.
# Used to check if files need downloading and to compute dataset identity.
_ALL_FILES: Tuple[str, ...] = tuple({
    f for spec in _MAPPING_SPECS.values()
    for f in (spec["meta"], spec["sample"], spec["keys"], spec["values"])
})

# Kodi settings keys for persisting user preferences.
_SETTING_KEY_MODE    = "fenlight.watchmode.mode"      # "auto", "ram", or "mmap"
_SETTING_KEY_API_KEY = "fenlight.watchmode.api_key"    # User's Watchmode API key

# Kodi settings keys for cache TTL (time-to-live) configuration.
_SETTING_KEY_CACHE_DETAILS_HOURS  = "fenlight.watchmode.cache.details_hours"
_SETTING_KEY_CACHE_SIMILAR_HOURS  = "fenlight.watchmode.cache.similar_hours"
_SETTING_KEY_CACHE_NEGATIVE_HOURS = "fenlight.watchmode.cache.negative_hours"

# Default cache durations:
# - Details: 7 days (API responses rarely change)
# - Similar: 1 day  (the assembled bundle of similar titles)
# - Negative: 1 day (remember "no data found" to avoid repeated failures)
_DEFAULT_DETAILS_CACHE_HOURS  = 168   # 7 days
_DEFAULT_SIMILAR_CACHE_HOURS  = 24    # 1 day
_DEFAULT_NEGATIVE_CACHE_HOURS = 24    # 1 day

# Key in the cache database that holds the current cache version string.
# Changing this version effectively invalidates all cached results.
_CACHE_VERSION_KEY   = "watchmode:cache_version"

# How long to wait for an HTTP response before giving up.
_REQUEST_TIMEOUT_S   = 30

# Sentinel value stored in the cache to represent "we looked this up
# and got nothing back." This prevents us from hitting the API again
# for the same failed lookup until the negative cache expires.
_NEGATIVE_SENTINEL   = "WATCHMODE_NEGATIVE"

# How long an idle TCP connection stays in the pool before being closed.
# Set slightly above 60s because most servers close idle connections
# at the 60-second mark.
_CONN_IDLE_TIMEOUT_S = 65

# Memory thresholds for auto-detecting whether to use RAM or mmap mode.
# If available memory is above 200MB AND covers the file sizes plus a
# safety margin, we use RAM mode. Otherwise, mmap.
_AUTO_THRESHOLD_DEFAULT = 200 * 1024 * 1024   # 200 MB minimum available
_SAFETY_MARGIN_MIN      = 64 * 1024 * 1024    # At least 64 MB headroom
_SAFETY_MARGIN_FRAC     = 0.05                 # Or 5% of total RAM

# True if this system is little-endian. The binary files are stored as
# little-endian u32 arrays, so on big-endian systems we need to byteswap.
_IS_LITTLE = sys.byteorder == "little"

# The fields we extract from the Watchmode /details/ response as "extras."
# These provide additional metadata about the title beyond just similar titles.
_EXTRAS_FIELDS = (
    "will_you_like_this",       # AI-generated recommendation text
    "review_summary",           # Aggregated review summary (pros/cons)
    "user_rating",              # Average user rating (float, e.g. 7.4)
    "critic_score",             # Aggregated critic score (int, e.g. 71)
    "relevance_percentile",     # How relevant vs all titles (float, 0-100)
    "popularity_percentile",    # How popular vs all titles (float, 0-100)
)


# ═══════════════════════════════════════════════════════════════════════════
# Logging
#
# Thin wrapper around Kodi's logging system. Silently swallows errors
# so a logging failure never crashes the module.
# ═══════════════════════════════════════════════════════════════════════════
def _log(msg: str, level: int = 0) -> None:
    """Log a message through Kodi's logger. level=0 is debug, level=1 is info."""
    try:
        kodi_log(msg, level)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Pack / Unpack helpers
#
# These functions convert between (tmdb_id, media_type) pairs and
# single packed integers. The packed format stores both pieces of
# information in one number:
#
#   packed = (tmdb_id << 1) | type_bit
#
# where type_bit is 0 for "movie" and 1 for "tv".
#
# This is used as the key format in the binary mapping files and as
# an efficient output format for callers who want minimal overhead.
# ═══════════════════════════════════════════════════════════════════════════
def _is_tv(kind: str) -> bool:
    """Check if the media type string represents a TV show.

    Accepts "tv", "tvshow", "tv_show", etc. — anything starting with "tv".
    """
    return kind.lower().startswith("tv")


def _pack_tmdb(tmdb_id: int, kind: str) -> int:
    """Encode TMDB ID + type into a single int: (tmdb_id << 1) | type_bit.

    Examples:
        _pack_tmdb(278, "movie")  → 556    (278 << 1 | 0)
        _pack_tmdb(1396, "tv")    → 2793   (1396 << 1 | 1)
    """
    return (tmdb_id << 1) | (1 if _is_tv(kind) else 0)


def _unpack_tmdb(packed: int) -> Tuple[int, str]:
    """Decode packed int back to (tmdb_id, media_type).

    Examples:
        _unpack_tmdb(556)  → (278, "movie")
        _unpack_tmdb(2793) → (1396, "tv")
    """
    return packed >> 1, ("tv" if packed & 1 else "movie")


def _tmdb_format_id(tmdb_id: int, kind: str) -> str:
    """Build the TMDB-format ID accepted by Watchmode: 'movie-278' or 'tv-1396'.

    This format is used as a fallback when we don't have a local mapping
    for the TMDB ID. The Watchmode API accepts this format but charges
    2 credits instead of 1.
    """
    return f"tv-{tmdb_id}" if _is_tv(kind) else f"movie-{tmdb_id}"


# ═══════════════════════════════════════════════════════════════════════════
# Cache version tracking
#
# We maintain a "version" string in the cache database. Every cache key
# includes this version string, so when we bump the version (by calling
# clear_watchmode_cache()), all existing cached entries effectively become
# invisible — their keys no longer match the new version.
#
# This is cheaper than actually deleting all cached rows from SQLite.
# ═══════════════════════════════════════════════════════════════════════════
_cache_version: Optional[str] = None
_cache_version_lock = threading.Lock()


def _get_cache_version() -> str:
    """Get the current cache version string.

    On first call, reads it from the database. If not found (fresh install),
    generates one from the current timestamp and stores it.

    Thread-safe via double-checked locking.
    """
    global _cache_version
    if _cache_version is not None:
        return _cache_version
    with _cache_version_lock:
        if _cache_version is not None:
            return _cache_version
        v = main_cache.get(_CACHE_VERSION_KEY)
        if v is None:
            # First run — create an initial version.
            v = str(int(time.time()))
            try:
                main_cache.set(_CACHE_VERSION_KEY, v, expiration=0)
            except Exception:
                pass
        _cache_version = str(v)
        return _cache_version


def clear_watchmode_cache() -> None:
    """Invalidate all cached Watchmode data by bumping the version string.

    Existing cache entries are not deleted — they just become unreachable
    because all future cache keys will contain the new version.
    """
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
#
# Functions to read the user-configurable cache duration (in hours)
# for each type of cached data. Falls back to defaults if the setting
# is missing or invalid.
# ═══════════════════════════════════════════════════════════════════════════
def _get_ttl_hours(setting_key: str, default: int) -> int:
    """Read a cache TTL setting (in hours). Returns default if invalid."""
    try:
        h = int(get_setting(setting_key, str(default)))
        return h if h >= 1 else default
    except Exception:
        return default


def _details_cache_hours() -> int:
    """How long to cache raw Watchmode API responses (default: 7 days)."""
    return _get_ttl_hours(_SETTING_KEY_CACHE_DETAILS_HOURS,
                          _DEFAULT_DETAILS_CACHE_HOURS)


def _similar_cache_hours() -> int:
    """How long to cache assembled bundles (default: 1 day)."""
    return _get_ttl_hours(_SETTING_KEY_CACHE_SIMILAR_HOURS,
                          _DEFAULT_SIMILAR_CACHE_HOURS)


def _negative_cache_hours() -> int:
    """How long to cache 'no data found' results (default: 1 day)."""
    return _get_ttl_hours(_SETTING_KEY_CACHE_NEGATIVE_HOURS,
                          _DEFAULT_NEGATIVE_CACHE_HOURS)


def _hours_to_seconds(hours: int) -> int:
    """Convert hours to seconds for cache expiration."""
    return hours * 3600


# ═══════════════════════════════════════════════════════════════════════════
# Memory detection
#
# Used to decide between RAM mode and mmap mode when set to "auto."
# Tries multiple methods to detect available memory:
#   1. Kodi's built-in System.Memory info labels
#   2. psutil library (if installed)
#   3. /proc/meminfo (Linux/Android)
#   4. Win32 GlobalMemoryStatusEx (Windows)
#
# Returns (available_bytes, total_bytes). Either may be 0 if detection
# fails on all methods.
# ═══════════════════════════════════════════════════════════════════════════
def _parse_kodi_memory_mb(label: str) -> int:
    """Parse a Kodi memory label like '1.5 GB' or '512 MB' into megabytes."""
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
    """Try to get memory info from Kodi's info labels.

    Returns (available_bytes, total_bytes) or None if Kodi isn't available
    or the labels are empty/unparseable.
    """
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
        free_mb = _parse_kodi_memory_mb(free_str)
        if total_mb <= 0 or free_mb <= 0:
            return None
        # Convert MB to bytes: mb << 20 is equivalent to mb * 1048576.
        return free_mb << 20, total_mb << 20
    except Exception:
        return None


def _get_mem_psutil() -> Optional[Tuple[int, int]]:
    """Try to get memory info from the psutil library.

    Returns (available_bytes, total_bytes) or None if psutil isn't installed.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    except Exception:
        return None


def _get_mem_proc() -> Optional[Tuple[int, int]]:
    """Try to get memory info from /proc/meminfo (Linux/Android only).

    Reads MemAvailable (or estimates it from MemFree + Cached + Buffers)
    and MemTotal. Returns (available_bytes, total_bytes) or None.
    """
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    try:
                        # Values in /proc/meminfo are in kB.
                        info[parts[0].strip()] = int(
                            parts[1].strip().split(None, 1)[0]
                        )
                    except (ValueError, IndexError):
                        pass
        avail_kb = info.get("MemAvailable")
        if avail_kb is None:
            # MemAvailable not present (older kernels) — estimate it.
            # The 0.7 factor is a conservative estimate since not all
            # cached/buffered memory is actually reclaimable.
            avail_kb = int(
                (info.get("MemFree", 0)
                 + info.get("Cached", 0)
                 + info.get("Buffers", 0)) * 0.7
            )
        total_kb = info.get("MemTotal", 0)
        if avail_kb <= 0 or total_kb <= 0:
            return None
        # Convert kB to bytes: kb << 10 is equivalent to kb * 1024.
        return avail_kb << 10, total_kb << 10
    except Exception:
        return None


def _get_mem_windows() -> Optional[Tuple[int, int]]:
    """Try to get memory info from Windows API (GlobalMemoryStatusEx).

    Returns (available_bytes, total_bytes) or None if not on Windows
    or the API call fails.
    """
    try:
        # Define the MEMORYSTATUSEX structure that Windows expects.
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
    """Detect available and total system memory using the best available method.

    Returns (available_bytes, total_bytes). Either may be 0 if all
    detection methods fail.

    Tries methods in order of preference:
    1. Kodi info labels (most relevant in a Kodi addon context)
    2. psutil (cross-platform, accurate)
    3. Platform-specific: /proc/meminfo on Linux, Win32 API on Windows
    """
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
# SHA-1 short hash
#
# Used to include the API key in cache keys without exposing the actual
# key. We hash it to a short 10-character hex string. This way,
# different API keys get different cache entries, but the key itself
# is never stored in plaintext in cache key strings.
# ═══════════════════════════════════════════════════════════════════════════
_sha1_cache: Dict[str, str] = {}


def _sha1_short(s: str) -> str:
    """Return a 10-char hex SHA-1 prefix of the input string.

    Results are memoized in a dict so we hash each distinct string
    only once during the process lifetime.
    """
    h = _sha1_cache.get(s)
    if h is None:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        _sha1_cache[s] = h
    return h


# ═══════════════════════════════════════════════════════════════════════════
# Atomic file write
#
# When downloading binary mapping files, we want to avoid leaving
# partially-written files on disk if the process crashes mid-write.
# The strategy is:
#   1. Write to a temporary file in the same directory
#   2. os.replace() atomically swaps it into the target path
#
# os.replace() is atomic on POSIX (it's a rename) and on Windows
# (since Python 3.3). If anything goes wrong, we clean up the temp file.
# ═══════════════════════════════════════════════════════════════════════════
def _atomic_write_bytes(target_path: str, data: bytes) -> None:
    """Write data to target_path atomically (all-or-nothing).

    Uses a temp file + os.replace() to prevent partial/corrupt files.
    """
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    # Create a temp file in the same directory (same filesystem)
    # so os.replace() can be a simple rename operation.
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".tmp_wm",
                               suffix=".bin")
    try:
        os.write(fd, data)
        os.close(fd)
        fd = -1
        # Atomically replace the target file with the temp file.
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
# Proxy detection
#
# Reads proxy settings from the system (environment variables, Windows
# registry, macOS system config) WITHOUT making any HTTP requests.
#
# Uses urllib.request.getproxies() which inspects:
#   • Environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY, etc.)
#   • Windows registry (Internet Settings)
#   • macOS System Configuration framework
#
# Result is cached after the first call. proxy_bypass() performs simple
# string / pattern matching against the no_proxy list — no network I/O.
# ═══════════════════════════════════════════════════════════════════════════
_proxy_map: Optional[Dict[str, str]] = None
_proxy_map_lock = threading.Lock()


def _get_proxies() -> Dict[str, str]:
    """Return the system proxy dict, cached after the first call.

    Example return: {"http": "http://proxy:8080", "https": "http://proxy:8080"}
    Returns {} if no proxies are configured.
    """
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


def _proxy_for(scheme: str,
               host: str) -> Optional[urllib.parse.ParseResult]:
    """Return the parsed proxy URL for the given scheme and target host,
    or None if no proxy applies.

    Returns None when:
    - No proxy is configured for this scheme (http/https)
    - The target host is in the no_proxy / bypass list
    """
    try:
        # Check if this host should bypass the proxy (e.g., localhost).
        if urllib.request.proxy_bypass(host):
            return None
    except Exception:
        pass
    proxies = _get_proxies()
    url = proxies.get(scheme)
    if not url:
        return None
    # Ensure the URL has a scheme so urlparse splits it correctly.
    # (e.g. a bare "proxy.local:8080" would otherwise land in path).
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.hostname:
        return None
    return parsed


def _proxy_auth_header(
    proxy: urllib.parse.ParseResult,
) -> Optional[str]:
    """Build a 'Basic ...' Proxy-Authorization header value from proxy
    credentials, or return None if no username is set in the proxy URL.

    Credentials are expected to be embedded in the proxy URL like:
    http://user:pass@proxy.example.com:8080
    """
    if not proxy.username:
        return None
    cred = "{}:{}".format(
        urllib.parse.unquote(proxy.username),
        urllib.parse.unquote(proxy.password or ""),
    )
    return "Basic " + base64.b64encode(cred.encode()).decode()


# ═══════════════════════════════════════════════════════════════════════════
# HTTP Connection Pool
#
# A thread-safe pool of keep-alive HTTP(S) connections. Benefits:
#
# - Reuses TCP connections: avoids the cost of TCP handshake + TLS
#   negotiation on every request (significant for HTTPS).
#
# - Follow redirects: transparently follows HTTP 3xx redirects up to
#   5 hops (needed for GitHub Pages CDN redirects).
#
# - Proxy support: detects system proxy settings and routes traffic
#   through HTTP proxies (with CONNECT tunneling for HTTPS targets).
#
# - Idle eviction: connections unused for >65 seconds are dropped on
#   the next access. This prevents using stale connections that the
#   server has already closed.
#
# Thread safety: a single lock protects the connection dict. Connections
# are "checked out" (removed from dict) while in use and "returned"
# after the response is read. Only one thread uses a connection at a time.
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
        # Maps (scheme, host, port) → (connection_object, last_used_timestamp)
        # The timestamp uses time.monotonic() for reliable elapsed-time checks.
        self._conns: Dict[
            Tuple[str, str, int],
            Tuple[http.client.HTTPConnection, float],
        ] = {}

    def fetch(
        self, url: str, timeout: int = _REQUEST_TIMEOUT_S
    ) -> Tuple[int, bytes]:
        """Perform a GET request to *url*, following redirects.

        Returns (http_status_code, response_body_bytes).
        Raises IOError if too many redirects are encountered.
        """
        current = url
        for _ in range(self._MAX_REDIRECTS + 1):
            status, body, location = self._get_once(current, timeout)
            if 300 <= status < 400 and location:
                # Server says "go here instead" — follow the redirect.
                current = urllib.parse.urljoin(current, location)
                continue
            return status, body
        raise IOError(f"Too many redirects: {url}")

    def close_all(self) -> None:
        """Close and discard all pooled connections."""
        with self._lock:
            for conn, _ts in self._conns.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()

    # ── internals ─────────────────────────────────────────────────────

    def _get_once(
        self, url: str, timeout: int
    ) -> Tuple[int, bytes, Optional[str]]:
        """Perform a single GET request (no redirect following).

        Returns (status, body, location_header_or_none).
        """
        # Parse the URL into its components.
        parsed = urllib.parse.urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        # Connection pool key — connections to the same (scheme, host, port)
        # are reusable.
        key = (scheme, host, port)

        # ── Proxy detection ──────────────────────────────────────
        proxy = _proxy_for(scheme, host)

        # For plain HTTP through a proxy, the request line must contain
        # the full URL (e.g., "GET http://example.com/path HTTP/1.1")
        # so the proxy knows where to forward the request.
        # For HTTPS through a proxy, we use CONNECT tunneling, and the
        # request line uses the normal path.
        is_http_proxy = proxy is not None and scheme != "https"
        request_path = url if is_http_proxy else path

        headers = {
            "Host":       host,
            "User-Agent": "watchmode_api/2.0",
            "Accept":     "application/json,*/*",
            "Connection": "keep-alive",   # Ask server to keep connection open
        }

        # Add proxy authentication header for plain-HTTP proxy requests.
        if is_http_proxy and proxy is not None:
            auth = _proxy_auth_header(proxy)
            if auth:
                headers["Proxy-Authorization"] = auth

        # ── Try reusing an existing pooled connection ────────────
        conn: Optional[http.client.HTTPConnection] = None
        with self._lock:
            # Remove from pool (we "check it out" while using it).
            entry = self._conns.pop(key, None)
        if entry is not None:
            stored_conn, stored_ts = entry
            # Only reuse if it hasn't been idle too long.
            if (time.monotonic() - stored_ts) <= _CONN_IDLE_TIMEOUT_S:
                conn = stored_conn
            else:
                # Too old — close it and we'll make a fresh one below.
                try:
                    stored_conn.close()
                except Exception:
                    pass

        # Try the reused connection. If it fails (broken pipe, etc.),
        # fall through to creating a new one.
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

        # ── Open a new connection (direct or through proxy) ──────
        conn = self._new_conn(scheme, host, port, timeout, proxy)
        try:
            conn.request("GET", request_path, headers=headers)
            resp = conn.getresponse()
            body = resp.read()
            # Return to pool for future reuse (if server allows it).
            self._return_conn(key, conn, resp)
            return resp.status, body, resp.getheader("Location")
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

    def _return_conn(self, key, conn, resp) -> None:
        """Return a connection to the pool for reuse, unless the server
        indicated it wants to close the connection.

        If the server sent "Connection: close" or the response indicates
        the connection will be closed, we close it immediately instead
        of pooling it.
        """
        conn_header = (resp.getheader("Connection") or "").lower()
        if conn_header == "close" or getattr(resp, "will_close", False):
            try:
                conn.close()
            except Exception:
                pass
            return
        now = time.monotonic()
        with self._lock:
            # If there was already a different connection in the pool for
            # this key (shouldn't normally happen), close the old one.
            old_entry = self._conns.pop(key, None)
            self._conns[key] = (conn, now)
        if old_entry is not None and old_entry[0] is not conn:
            try:
                old_entry[0].close()
            except Exception:
                pass

    @staticmethod
    def _new_conn(
        scheme: str,
        host: str,
        port: int,
        timeout: int,
        proxy: Optional[urllib.parse.ParseResult] = None,
    ) -> http.client.HTTPConnection:
        """Create a new HTTP(S) connection, optionally through *proxy*.

        For HTTPS through a proxy:
          1. Opens a TCP connection to the proxy
          2. Sends a CONNECT request to establish a tunnel to the target
          3. Wraps the tunnel in TLS for end-to-end encryption

        For HTTP through a proxy:
          1. Opens a TCP connection to the proxy
          2. Sends requests with the full URL as the path

        Without a proxy:
          1. Opens a direct TCP (or TLS) connection to the target
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
                # set_tunnel tells HTTPSConnection to issue a CONNECT
                # request to the proxy before starting TLS.
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
            return http.client.HTTPSConnection(
                host, port, timeout=timeout, context=ctx,
            )
        return http.client.HTTPConnection(host, port, timeout=timeout)


# Module-level singleton connection pool used by all HTTP operations.
_http_pool = _ConnectionPool()


# ═══════════════════════════════════════════════════════════════════════════
# File paths
#
# All binary mapping files are stored in the Kodi addon's data directory
# under cache/watchmode/. The path is resolved via Kodi's translate_path()
# which handles platform-specific paths (Android, Windows, Linux, etc.).
# ═══════════════════════════════════════════════════════════════════════════
_cache_dir: Optional[str] = None


def _ensure_cache_dir() -> str:
    """Return the path to the watchmode cache directory, creating it if needed.

    Result is cached so subsequent calls don't hit the filesystem.
    """
    global _cache_dir
    if _cache_dir is not None:
        return _cache_dir
    base = translate_path(
        "special://profile/addon_data/plugin.video.fenlight/"
    )
    d = os.path.join(base, "cache", "watchmode")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    _cache_dir = d
    return d


def _file_path(filename: str) -> str:
    """Return the full filesystem path for a mapping file."""
    return os.path.join(_ensure_cache_dir(), filename)


# ═══════════════════════════════════════════════════════════════════════════
# Download mapping files from GitHub
#
# On first run (or if files are missing), downloads the binary mapping
# files from GitHub Pages. Each file is typically a few MB.
# Uses atomic writes so a crash during download won't leave corrupt files.
# ═══════════════════════════════════════════════════════════════════════════
def _ensure_files_downloaded() -> bool:
    """Download any missing mapping files. Returns True if all files are present.

    Only downloads files that are missing or empty. Already-present files
    are not re-downloaded (there's no update check here — that's handled
    separately by reload_dataset()).
    """
    all_ok = True
    for fname in _ALL_FILES:
        path = _file_path(fname)
        # Skip files that already exist and have content.
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        url = f"{_BASE_URL}/{fname}"
        try:
            status, data = _http_pool.fetch(url)
            if status >= 400:
                raise IOError(f"HTTP {status}")
            _atomic_write_bytes(path, data)
        except Exception as e:
            _log(f"watchmode_api: download failed {fname}: {e}", 1)
            all_ok = False
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# Manual binary search
#
# We implement our own binary search instead of using Python's bisect
# module because bisect uses PySequence_GetItem internally, which can
# cause issues with mmap-backed memoryview objects on some platforms
# (particularly CPython on Android). Our version only does simple
# indexing (keys[mid]) which is safe on both array and memoryview.
# ═══════════════════════════════════════════════════════════════════════════
def _binary_search(keys, key: int, lo: int, hi: int) -> int:
    """Return the index of *key* inside keys[lo:hi], or -1 if absent.

    Uses a plain while-loop so each iteration only does a single
    keys[mid] element access — safe on mmap-backed memoryview objects
    where the C bisect module's PySequence_GetItem can fault pages
    unexpectedly on some CPython/Android builds.

    Works identically on array('I') objects in RAM mode.

    This is a standard binary search: O(log n) comparisons.
    """
    while lo < hi:
        mid = (lo + hi) >> 1  # Same as (lo + hi) // 2, but faster.
        v = keys[mid]
        if v < key:
            lo = mid + 1
        elif v > key:
            hi = mid
        else:
            return mid  # Found it!
    return -1  # Not found.


def _upper_bound(sample, key: int) -> int:
    """Return the index of the first element in *sample* that is > *key*,
    or len(sample) if all elements are ≤ key.

    Equivalent to bisect.bisect_right(sample, key) but safe on memoryview.

    Used with the sample array to find which "stride" (block of keys)
    a given key might be in.
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
# madvise helper
#
# When using mmap, the OS kernel normally expects sequential access and
# prefetches pages ahead of the current read position. But our access
# pattern is random (binary search jumps around), so prefetching wastes
# memory and I/O. MADV_RANDOM tells the kernel "don't bother prefetching."
#
# Available since Python 3.8 on Unix. Silently skipped on older Python
# or Windows (where mmap.MADV_RANDOM doesn't exist).
# ═══════════════════════════════════════════════════════════════════════════
def _madvise_random(mm: mmap.mmap) -> None:
    """Tell the OS kernel that this mmap will be accessed randomly,
    so it should NOT prefetch sequential pages.

    Silently does nothing if MADV_RANDOM is not available (old Python,
    Windows, etc.).
    """
    try:
        mm.madvise(mmap.MADV_RANDOM)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Binary mapping (sorted u32 key→value files, O(log n) lookup)
#
# This is the core data structure. Each mapping table is a pair of sorted
# parallel arrays:
#
#   keys:   [100, 200, 300, 400, 500, ...]   (sorted, ascending)
#   values: [ 10,  20,  30,  40,  50, ...]   (same index = same entry)
#
# To find the value for key 300:
#   1. Binary search the keys array → found at index 2
#   2. Return values[2] → 30
#
# To make binary search faster, we use two acceleration structures:
#
# RAM mode - Direct Map (64K bucket index):
#   A 65537-element array where entry[i] = first index where (key >> 16) >= i.
#   This lets us jump directly to the right neighborhood in O(1), then
#   binary search only a tiny range.
#
#   Example: If we're looking for key 0x0003_ABCD, we look at bucket 3,
#   which tells us "keys with top 16 bits = 3 start at index 47."
#   Bucket 4 says "keys with top 16 bits = 4 start at index 52."
#   So we only binary search indices 47..52 instead of the full array.
#
# mmap mode - Sample array:
#   Every Nth key is stored in a small "sample" array that fits in RAM.
#   First, we binary search the sample to find which stride-sized block
#   contains our key, then binary search only that block (which will
#   page-fault in ~1 page from disk).
#
#   Example with stride=64: If the sample tells us our key is between
#   sample[5] and sample[6], we know the key is somewhere in
#   keys[320..384] (indices 5*64 to 6*64), so we only search there.
# ═══════════════════════════════════════════════════════════════════════════
class PackedMapping:
    """Lookup table backed by sorted binary files of u32 keys and values.

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
        self._keys: Any = None         # array('I') in RAM mode, memoryview in mmap mode
        self._vals: Any = None         # Same as _keys — parallel array of values
        self._sample: Optional[array] = None  # Small array of every Nth key (always in RAM)
        self._meta: Optional[Dict[str, Any]] = None  # Parsed meta.json
        self._count: int = 0           # Total number of key-value pairs
        self._stride: int = 64         # How many keys between each sample entry
        self._direct_map: Optional[array] = None  # 64K bucket index (RAM mode only)
        self._fileobjs: List[Any] = []  # Open file handles (mmap mode, for cleanup)
        self._mmaps: List[Any] = []     # Active mmap objects (mmap mode, for cleanup)
        self._is_mmap: bool = False     # True if loaded in mmap mode

    @property
    def loaded(self) -> bool:
        """True if the mapping data has been loaded and is ready for lookups."""
        return self._keys is not None

    @property
    def count(self) -> int:
        """Number of key-value pairs in this mapping."""
        return self._count

    @property
    def is_mmap(self) -> bool:
        """True if this mapping is using memory-mapped files (vs RAM arrays)."""
        return self._is_mmap

    def close(self) -> None:
        """Release all resources: close mmap objects and file handles,
        clear references to arrays."""
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
        """Load all binary files fully into RAM arrays, then build the
        direct-map index for fast lookups.

        After this call, all data lives in Python array objects — no file
        handles or mmaps are kept open.
        """
        self.close()
        self._is_mmap = False
        self._load_meta(spec)
        self._sample = self._load_u32_file(_file_path(spec["sample"]))
        self._keys = self._load_u32_file(_file_path(spec["keys"]))
        self._vals = self._load_u32_file(_file_path(spec["values"]))
        self._count = len(self._keys)
        self._build_direct_map()

    def load_mmap(self, spec: Dict[str, str]) -> None:
        """Load sample into RAM; memory-map keys and values from disk.

        Only the small sample array (a few KB) lives in RAM. The large
        keys and values arrays are memory-mapped, so the OS loads pages
        on demand as they're accessed during lookups.

        Each mmap is marked MADV_RANDOM so the kernel does not prefetch
        surrounding pages on every random-access fault.

        The direct-map is intentionally NOT built here: constructing it
        requires iterating every key, which would page-fault the entire
        mmap'd file into physical RAM and defeat the purpose of mmap.
        """
        self.close()
        self._is_mmap = True
        self._load_meta(spec)
        self._sample = self._load_u32_file(_file_path(spec["sample"]))
        self._direct_map = None  # explicitly skip — see docstring

        # Memory-map both the keys and values files.
        for attr, key in (("_keys", "keys"), ("_vals", "values")):
            path = _file_path(spec[key])
            size = os.path.getsize(path)
            # Each element is 4 bytes (u32), so align to 4-byte boundary.
            # Any trailing bytes that don't form a complete u32 are ignored.
            aligned = (size // 4) * 4
            if aligned == 0:
                raise ValueError(f"Mapping file too small: {path}")
            # Open the file and memory-map it as read-only.
            fo = open(path, "rb")
            self._fileobjs.append(fo)
            mm = mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ)
            _madvise_random(mm)
            self._mmaps.append(mm)
            # Create a memoryview over the mmap and reinterpret the raw
            # bytes as unsigned 32-bit integers. This lets us index into
            # the mmap as if it were an array of ints: keys[i] returns
            # the i-th u32 value.
            setattr(self, attr, memoryview(mm)[:aligned].cast("I"))

        self._count = len(self._keys)

    def lookup(self, key: int) -> Optional[int]:
        """Find the value for *key*, or return None if not found.

        This is the main entry point for lookups. It:
        1. Narrows the search range using the direct-map (RAM) or sample (mmap)
        2. Binary searches within that narrow range
        3. Returns the value at the found index, or None
        """
        keys = self._keys
        if keys is None:
            return None

        # ── Fast path: direct-map bucket index (RAM mode only) ──
        # The direct map uses the top 16 bits of the key to index into
        # a 64K array that tells us exactly where to start searching.
        dm = self._direct_map
        if dm is not None:
            hb = key >> 16  # Top 16 bits = bucket number (0..65535)
            if hb > 65535:
                return None
            lo = dm[hb]      # First index with this top-16 prefix
            hi = dm[hb + 1]  # First index with the NEXT prefix
        else:
            # ── Fallback: sample-based range narrowing (mmap mode) ──
            # The sample array contains every Nth key. We find which
            # stride-block our key falls into, then only search that block.
            sample = self._sample
            if sample is not None and len(sample) > 0:
                bucket = _upper_bound(sample, key) - 1
                lo = max(0, bucket * self._stride)
                hi = min(lo + self._stride, self._count)
            else:
                # No acceleration available — search the entire array.
                lo, hi = 0, self._count

        if lo >= hi:
            return None

        # Binary search within the narrowed range.
        i = _binary_search(keys, key, lo, hi)
        # If found (i >= 0), return the value at the same index.
        return self._vals[i] if i >= 0 else None

    # ── internals ─────────────────────────────────────────────────

    def _build_direct_map(self) -> None:
        """Build a 64K bucket index over the top 16 bits of each key.

        Only called in RAM mode where all keys are already resident in memory.

        The direct map is an array of 65537 entries (one extra for the sentinel).
        Entry dm[b] = the index of the first key whose top 16 bits >= b.

        Example:
          Keys: [0x0001_0000, 0x0001_0005, 0x0003_0000, 0x0003_FFFF]
          dm[0] = 0  (all keys have top bits >= 0)
          dm[1] = 0  (first key with top bits >= 1 is at index 0)
          dm[2] = 2  (first key with top bits >= 2 is at index 2)
          dm[3] = 2  (first key with top bits >= 3 is at index 2)
          dm[4] = 4  (first key with top bits >= 4 is at index 4 = end)
          ...
          dm[65536] = 4  (sentinel = total count)
        """
        keys = self._keys
        if keys is None or len(keys) == 0:
            self._direct_map = None
            return

        count = len(keys)
        # Initialize all buckets to count (meaning "no keys with this prefix").
        dm = array("I", [count] * 65537)

        # Pass 1: For each key, record the earliest index for its bucket.
        for i in range(count):
            hb = keys[i] >> 16
            if i < dm[hb]:
                dm[hb] = i

        # Pass 2: Backward fill — if a bucket has no keys, it should point
        # to the same place as the next non-empty bucket.
        # This ensures dm[b] is always a valid starting index.
        for hb in range(65535, -1, -1):
            if dm[hb] > dm[hb + 1]:
                dm[hb] = dm[hb + 1]

        # The last entry is always the total count (end sentinel).
        dm[65536] = count
        self._direct_map = dm

    def _load_meta(self, spec: Dict[str, str]) -> None:
        """Load and validate the meta.json file for this mapping.

        The meta file tells us the byte width of keys and values (must be 4)
        and the sample stride (how many keys between each sample entry).
        """
        with open(_file_path(spec["meta"]), "r", encoding="utf-8") as fh:
            self._meta = json.load(fh)
        if (int(self._meta.get("key_bytes", 4)) != 4
                or int(self._meta.get("value_bytes", 4)) != 4):
            raise ValueError("Only 32-bit key/value files supported")
        self._stride = int(self._meta.get("sample_stride", 64))

    @staticmethod
    def _load_u32_file(path: str) -> array:
        """Read a binary file as an array of unsigned 32-bit integers.

        The file is expected to contain little-endian u32 values packed
        contiguously. If this system is big-endian, the array is byteswapped
        after loading.

        Any trailing bytes that don't form a complete u32 are silently
        discarded.
        """
        with open(path, "rb") as fh:
            data = fh.read()
        a = array("I")
        a.frombytes(data[:len(data) // 4 * 4])  # Align to 4-byte boundary
        if not _IS_LITTLE:
            a.byteswap()  # Convert from little-endian to native byte order
        return a


# ═══════════════════════════════════════════════════════════════════════════
# Module state
#
# These module-level variables track whether the mapping tables are loaded,
# which mode they're in, and a fingerprint of the dataset files on disk.
#
# All access is protected by _lock for thread safety.
# ═══════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()

# Dict of mapping name → PackedMapping instance (or None if not yet loaded).
_mappings: Dict[str, Optional[PackedMapping]] = {
    "tmdb2wm_movie": None,  # TMDB movie ID → Watchmode ID
    "tmdb2wm_tv":    None,  # TMDB TV ID → Watchmode ID
    "wm2tmdb":       None,  # Watchmode ID → packed TMDB ID
}

# The mode actually used for loading ("RAM" or "mmap"), set after loading.
_runtime_mode: Optional[str] = None

# A fingerprint string derived from file sizes and modification times.
# Used to detect when files have changed on disk and need reloading.
_dataset_id: Optional[str] = None


def _is_loaded() -> bool:
    """Check if all three mapping tables are loaded and ready for lookups."""
    return all(m is not None and m.loaded for m in _mappings.values())


def _compute_dataset_id() -> str:
    """Compute a fingerprint string for the current set of mapping files.

    Format: "filename:mtime:size|filename:mtime:size|..."
    Changes if any file is updated, replaced, or deleted.
    """
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
    """Close all mapping tables and reset module state.

    Must be called with _lock held.
    """
    global _runtime_mode, _dataset_id
    for name in _mappings:
        if _mappings[name] is not None:
            _mappings[name].close()
            _mappings[name] = None
    _runtime_mode = _dataset_id = None


# ═══════════════════════════════════════════════════════════════════════════
# Settings
#
# Functions to get/set user-configurable settings:
# - mode: "auto" (default), "ram", or "mmap"
# - api_key: The user's Watchmode API key (required for API calls)
# ═══════════════════════════════════════════════════════════════════════════
def get_setting_mode() -> str:
    """Get the user's preferred loading mode: "auto", "RAM", or "mmap"."""
    v = (get_setting(_SETTING_KEY_MODE, "auto") or "auto").strip().lower()
    if v == "ram":
        return "RAM"
    if v == "mmap":
        return "mmap"
    return "auto"


def set_setting_mode(mode: str) -> bool:
    """Set the loading mode preference. Accepts "ram", "mmap", or "auto".

    Returns True if the setting was saved successfully.
    Note: Changing this setting doesn't take effect until the next
    ensure_loaded(force=True) or reload_dataset() call.
    """
    m = (mode or "auto").strip().lower()
    if m not in ("ram", "mmap", "auto"):
        return False
    try:
        set_setting(_SETTING_KEY_MODE, m)
        return True
    except Exception:
        return False


def get_runtime_mode() -> Optional[str]:
    """Get the mode actually used for the currently loaded mappings.

    Returns "RAM", "mmap", or None if mappings are not loaded yet.
    """
    return _runtime_mode


def get_api_key() -> str:
    """Get the stored Watchmode API key, or empty string if not set."""
    try:
        return get_setting(_SETTING_KEY_API_KEY, "") or ""
    except Exception:
        return ""


def set_api_key(key: str) -> bool:
    """Store a new Watchmode API key and invalidate all cached results.

    We clear the cache because results from different API keys should not
    be mixed (different accounts may have different data access levels).

    Returns True if the key was saved successfully.
    """
    try:
        set_setting(_SETTING_KEY_API_KEY, (key or "").strip())
        clear_watchmode_cache()
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle (load / reload / close)
#
# These functions manage the loading and unloading of mapping tables.
# ensure_loaded() is the main entry point — it's called automatically
# by every public query function, so callers don't need to call it
# explicitly (though they can if they want to pre-load during startup).
# ═══════════════════════════════════════════════════════════════════════════
def ensure_loaded(mode: Optional[str] = None, force: bool = False) -> None:
    """Make sure all three mapping tables are loaded.

    Thread-safe and idempotent — safe to call from multiple threads,
    and calling it when already loaded is a no-op (unless force=True).

    Args:
        mode:  Override the loading mode ("RAM" or "mmap"). If None,
               reads from settings (default: "auto").
        force: If True, reload even if already loaded. Used after
               updating the mapping files on disk.
    """
    global _runtime_mode, _dataset_id

    # Quick check without locking — if already loaded and not forced, done.
    if _is_loaded() and not force:
        return

    with _lock:
        # Double-checked locking: re-check inside the lock because another
        # thread may have loaded while we were waiting for the lock.
        if _is_loaded() and not force:
            return

        # Step 1: Make sure all binary files exist on disk.
        # Downloads any that are missing from GitHub.
        _ensure_files_downloaded()

        chosen = mode if mode else get_setting_mode()

        # Auto-select RAM vs mmap based on available memory.
        if chosen == "auto":
            # Calculate total size of all mapping files.
            total_file_size = sum(
                os.path.getsize(_file_path(f))
                for f in _ALL_FILES
                if os.path.isfile(_file_path(f))
            )
            # Detect available system memory.
            avail, total = _get_available_memory()
            # Calculate safety margin: at least 64MB, or 5% of total RAM.
            safety = (
                max(_SAFETY_MARGIN_MIN,
                    int(total * _SAFETY_MARGIN_FRAC))
                if total
                else _SAFETY_MARGIN_MIN
            )
            # Use RAM mode only if we have enough memory above both the
            # absolute minimum (200MB) and the file sizes + safety margin.
            chosen = (
                "RAM"
                if (avail >= _AUTO_THRESHOLD_DEFAULT
                    and avail >= total_file_size + safety)
                else "mmap"
            )

        # Step 2: Load each mapping table in the chosen mode.
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
        _dataset_id = _compute_dataset_id()


def reload_dataset(mode: Optional[str] = None) -> None:
    """Force-reload all mapping tables from disk and clear all caches.

    Call this after updating the binary files on disk (e.g., after
    downloading a newer version of the mappings).
    """
    with _lock:
        _close_all_mappings()
    clear_watchmode_cache()
    ensure_loaded(mode=mode, force=True)


def close_dataset() -> None:
    """Free all resources: close mapping tables and HTTP connections.

    Call this during addon shutdown to cleanly release memory, file
    handles, and network sockets.
    """
    with _lock:
        _close_all_mappings()
    _http_pool.close_all()


# ═══════════════════════════════════════════════════════════════════════════
# Local mapping lookups
#
# These are thin wrappers around PackedMapping.lookup() that handle:
# 1. Auto-loading the mapping tables if not yet loaded
# 2. Selecting the correct mapping table based on media type
# 3. Packing/unpacking the TMDB ID format
# ═══════════════════════════════════════════════════════════════════════════
def lookup_tmdb_to_watchmode(tmdb_id: int, kind: str) -> Optional[int]:
    """Look up a TMDB ID + media type in the local mapping → Watchmode ID.

    Returns the Watchmode ID (int) if found, or None if this TMDB ID
    isn't in our local mapping.

    Example:
        lookup_tmdb_to_watchmode(278, "movie")  → 345534 (or None)
    """
    if not _is_loaded():
        ensure_loaded()
    # Select the movie or TV mapping table based on the media type.
    table = _mappings[
        "tmdb2wm_tv" if _is_tv(kind) else "tmdb2wm_movie"
    ]
    # Pack the TMDB ID (includes type bit) and look it up.
    return table.lookup(_pack_tmdb(tmdb_id, kind)) if table else None


def lookup_watchmode_to_tmdb_packed(
    watchmode_id: int,
) -> Optional[int]:
    """Look up a Watchmode ID in the local mapping → packed TMDB int.

    Returns a packed integer encoding both the TMDB ID and media type:
        packed = (tmdb_id << 1) | type_bit

    Use _unpack_tmdb(packed) to decode it, or just:
        tmdb_id    = packed >> 1
        media_type = "tv" if (packed & 1) else "movie"

    Returns None if this Watchmode ID isn't in our mapping.
    """
    if not _is_loaded():
        ensure_loaded()
    table = _mappings.get("wm2tmdb")
    return table.lookup(watchmode_id) if table else None


# ═══════════════════════════════════════════════════════════════════════════
# Watchmode API calls
#
# This section handles all communication with the Watchmode REST API:
# - Building URLs and query strings
# - Caching responses in SQLite (positive and negative caching)
# - Deduplicating in-flight requests (so two threads asking for the same
#   title don't make two API calls)
# - Rate-limit retry (if we get HTTP 429, wait 2s and try once more)
# - Auth error detection (HTTP 401/403 → raise PermissionError)
# ═══════════════════════════════════════════════════════════════════════════
def _sorted_query_string(params: Dict[str, Any]) -> str:
    """Build a URL-encoded query string with parameters sorted by key.

    Sorting ensures the same parameters always produce the same string,
    which is important for cache key consistency.
    """
    return urllib.parse.urlencode(
        sorted(
            (str(k), str(v))
            for k, v in params.items()
            if v is not None
        ),
        doseq=True,
    )


def _make_cache_key(
    namespace: str,
    endpoint: str,
    params: Dict[str, Any],
    api_key: str,
) -> str:
    """Build a deterministic cache key for an API request.

    The cache key includes:
    - The cache version (so clearing the cache invalidates all keys)
    - A hash of the API key (so different accounts get separate caches)
    - The endpoint and parameters

    Example: "wnapi:details:1700000000:a1b2c3d4e5:/title/345534/details/:"
    """
    v = _get_cache_version()
    return (
        f"wnapi:{namespace}:{v}:{_sha1_short(api_key)}"
        f":{endpoint}:{_sorted_query_string(params)}"
    )


# In-flight deduplication:
# If two threads request the same URL simultaneously, only one should
# actually make the HTTP call. The other should wait and use the cached
# result. We achieve this with a per-cache-key lock.
_inflight_lock = threading.Lock()       # Protects the _inflight dict itself
_inflight: Dict[str, threading.Lock] = {}  # Maps cache_key → per-request lock


def _cached_api_get(
    url: str,
    cache_key: str,
    ttl_hours: int,
    neg_hours: int,
    bypass: bool = False,
) -> Any:
    """Fetch JSON from *url* with SQLite caching and negative-caching.

    Flow:
    1. Check the cache — if found, return immediately (cache hit).
       If the cached value is the negative sentinel, return None.
    2. Acquire a per-key lock (dedup: only one thread fetches at a time).
    3. Re-check the cache (another thread may have just filled it).
    4. Make the HTTP request.
    5. On success: parse JSON, store in cache with ttl_hours expiry.
    6. On failure: store negative sentinel with neg_hours expiry.
    7. Return the parsed data, or None on failure.
    """

    # ── Step 1: Check cache (fast path, no locking) ────────────
    if not bypass:
        try:
            cached = main_cache.get(cache_key)
            if cached is not None:
                # Negative sentinel means "we previously tried and got nothing."
                return (
                    None if cached == _NEGATIVE_SENTINEL else cached
                )
        except Exception:
            pass

    # ── Step 2: Acquire per-key lock for deduplication ─────────
    # Multiple threads asking for the same URL will queue up here.
    with _inflight_lock:
        lk = _inflight.get(cache_key)
        if lk is None:
            lk = threading.Lock()
            _inflight[cache_key] = lk

    with lk:
        # ── Step 3: Re-check cache (double-checked locking) ───
        # While we were waiting for the lock, another thread may have
        # fetched the data and cached it.
        if not bypass:
            try:
                cached = main_cache.get(cache_key)
                if cached is not None:
                    with _inflight_lock:
                        _inflight.pop(cache_key, None)
                    return (
                        None
                        if cached == _NEGATIVE_SENTINEL
                        else cached
                    )
            except Exception:
                pass

        # ── Step 4: Make the HTTP request ──────────────────────
        try:
            status, body = _http_pool.fetch(
                url, timeout=_REQUEST_TIMEOUT_S
            )
        except Exception:
            with _inflight_lock:
                _inflight.pop(cache_key, None)
            return None

        # ── Step 4b: Simple retry on rate-limit (HTTP 429) ─────
        # Watchmode's rate limiter returns 429 when we've made too many
        # requests. Wait 2 seconds and try once more.
        if status == 429:
            time.sleep(2)
            try:
                status, body = _http_pool.fetch(
                    url, timeout=_REQUEST_TIMEOUT_S
                )
            except Exception:
                with _inflight_lock:
                    _inflight.pop(cache_key, None)
                return None

        # ── Step 4c: Auth errors are raised as exceptions ──────
        # 401/403 typically means the API key is invalid or expired.
        if status in (401, 403):
            with _inflight_lock:
                _inflight.pop(cache_key, None)
            raise PermissionError(
                f"Watchmode API auth error HTTP {status}"
            )

        # ── Step 5: Handle other HTTP errors ───────────────────
        # For any other error (404, 500, etc.), store a negative sentinel
        # so we don't retry this exact request for neg_hours.
        if status >= 400:
            try:
                main_cache.set(
                    cache_key,
                    _NEGATIVE_SENTINEL,
                    expiration=_hours_to_seconds(neg_hours),
                )
            except Exception:
                pass
            with _inflight_lock:
                _inflight.pop(cache_key, None)
            return None

        # ── Step 6: Parse the JSON response ────────────────────
        try:
            data = json.loads(body)
        except Exception:
            # Unparseable response — treat as an error.
            try:
                main_cache.set(
                    cache_key,
                    _NEGATIVE_SENTINEL,
                    expiration=_hours_to_seconds(neg_hours),
                )
            except Exception:
                pass
            with _inflight_lock:
                _inflight.pop(cache_key, None)
            return None

        # ── Step 7: Cache the successful response ──────────────
        try:
            main_cache.set(
                cache_key,
                data,
                expiration=_hours_to_seconds(ttl_hours),
            )
        except Exception:
            pass

        # Clean up the dedup lock — this request is done.
        with _inflight_lock:
            _inflight.pop(cache_key, None)
        return data


def _call_api(
    endpoint: str,
    params: Dict[str, Any],
    api_key: str,
    namespace: str,
    ttl_hours: int,
    bypass: bool = False,
) -> Optional[Dict[str, Any]]:
    """Low-level Watchmode API GET.

    Builds the full URL (base + endpoint + parameters including API key),
    constructs a cache key (WITHOUT the API key in plaintext), and
    delegates to _cached_api_get().

    Args:
        endpoint:  API path, e.g. "/title/345534/details/"
        params:    Query parameters (excluding apiKey, which is added automatically)
        api_key:   The Watchmode API key
        namespace: Cache namespace (e.g. "details") for key grouping
        ttl_hours: How long to cache a successful response
        bypass:    If True, skip cache reads (but still write to cache)

    Returns the parsed JSON dict, or None on failure.
    """
    # Ensure endpoint starts with '/'.
    ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    # Add the API key to the URL parameters (but NOT to the cache key,
    # where we use a hash of it instead).
    url_params = dict(params)
    url_params["apiKey"] = api_key
    url = (
        f"{_WATCHMODE_API_BASE}{ep}"
        f"?{_sorted_query_string(url_params)}"
    )

    # Build cache key (uses hashed API key, not the real one).
    ck = _make_cache_key(namespace, ep, params, api_key)
    data = _cached_api_get(
        url, ck, ttl_hours, _negative_cache_hours(), bypass=bypass
    )
    return data if isinstance(data, dict) else None


def _fetch_details(
    title_id: Union[int, str],
    api_key: str,
    bypass_cache: bool = False,
) -> Optional[Dict[str, Any]]:
    """Fetch /title/{id}/details/ from Watchmode API.

    This endpoint returns comprehensive information about a title including
    similar_titles (list of Watchmode IDs) and various metadata fields.

    *title_id* may be:
    - A numeric Watchmode ID (e.g. 345534) — costs 1 API credit
    - An IMDB ID string (e.g. "tt0903747") — costs 2 API credits
    - A TMDB-format ID string (e.g. "movie-278" or "tv-1396") — costs 2 credits

    We prefer numeric Watchmode IDs (from our local mapping) to save credits.
    """
    return _call_api(
        f"/title/{title_id}/details/",
        {},
        api_key,
        namespace="details",
        ttl_hours=_details_cache_hours(),
        bypass=bypass_cache,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Core: bundle function (similar titles + extras in one API call)
#
# This is the SINGLE WORKHORSE of the entire module. Every public query
# function below delegates to get_watchmode_bundle_for_addon() and then
# returns whichever slice of the result the caller asked for.
#
# Because the bundle is cached, the second call for the same title is
# an instant dict lookup — no network, no computation.
#
# The flow:
#   1. Check if the bundle is already cached → return immediately
#   2. Look up TMDB ID → Watchmode ID (local binary mapping, ~microseconds)
#   3. Call Watchmode API for /title/{id}/details/ (HTTP, ~200-500ms)
#   4. Extract "extras" fields from the response (dict lookups)
#   5. Extract similar_titles (list of Watchmode IDs) from the response
#   6. Convert each Watchmode ID → TMDB ID (local binary mapping)
#   7. Assemble the bundle dict and cache it
# ═══════════════════════════════════════════════════════════════════════════
def _bundle_cache_key(tmdb_id: int, kind: str, api_key: str) -> str:
    """Build a cache key for a complete bundle (similar titles + extras).

    Includes the dataset ID (file fingerprint) so that bundles are
    automatically invalidated when mapping files are updated.
    """
    v = _get_cache_version()
    did = _dataset_id or "-"
    return (
        f"wn:bundle:{v}:{did}:{_sha1_short(api_key)}"
        f":{tmdb_id}:{kind}"
    )


def _empty_bundle() -> Dict[str, Any]:
    """Return a bundle with no results and None for every extras field.

    Used as the return value when the API call fails or no API key is set.
    """
    return {
        "results":          [],
        "total_results":    0,
        "watchmode_extras": {f: None for f in _EXTRAS_FIELDS},
    }


def get_watchmode_bundle_for_addon(
    tmdb_id: int,
    kind: str,
    api_key: Optional[str] = None,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """One-stop call: returns similar titles AND watchmode extras.

    This is the MAIN function of the module. All other public query
    functions are thin wrappers around this one.

    The first call for a given (tmdb_id, kind) makes one HTTP request
    to the Watchmode API. All subsequent calls return the cached result.

    Args:
        tmdb_id:      The TMDB ID of the title (e.g. 278 for The Shawshank Redemption)
        kind:         "movie" or "tv" (or anything starting with "tv")
        api_key:      Optional override; if None, reads from settings
        bypass_cache: If True, skip reading from cache (still writes to cache)

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
            "popularity_percentile": 98.703,
        }
    }

    If the API call fails, "results" will be [] and every extras field
    will be None.
    """
    # Make sure mapping tables are loaded (no-op if already loaded).
    if not _is_loaded():
        ensure_loaded()

    # Use provided API key or fall back to the stored one.
    key = api_key or get_api_key()
    if not key:
        return _empty_bundle()

    cache_key = _bundle_cache_key(tmdb_id, kind, key)

    # ── Check bundle cache (fast path) ────────────────────────
    if not bypass_cache:
        try:
            cached = main_cache.get(cache_key)
            if cached is not None:
                return (
                    cached
                    if isinstance(cached, dict)
                    else _empty_bundle()
                )
        except Exception:
            pass

    neg_hours = _negative_cache_hours()
    ttl = _similar_cache_hours()

    # ── Step 1: TMDB → Watchmode title identifier (local) ─────
    # Try to find the Watchmode ID in our local binary mapping.
    # If found: we can use the numeric ID (costs 1 API credit).
    # If not found: we fall back to the TMDB-format string like
    # "movie-278" (costs 2 API credits but always works).
    wm_id: Optional[int] = lookup_tmdb_to_watchmode(tmdb_id, kind)
    title_id: Union[int, str] = (
        wm_id
        if wm_id is not None
        else _tmdb_format_id(tmdb_id, kind)
    )

    # ── Step 2: Call Watchmode API (the only network wait) ─────
    try:
        details = _fetch_details(
            title_id, key, bypass_cache=bypass_cache
        )
    except Exception:
        details = None

    # If the API call failed, cache and return an empty bundle.
    if not isinstance(details, dict):
        empty = _empty_bundle()
        try:
            main_cache.set(
                cache_key,
                empty,
                expiration=_hours_to_seconds(neg_hours),
            )
        except Exception:
            pass
        return empty

    # ── Step 3: Extract extras from the API response ──────────
    # These are simple dict.get() calls on the API response — instant.
    extras: Dict[str, Any] = {}
    for field in _EXTRAS_FIELDS:
        extras[field] = details.get(field)

    # ── Step 4: Extract similar_titles & convert to TMDB IDs ──
    # The API response contains "similar_titles": [wm_id1, wm_id2, ...].
    # We convert each Watchmode ID back to a TMDB ID + media type using
    # the local wm2tmdb mapping table.
    similar_ids = details.get("similar_titles")
    results: List[Dict[str, Any]] = []

    if isinstance(similar_ids, list) and similar_ids:
        wm2tmdb = _mappings.get("wm2tmdb")
        if wm2tmdb is not None and wm2tmdb.loaded:
            lookup = wm2tmdb.lookup  # Cache the method reference for speed
            for wid in similar_ids:
                try:
                    # Look up this Watchmode ID → packed TMDB int.
                    packed = lookup(int(wid))
                except (TypeError, ValueError):
                    continue
                if packed is not None:
                    # Unpack: high bits = TMDB ID, lowest bit = type flag.
                    results.append({
                        "id":         packed >> 1,
                        "media_type": "tv" if packed & 1 else "movie",
                    })

    # ── Step 5: Package everything into a bundle & cache it ───
    bundle: Dict[str, Any] = {
        "results":          results,
        "total_results":    len(results),
        "watchmode_extras": extras,
    }

    try:
        main_cache.set(
            cache_key,
            bundle,
            expiration=_hours_to_seconds(ttl),
        )
    except Exception:
        pass

    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrappers
#
# All of these call get_watchmode_bundle_for_addon() and then return
# just the slice the caller wants. Because the bundle is cached after
# the first call, these are essentially free dict operations.
# ═══════════════════════════════════════════════════════════════════════════
def get_watchmode_for_addon(
    tmdb_id: int, kind: str, **kwargs
) -> Dict[str, Any]:
    """Similar titles only (no extras).

    Returns
    -------
    {"results": [{"id": …, "media_type": …}, ...], "total_results": N}
    """
    b = get_watchmode_bundle_for_addon(tmdb_id, kind, **kwargs)
    return {
        "results":       b["results"],
        "total_results": b["total_results"],
    }


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
    return get_watchmode_bundle_for_addon(
        tmdb_id, kind, **kwargs
    )["watchmode_extras"]


def query_watchmode_packed(
    tmdb_id: int, kind: str, **kwargs
) -> List[int]:
    """Similar titles as packed ints: (tmdb_id << 1) | type_bit.

    This is the fastest output format — no dicts allocated per item.
    type_bit: 0 = movie, 1 = tv.

    Example return: [556, 2793, ...]
    Decode: tmdb_id = packed >> 1, media_type = "tv" if packed & 1 else "movie"
    """
    b = get_watchmode_bundle_for_addon(tmdb_id, kind, **kwargs)
    return [
        (r["id"] << 1) | (1 if r["media_type"] == "tv" else 0)
        for r in b["results"]
    ]


def query_watchmode_pairs(
    tmdb_id: int, kind: str, **kwargs
) -> Dict[str, Any]:
    """Similar titles as [tmdb_id, type_bit] pairs.

    Returns {"count": N, "results": [[tmdb_id, type_bit], ...]}
    type_bit: 0 = movie, 1 = tv.

    Example: {"count": 2, "results": [[278, 0], [1396, 1]]}
    """
    packed = query_watchmode_packed(tmdb_id, kind, **kwargs)
    return {
        "count":   len(packed),
        "results": [[p >> 1, p & 1] for p in packed],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Debug / status
#
# Returns a snapshot of the module's current state. Useful for debugging
# issues like "why aren't my lookups working?" or "is mmap mode active?"
# ═══════════════════════════════════════════════════════════════════════════
def dataset_info() -> Dict[str, Any]:
    """Return a dict describing the current state of all mappings.

    Includes:
    - Whether each mapping is loaded, its size, and its mode
    - The overall runtime mode ("RAM" or "mmap")
    - Whether an API key is configured
    - Whether a system proxy is detected
    - The dataset fingerprint (changes when files are updated)
    """
    mapping_info = {}
    for name, m in _mappings.items():
        if m is not None and m.loaded:
            mapping_info[name] = {
                "loaded":  True,
                "count":   m.count,
                "is_mmap": m.is_mmap,
            }
        else:
            mapping_info[name] = {"loaded": False}
    return {
        "loaded":           _is_loaded(),
        "dataset_id":       _dataset_id,
        "runtime_mode":     _runtime_mode,
        "api_key_set":      bool(get_api_key()),
        "proxy_configured": bool(
            _get_proxies().get("http")
            or _get_proxies().get("https")
        ),
        "mappings":         mapping_info,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API REFERENCE & USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════
#
# All five public query functions ultimately call
# get_watchmode_bundle_for_addon(), which:
#   1. Looks up TMDB→Watchmode ID via local binary mapping (instant)
#   2. Calls Watchmode /title/{id}/details/ (one HTTP call, cached)
#   3. Extracts similar_titles + extras from the response (instant)
#   4. Converts Watchmode IDs → TMDB IDs via local mapping (instant)
#   5. Caches the assembled bundle (instant)
#
# The FIRST call for a given (tmdb_id, kind) pays the network cost.
# Every subsequent call hits the cache and returns instantly.
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 1: get_watchmode_bundle_for_addon (RECOMMENDED — everything)
# ───────────────────────────────────────────────────────────────────────
#   from modules.watchmode_api import get_watchmode_bundle_for_addon
#   bundle = get_watchmode_bundle_for_addon(8337, "movie")
#
#   # ── Extras: available IMMEDIATELY, same response ──────────
#   extras = bundle["watchmode_extras"]
#   extras["will_you_like_this"]     # str or None
#   extras["review_summary"]         # str or None
#   extras["user_rating"]            # float or None
#   extras["critic_score"]           # int or None
#   extras["relevance_percentile"]   # float or None
#   extras["popularity_percentile"]  # float or None
#
#   # ── Similar titles ────────────────────────────────────────
#   bundle["total_results"]   # 25
#   bundle["results"]         # [{"id": 278, "media_type": "movie"}, ...]
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 2: get_watchmode_for_addon (similar titles only)
# ───────────────────────────────────────────────────────────────────────
#   from modules.watchmode_api import get_watchmode_for_addon
#   data = get_watchmode_for_addon(8337, "movie")
#   data["total_results"]   # 25
#   data["results"]         # [{"id": 278, "media_type": "movie"}, ...]
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 3: get_watchmode_extras (extras only)
# ───────────────────────────────────────────────────────────────────────
#   from modules.watchmode_api import get_watchmode_extras
#   extras = get_watchmode_extras(8337, "movie")
#   extras["will_you_like_this"]   # str or None
#   extras["review_summary"]       # str or None
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 4: query_watchmode_packed (packed ints)
# ───────────────────────────────────────────────────────────────────────
#   from modules.watchmode_api import query_watchmode_packed
#   packed_list = query_watchmode_packed(8337, "movie")
#   # [556, 2793, ...]  ← (tmdb_id << 1) | type_bit
#
# ───────────────────────────────────────────────────────────────────────
# FUNCTION 5: query_watchmode_pairs ([id, bit] pairs)
# ───────────────────────────────────────────────────────────────────────
#   from modules.watchmode_api import query_watchmode_pairs
#   data = query_watchmode_pairs(8337, "movie")
#   data["count"]     # 25
#   data["results"]   # [[278, 0], [1396, 1], ...]
#
# ───────────────────────────────────────────────────────────────────────
# OTHER PUBLIC FUNCTIONS (settings, lifecycle, debug)
# ───────────────────────────────────────────────────────────────────────
#   get_api_key()                              → str
#   set_api_key(key)                           → bool
#   get_setting_mode()                         → str
#   set_setting_mode(mode)                     → bool
#   get_runtime_mode()                         → str | None
#   ensure_loaded(mode, force)                 → None
#   reload_dataset(mode)                       → None
#   close_dataset()                            → None
#   clear_watchmode_cache()                    → None
#   dataset_info()                             → dict
#   lookup_tmdb_to_watchmode(tmdb_id, kind)    → int | None
#   lookup_watchmode_to_tmdb_packed(wm_id)     → int | None
# ═══════════════════════════════════════════════════════════════════════════