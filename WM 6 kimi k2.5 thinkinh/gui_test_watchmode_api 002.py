#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui_test_watchmode_api.py — Desktop tester for watchmode_api.py.

Place alongside watchmode_api.py.

Run:  python gui_test_watchmode_api.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict

# Settings persistence filename (next to this script)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_FILE = os.path.join(_THIS_DIR, ".watchmode_settings.json")

# Default cache location
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".watchmode_cache")


# ---------------------------------------------------------------------
# Import watchmode_api with robust stubbing fallback
# ---------------------------------------------------------------------
def import_watchmode_api_with_stubs() -> Any:
    try:
        import watchmode_api  # type: ignore
        return watchmode_api
    except Exception as exc:
        if not os.path.exists(os.path.join(_THIS_DIR, "watchmode_api.py")):
            raise RuntimeError(
                "watchmode_api.py not found in current directory. "
                "Place your watchmode_api.py here before running this tester."
            ) from exc
        print("Initial import of watchmode_api failed; installing stubs for testing. Error:", exc)

    # ---- lightweight stubs ------------------------------------------------
    class SimpleCache:
        def __init__(self):
            self._d: Dict[str, Any] = {}
        def get(self, k, default=None):
            return self._d.get(k, default)
        def set(self, k, v, expiration=0):
            self._d[k] = v
        def delete(self, k):
            try:
                del self._d[k]
            except KeyError:
                pass

    def _load_settings_file() -> Dict[str, str]:
        try:
            if os.path.exists(_SETTINGS_FILE):
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_settings_file(d: Dict[str, str]) -> None:
        try:
            with open(_SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(d, fh)
        except Exception:
            pass

    _SETTINGS_STORE: Dict[str, str] = _load_settings_file()

    def get_setting_stub(key: str, default: str = "") -> str:
        return _SETTINGS_STORE.get(key, default)

    def set_setting_stub(key: str, value: str) -> None:
        _SETTINGS_STORE[key] = str(value)
        _save_settings_file(_SETTINGS_STORE)

    def translate_path_stub(path: str) -> str:
        if path.startswith("special://profile"):
            base = os.path.expanduser("~")
            tail = path[len("special://profile"):].lstrip("/\\")
            return os.path.join(base, tail)
        return path

    def kodi_dialog_stub(*args, **kwargs):
        print("kodi_dialog:", args, kwargs)

    def kodi_log_stub(msg: str, level: int = 0):
        print(f"KODI_LOG[{level}]: {msg}")

    import types

    caches_pkg = types.ModuleType("caches")
    caches_main_mod = types.ModuleType("caches.main_cache")
    caches_main_mod.main_cache = SimpleCache()
    caches_settings_mod = types.ModuleType("caches.settings_cache")
    caches_settings_mod.get_setting = get_setting_stub
    caches_settings_mod.set_setting = set_setting_stub

    modules_pkg = types.ModuleType("modules")
    kodi_mod = types.ModuleType("modules.kodi_utils")
    kodi_mod.translate_path = translate_path_stub
    kodi_mod.kodi_dialog = kodi_dialog_stub
    kodi_mod.kodi_log = kodi_log_stub

    sys.modules["caches"] = caches_pkg
    sys.modules["caches.main_cache"] = caches_main_mod
    sys.modules["caches.settings_cache"] = caches_settings_mod
    sys.modules["modules"] = modules_pkg
    sys.modules["modules.kodi_utils"] = kodi_mod

    try:
        watchmode_api = importlib.import_module("watchmode_api")
        print("Imported watchmode_api with stubs (testing mode).")
        return watchmode_api
    except Exception as exc2:
        raise RuntimeError(f"Could not import watchmode_api even after installing stubs: {exc2}") from exc2


try:
    wm = import_watchmode_api_with_stubs()
except Exception as e:
    raise SystemExit(f"Failed to import watchmode_api.py for testing: {e}")


# ---------------------------------------------------------------------
# Helper to read persisted mode
# ---------------------------------------------------------------------
def read_persisted_mode() -> str:
    try:
        if hasattr(wm, "get_setting_mode"):
            m = wm.get_setting_mode()
            if isinstance(m, str) and m:
                return m
    except Exception:
        pass
    return "auto"


def read_persisted_api_key() -> str:
    try:
        if hasattr(wm, "get_api_key"):
            return wm.get_api_key() or ""
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Watchmode API Tester")
        root.geometry("1000x720")

        self.api = wm

        # ---- Top frame: API key ----
        api_frame = ttk.LabelFrame(root, text="Watchmode API Key")
        api_frame.pack(fill="x", padx=8, pady=(8, 4))

        self.api_key_var = tk.StringVar(value=read_persisted_api_key())
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        self.api_key_entry.pack(side="left", padx=4, pady=4)

        self.show_key_var = tk.BooleanVar(value=False)
        self.show_key_cb = ttk.Checkbutton(
            api_frame, text="Show", variable=self.show_key_var, command=self._toggle_key_visibility
        )
        self.show_key_cb.pack(side="left", padx=4)

        self.save_key_btn = ttk.Button(api_frame, text="Save Key", command=self.on_save_key)
        self.save_key_btn.pack(side="left", padx=4)

        ttk.Label(api_frame, text="(Required for API calls)", foreground="gray").pack(side="left", padx=8)

        # ---- Controls frame ----
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(ctrl_frame, text="TMDB ID:").grid(row=0, column=0, sticky="w")
        self.tmdb_id_var = tk.StringVar(value="")
        ttk.Entry(ctrl_frame, textvariable=self.tmdb_id_var, width=12).grid(row=0, column=1, sticky="w")

        ttk.Label(ctrl_frame, text="Type:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.type_var = tk.StringVar(value="movie")
        ttk.Combobox(ctrl_frame, textvariable=self.type_var, values=("movie", "tv"),
                      width=8, state="readonly").grid(row=0, column=3, sticky="w")

        ttk.Label(ctrl_frame, text="WM ID:").grid(row=0, column=4, sticky="w", padx=(10, 0))
        self.wm_id_var = tk.StringVar(value="")
        ttk.Entry(ctrl_frame, textvariable=self.wm_id_var, width=12).grid(row=0, column=5, sticky="w")

        ttk.Label(ctrl_frame, text="Mode:").grid(row=0, column=6, sticky="w", padx=(10, 0))
        init_mode = read_persisted_mode()
        if init_mode not in ("auto", "RAM", "mmap"):
            init_mode = "auto"
        self.mode_var = tk.StringVar(value=init_mode)
        self.mode_combo = ttk.Combobox(
            ctrl_frame, textvariable=self.mode_var, values=("auto", "RAM", "mmap"),
            width=10, state="readonly"
        )
        self.mode_combo.grid(row=0, column=7, sticky="w", padx=(4, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_selected)

        # ---- Buttons frame ----
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=8, pady=4)

        self.load_btn = ttk.Button(btn_frame, text="Load Mappings", command=self.on_load)
        self.load_btn.pack(side="left", padx=4)

        self.tmdb_to_wm_btn = ttk.Button(btn_frame, text="TMDB → WM", command=self.on_tmdb_to_wm)
        self.tmdb_to_wm_btn.pack(side="left", padx=4)

        self.wm_to_tmdb_btn = ttk.Button(btn_frame, text="WM → TMDB", command=self.on_wm_to_tmdb)
        self.wm_to_tmdb_btn.pack(side="left", padx=4)

        self.details_btn = ttk.Button(btn_frame, text="WM Details (API)", command=self.on_details)
        self.details_btn.pack(side="left", padx=4)

        self.similar_btn = ttk.Button(btn_frame, text="Full Similar Pipeline", command=self.on_similar)
        self.similar_btn.pack(side="left", padx=4)

        self.reload_btn = ttk.Button(btn_frame, text="Reload", command=self.on_reload)
        self.reload_btn.pack(side="left", padx=4)

        self.clear_btn = ttk.Button(btn_frame, text="Clear Cache", command=self.on_clear)
        self.clear_btn.pack(side="left", padx=4)

        self.info_btn = ttk.Button(btn_frame, text="Info", command=self.on_info)
        self.info_btn.pack(side="left", padx=4)

        # ---- Second button row: connection tools ----
        btn_frame2 = ttk.Frame(root)
        btn_frame2.pack(fill="x", padx=8, pady=(0, 4))

        self.warmup_btn = ttk.Button(btn_frame2, text="⚡ Warm Up Connection", command=self.on_warmup)
        self.warmup_btn.pack(side="left", padx=4)

        self.bench_btn = ttk.Button(btn_frame2, text="🔁 Benchmark (3x API)", command=self.on_benchmark)
        self.bench_btn.pack(side="left", padx=4)

        self.conn_status_label = ttk.Label(btn_frame2, text="Connection: not warmed", foreground="gray")
        self.conn_status_label.pack(side="left", padx=12)

        # ---- Status bar ----
        status_frame = ttk.Frame(root)
        status_frame.pack(fill="x", padx=8, pady=(4, 0))
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(status_frame, text="Status:").pack(side="left")
        ttk.Label(status_frame, textvariable=self.status_var, foreground="blue").pack(side="left", padx=(4, 20))
        self.meta_label = ttk.Label(status_frame, text="Meta: not loaded", foreground="gray")
        self.meta_label.pack(side="left")

        # ---- Result box ----
        self.result_box = ScrolledText(root, wrap="none", font=("Courier", 11))
        self.result_box.pack(fill="both", expand=True, padx=8, pady=8)

        self.dataset_loaded = False

        # Background load on startup
        threading.Thread(target=self.background_load, daemon=True).start()

    def _toggle_key_visibility(self):
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")

    def set_status(self, s: str):
        self.status_var.set(s)
        self.root.update_idletasks()

    def set_meta(self, text: str):
        self.meta_label.config(text=text)
        self.root.update_idletasks()

    def set_result(self, text: str):
        if len(text) > 2_000_000:
            text = text[:2_000_000] + "\n\n[truncated large output]"
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text)
        self.root.update_idletasks()

    def append_result(self, text: str):
        self.result_box.insert(tk.END, text)
        self.result_box.see(tk.END)
        self.root.update_idletasks()

    def set_conn_status(self, text: str, color: str = "gray"):
        self.conn_status_label.config(text=f"Connection: {text}", foreground=color)
        self.root.update_idletasks()

    def _set_buttons_state(self, enabled: bool = True):
        state = "normal" if enabled else "disabled"
        for btn in (self.load_btn, self.tmdb_to_wm_btn, self.wm_to_tmdb_btn,
                    self.details_btn, self.similar_btn, self.reload_btn,
                    self.clear_btn, self.info_btn, self.save_key_btn,
                    self.warmup_btn, self.bench_btn):
            try:
                btn.config(state=state)
            except Exception:
                pass
        self.root.update_idletasks()

    # ---- API Key ----

    def on_save_key(self):
        key = self.api_key_var.get().strip()
        if hasattr(self.api, "set_api_key"):
            ok = self.api.set_api_key(key)
            if ok:
                messagebox.showinfo("Saved", "API key saved successfully.")
            else:
                messagebox.showerror("Error", "Failed to save API key.")
        else:
            messagebox.showwarning("Warning", "watchmode_api has no set_api_key function.")

    # ---- Mode ----

    def on_mode_selected(self, event=None):
        new_mode = self.mode_var.get()
        persisted = False
        try:
            if hasattr(self.api, "set_setting_mode"):
                persisted = bool(self.api.set_setting_mode(new_mode))
        except Exception:
            persisted = False

        if persisted:
            if messagebox.askyesno("Mode changed",
                                   f"Mode set to '{new_mode}'. Reload mappings now to apply?"):
                self.on_reload()
        else:
            messagebox.showwarning("Persist failed",
                                   "Could not persist mode selection.")

    # ---- Connection warmup ----

    def on_warmup(self):
        """Pre-establish the TLS connection to api.watchmode.com so subsequent
        requests skip the ~200ms handshake."""
        def task():
            try:
                self.set_status("warming up connection...")
                self._set_buttons_state(False)
                self.set_conn_status("warming up...", "orange")

                # Force session creation
                session = self.api._get_http_session()

                # Make a lightweight request to establish the TCP+TLS connection
                t0 = time.perf_counter()
                try:
                    # Use a cheap endpoint — just hit the search with a dummy to open connection
                    api_key = self.api_key_var.get().strip() or self.api.get_api_key()
                    if api_key:
                        url = f"https://api.watchmode.com/v1/search/?apiKey={api_key}&search_field=tmdb_movie_id&search_value=0"
                        self.api._fetch_url_json(url)
                    else:
                        # No API key — just do a TLS handshake to the host
                        self.api._fetch_url_bytes("https://api.watchmode.com/")
                except Exception:
                    pass  # We don't care about the response, just the connection
                warmup_ms = (time.perf_counter() - t0) * 1000.0

                self.set_conn_status(f"warm ({warmup_ms:.0f}ms handshake)", "green")
                self.set_result(
                    f"Connection warmed up in {warmup_ms:.1f}ms\n"
                    f"Subsequent requests to api.watchmode.com will reuse this connection.\n\n"
                    f"Expected improvement: ~150-250ms per request\n"
                    f"(TLS handshake + TCP setup are now cached)\n\n"
                    f"Session type: {self.api.dataset_info().get('http_session', 'unknown')}"
                )
                self.set_status("idle")
            except Exception as e:
                self.set_conn_status("warmup failed", "red")
                self.set_result(f"Warmup error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    # ---- Benchmark ----

    def on_benchmark(self):
        """Make 3 consecutive API calls to show connection reuse speedup."""
        wm_txt = self.wm_id_var.get().strip()
        if not wm_txt.isdigit():
            # Try to resolve from TMDB first
            tmdb_txt = self.tmdb_id_var.get().strip()
            if tmdb_txt.isdigit():
                self.api.ensure_loaded(mode=self.mode_var.get())
                wm_id = self.api.lookup_tmdb_to_watchmode(int(tmdb_txt), self.type_var.get())
                if wm_id:
                    self.wm_id_var.set(str(wm_id))
                    wm_txt = str(wm_id)
                else:
                    messagebox.showerror("Invalid", "Could not resolve TMDB ID. Enter a Watchmode ID.")
                    return
            else:
                messagebox.showerror("Invalid", "Enter a Watchmode ID for benchmarking.")
                return
        wm_id = int(wm_txt)

        def task():
            try:
                self.set_status("benchmarking (3 calls)...")
                self._set_buttons_state(False)

                api_key = self.api_key_var.get().strip() or self.api.get_api_key()
                if not api_key:
                    self.set_result("Error: No API key set.")
                    self.set_status("error")
                    return

                self.set_result("=== Connection Reuse Benchmark ===\n\n")
                times = []

                for i in range(3):
                    t0 = time.perf_counter()
                    try:
                        details = self.api.call_details_api(wm_id, api_key=api_key)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        times.append(elapsed_ms)
                        title = details.get("title", "?") if details else "error"
                        self.append_result(
                            f"  Call #{i+1}: {elapsed_ms:7.1f}ms  — '{title}'\n"
                        )
                    except Exception as e:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        times.append(elapsed_ms)
                        self.append_result(
                            f"  Call #{i+1}: {elapsed_ms:7.1f}ms  — ERROR: {e}\n"
                        )

                self.append_result(f"\n{'─' * 50}\n")
                if len(times) >= 2:
                    self.append_result(
                        f"  1st call (cold):  {times[0]:7.1f}ms  (includes TLS handshake)\n"
                        f"  2nd call (warm):  {times[1]:7.1f}ms  (connection reused)\n"
                    )
                    if len(times) >= 3:
                        self.append_result(
                            f"  3rd call (warm):  {times[2]:7.1f}ms  (connection reused)\n"
                        )
                    speedup = times[0] - min(times[1:])
                    self.append_result(
                        f"\n  Speedup from reuse: ~{speedup:.0f}ms saved per request\n"
                        f"  Session type: {self.api.dataset_info().get('http_session', 'unknown')}\n"
                    )
                    self.set_conn_status(f"warm (reuse saves ~{speedup:.0f}ms)", "green")

                self.set_status("idle")
            except Exception as e:
                self.set_result(f"Benchmark error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    # ---- Background load ----

    def background_load(self):
        try:
            self.set_status("background loading...")
            self._set_buttons_state(False)
            self.mode_combo.config(state="disabled")
            mode = self.mode_var.get()
            self.api.ensure_loaded(mode=mode)
            self.dataset_loaded = True
            runtime = self.api.get_runtime_mode() if hasattr(self.api, "get_runtime_mode") else None
            info = self.api.dataset_info() if hasattr(self.api, "dataset_info") else {}
            self.set_meta(f"Loaded. runtime_mode={runtime or 'unknown'}")
            self.set_result(json.dumps(info, indent=2, ensure_ascii=False, default=str))
            self.set_status("idle")
        except Exception as e:
            print("background_load error:", e)
            import traceback
            traceback.print_exc()
            self.set_meta(f"Load error: {e}")
            self.set_result(f"Load error:\n{e}")
            self.set_status("error")
        finally:
            try:
                self.mode_combo.config(state="readonly")
            except Exception:
                pass
            self._set_buttons_state(True)

    # ---- Load ----

    def on_load(self):
        def task():
            try:
                self.set_status("loading...")
                self._set_buttons_state(False)
                mode = self.mode_var.get()
                self.api.ensure_loaded(mode=mode)
                self.dataset_loaded = True
                runtime = self.api.get_runtime_mode() if hasattr(self.api, "get_runtime_mode") else None
                self.set_meta(f"Loaded. runtime_mode={runtime or 'unknown'}")
                info = self.api.dataset_info() if hasattr(self.api, "dataset_info") else {}
                self.set_result(json.dumps(info, indent=2, ensure_ascii=False, default=str))
                self.set_status("idle")
            except Exception as e:
                self.set_meta(f"Load error: {e}")
                self.set_result(f"Load error:\n{e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    # ---- Reload ----

    def on_reload(self):
        def task():
            try:
                self.set_status("reloading...")
                self._set_buttons_state(False)
                mode = self.mode_var.get()
                if hasattr(self.api, "reload_dataset"):
                    self.api.reload_dataset(mode=mode)
                else:
                    self.api.ensure_loaded(mode=mode, force=True)
                runtime = self.api.get_runtime_mode() if hasattr(self.api, "get_runtime_mode") else None
                self.set_meta(f"Reloaded. runtime_mode={runtime or 'unknown'}")
                info = self.api.dataset_info() if hasattr(self.api, "dataset_info") else {}
                self.set_result(json.dumps(info, indent=2, ensure_ascii=False, default=str))
                self.set_status("idle")
            except Exception as e:
                self.set_meta(f"Reload error: {e}")
                self.set_result(f"Reload error:\n{e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    # ---- Clear cache ----

    def on_clear(self):
        try:
            if hasattr(self.api, "clear_watchmode_cache"):
                self.api.clear_watchmode_cache()
            cache_dir = os.path.join(os.path.expanduser("~"),
                                     "addon_data", "plugin.video.fenlight", "cache", "watchmode")
            if os.path.isdir(cache_dir):
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
            self.set_meta("Cache cleared.")
            self.set_result(json.dumps({"cache_cleared": True}, indent=2))
            self.set_conn_status("not warmed", "gray")
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear cache: {e}")

    # ---- Info ----

    def on_info(self):
        try:
            info = self.api.dataset_info() if hasattr(self.api, "dataset_info") else {"error": "no dataset_info"}
            self.set_result(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        except Exception as e:
            self.set_result(f"Error: {e}")

    # ---- TMDB → Watchmode ----

    def on_tmdb_to_wm(self):
        tmdb_txt = self.tmdb_id_var.get().strip()
        if not tmdb_txt.isdigit():
            messagebox.showerror("Invalid", "Enter a numeric TMDB ID.")
            return
        tmdb_id = int(tmdb_txt)
        kind = self.type_var.get()

        def task():
            try:
                self.set_status("looking up TMDB → WM...")
                self._set_buttons_state(False)
                self.api.ensure_loaded(mode=self.mode_var.get())

                t0 = time.perf_counter()
                wm_id = self.api.lookup_tmdb_to_watchmode(tmdb_id, kind)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                result = {
                    "tmdb_id": tmdb_id,
                    "type": kind,
                    "watchmode_id": wm_id,
                    "found": wm_id is not None,
                    "lookup_ms": round(elapsed_ms, 3),
                }
                self.set_result(json.dumps(result, indent=2))
                self.set_meta(f"TMDB {tmdb_id} → WM {wm_id or 'not found'} ({elapsed_ms:.3f} ms)")

                if wm_id is not None:
                    self.wm_id_var.set(str(wm_id))

                self.set_status("idle")
            except Exception as e:
                self.set_result(f"Error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    # ---- Watchmode → TMDB ----

    def on_wm_to_tmdb(self):
        wm_txt = self.wm_id_var.get().strip()
        if not wm_txt.isdigit():
            messagebox.showerror("Invalid", "Enter a numeric Watchmode ID.")
            return
        wm_id = int(wm_txt)

        def task():
            try:
                self.set_status("looking up WM → TMDB...")
                self._set_buttons_state(False)
                self.api.ensure_loaded(mode=self.mode_var.get())

                t0 = time.perf_counter()
                result = self.api.lookup_watchmode_to_tmdb(wm_id)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                out = {
                    "watchmode_id": wm_id,
                    "tmdb_id": result["tmdb_id"] if result else None,
                    "media_type": result["media_type"] if result else None,
                    "found": result is not None,
                    "lookup_ms": round(elapsed_ms, 3),
                }
                self.set_result(json.dumps(out, indent=2))

                if result:
                    self.tmdb_id_var.set(str(result["tmdb_id"]))
                    self.type_var.set(result["media_type"])
                    self.set_meta(f"WM {wm_id} → TMDB {result['tmdb_id']} ({result['media_type']}) ({elapsed_ms:.3f} ms)")
                else:
                    self.set_meta(f"WM {wm_id} → not found ({elapsed_ms:.3f} ms)")

                self.set_status("idle")
            except Exception as e:
                self.set_result(f"Error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    # ---- WM Details (API) ----

    def on_details(self):
        wm_txt = self.wm_id_var.get().strip()
        if not wm_txt.isdigit():
            tmdb_txt = self.tmdb_id_var.get().strip()
            if tmdb_txt.isdigit():
                self._resolve_then_details(int(tmdb_txt), self.type_var.get())
                return
            messagebox.showerror("Invalid", "Enter a Watchmode ID or TMDB ID first.")
            return
        wm_id = int(wm_txt)

        def task():
            try:
                self.set_status("calling WM details API...")
                self._set_buttons_state(False)

                api_key = self.api_key_var.get().strip() or self.api.get_api_key()
                if not api_key:
                    self.set_result("Error: No API key set. Save your API key first.")
                    self.set_status("error")
                    return

                t0 = time.perf_counter()
                details = self.api.call_details_api(wm_id, api_key=api_key)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if details:
                    out = {
                        "watchmode_id": wm_id,
                        "api_time_ms": round(elapsed_ms, 3),
                        "details": details,
                    }
                    self.set_result(json.dumps(out, indent=2, ensure_ascii=False, default=str))
                    title = details.get("title", "?")
                    similar_count = len(details.get("similar_titles", []))
                    self.set_meta(f"Details: '{title}' — {similar_count} similar ({elapsed_ms:.1f} ms)")
                    self.set_conn_status(f"warm ({elapsed_ms:.0f}ms)", "green")
                else:
                    self.set_result(json.dumps({"error": "No data returned"}, indent=2))
                    self.set_meta(f"Details: no data ({elapsed_ms:.1f} ms)")

                self.set_status("idle")
            except Exception as e:
                self.set_result(f"Error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    def _resolve_then_details(self, tmdb_id: int, kind: str):
        def task():
            try:
                self.set_status("resolving TMDB → WM, then details...")
                self._set_buttons_state(False)
                self.api.ensure_loaded(mode=self.mode_var.get())

                wm_id = self.api.lookup_tmdb_to_watchmode(tmdb_id, kind)
                if wm_id is None:
                    api_key = self.api_key_var.get().strip() or self.api.get_api_key()
                    if api_key:
                        try:
                            wm_id = self.api.search_watchmode_via_api(tmdb_id, kind, api_key=api_key)
                        except Exception:
                            pass

                if wm_id is None:
                    self.set_result(json.dumps({"error": f"Could not resolve TMDB {tmdb_id} to Watchmode ID"}))
                    self.set_status("idle")
                    return

                self.wm_id_var.set(str(wm_id))

                api_key = self.api_key_var.get().strip() or self.api.get_api_key()
                t0 = time.perf_counter()
                details = self.api.call_details_api(wm_id, api_key=api_key)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if details:
                    out = {
                        "tmdb_id": tmdb_id,
                        "watchmode_id": wm_id,
                        "api_time_ms": round(elapsed_ms, 3),
                        "details": details,
                    }
                    self.set_result(json.dumps(out, indent=2, ensure_ascii=False, default=str))
                    self.set_meta(f"TMDB {tmdb_id} → WM {wm_id} details ({elapsed_ms:.1f} ms)")
                    self.set_conn_status(f"warm ({elapsed_ms:.0f}ms)", "green")
                else:
                    self.set_result(json.dumps({"error": "No details returned"}, indent=2))

                self.set_status("idle")
            except Exception as e:
                self.set_result(f"Error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()

    # ---- Full Similar Pipeline ----

    def on_similar(self):
        tmdb_txt = self.tmdb_id_var.get().strip()
        if not tmdb_txt.isdigit():
            messagebox.showerror("Invalid", "Enter a numeric TMDB ID for the full similar pipeline.")
            return
        tmdb_id = int(tmdb_txt)
        kind = self.type_var.get()

        def task():
            try:
                self.set_status("running full similar pipeline...")
                self._set_buttons_state(False)
                self.api.ensure_loaded(mode=self.mode_var.get())

                api_key = self.api_key_var.get().strip() or self.api.get_api_key()

                t0 = time.perf_counter()
                result, total_ms = self.api.get_similar_titles_resolved(
                    tmdb_id, kind, api_key=api_key, timing=True
                )
                elapsed = (time.perf_counter() - t0) * 1000.0

                result["total_pipeline_ms"] = round(elapsed, 3)

                self.set_result(json.dumps(result, indent=2, ensure_ascii=False, default=str))

                wm_id = result.get("watchmode_id")
                count = result.get("similar_count", 0)
                title = result.get("details_title", "?")
                if wm_id:
                    self.wm_id_var.set(str(wm_id))
                self.set_meta(
                    f"'{title}' — WM {wm_id} — {count} similar resolved ({elapsed:.1f} ms total)"
                )
                self.set_conn_status(f"warm ({elapsed:.0f}ms pipeline)", "green")

                self.set_status("idle")
            except Exception as e:
                import traceback
                self.set_result(f"Error:\n{e}\n\n{traceback.format_exc()}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)
        threading.Thread(target=task, daemon=True).start()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()