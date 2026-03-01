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
from typing import Any, Dict, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_FILE = os.path.join(_THIS_DIR, ".watchmode_settings.json")


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
                json.dump(d, fh, indent=2)
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
        raise RuntimeError(
            f"Could not import watchmode_api even after installing stubs: {exc2}"
        ) from exc2


try:
    wm = import_watchmode_api_with_stubs()
except Exception as e:
    raise SystemExit(f"Failed to import watchmode_api.py for testing: {e}")


try:
    from caches.settings_cache import get_setting as sc_get_setting  # type: ignore
    from caches.settings_cache import set_setting as sc_set_setting  # type: ignore
except Exception:
    sc_get_setting = None  # type: ignore
    sc_set_setting = None  # type: ignore


# ---------------------------------------------------------------------
# Helper reads
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


def compute_watchmode_cache_dir() -> str:
    try:
        base = wm.translate_path(  # type: ignore[attr-defined]
            "special://profile/addon_data/plugin.video.fenlight/"
        )
        return os.path.join(base, "cache", "watchmode")
    except Exception:
        return os.path.join(
            os.path.expanduser("~"),
            "addon_data", "plugin.video.fenlight", "cache", "watchmode",
        )


def safe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s.isdigit():
        return None
    try:
        return int(s)
    except Exception:
        return None


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("watchmode API Tester")
        root.geometry("1100x780")

        self.api = wm
        self._main_thread = threading.current_thread()

        # ---- Top frame: API key ----
        api_frame = ttk.LabelFrame(root, text="Watchmode API Key")
        api_frame.pack(fill="x", padx=8, pady=(8, 4))

        self.api_key_var = tk.StringVar(value=read_persisted_api_key())
        self.api_key_entry = ttk.Entry(
            api_frame, textvariable=self.api_key_var, width=50, show="*"
        )
        self.api_key_entry.pack(side="left", padx=4, pady=4)

        self.show_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            api_frame, text="Show",
            variable=self.show_key_var, command=self._toggle_key_visibility,
        ).pack(side="left", padx=4)

        self.save_key_btn = ttk.Button(
            api_frame, text="Save Key", command=self.on_save_key
        )
        self.save_key_btn.pack(side="left", padx=4)

        ttk.Label(
            api_frame,
            text="(key change invalidates API cache)",
            foreground="gray",
        ).pack(side="left", padx=8)

        # ---- Controls frame ----
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(ctrl_frame, text="TMDB ID:").grid(row=0, column=0, sticky="w")
        self.tmdb_id_var = tk.StringVar(value="")
        ttk.Entry(ctrl_frame, textvariable=self.tmdb_id_var, width=12).grid(
            row=0, column=1, sticky="w"
        )

        ttk.Label(ctrl_frame, text="Type:").grid(
            row=0, column=2, sticky="w", padx=(10, 0)
        )
        self.type_var = tk.StringVar(value="movie")
        ttk.Combobox(
            ctrl_frame, textvariable=self.type_var,
            values=("movie", "tv"), width=8, state="readonly",
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(ctrl_frame, text="WM ID:").grid(
            row=0, column=4, sticky="w", padx=(10, 0)
        )
        self.wm_id_var = tk.StringVar(value="")
        ttk.Entry(ctrl_frame, textvariable=self.wm_id_var, width=12).grid(
            row=0, column=5, sticky="w"
        )

        ttk.Label(ctrl_frame, text="Mode:").grid(
            row=0, column=6, sticky="w", padx=(10, 0)
        )
        init_mode = read_persisted_mode()
        if init_mode not in ("auto", "RAM", "mmap"):
            init_mode = "auto"
        self.mode_var = tk.StringVar(value=init_mode)
        self.mode_combo = ttk.Combobox(
            ctrl_frame, textvariable=self.mode_var,
            values=("auto", "RAM", "mmap"), width=10, state="readonly",
        )
        self.mode_combo.grid(row=0, column=7, sticky="w", padx=(4, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_selected)

        # ---- watchmode output selector ----
        ttk.Label(ctrl_frame, text="watchmode output:").grid(
            row=0, column=8, sticky="w", padx=(14, 0)
        )
        self.watchmode_format_var = tk.StringVar(value="bundle")
        self.watchmode_format_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.watchmode_format_var,
            values=("bundle", "addon", "extras", "pairs", "packed"),
            width=16,
            state="readonly",
        )
        self.watchmode_format_combo.grid(row=0, column=9, sticky="w", padx=(4, 0))

        # ---- Cache options frame ----
        cacheopt = ttk.LabelFrame(root, text="Cache Options")
        cacheopt.pack(fill="x", padx=8, pady=4)

        self.bypass_api_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cacheopt,
            text="Bypass API cache (forces HTTP to Watchmode)",
            variable=self.bypass_api_cache_var,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=2)

        self.cache_version_var = tk.StringVar(value="cache_version: ?")
        ttk.Label(
            cacheopt, textvariable=self.cache_version_var, foreground="gray"
        ).grid(row=0, column=2, sticky="e", padx=6)
        cacheopt.grid_columnconfigure(2, weight=1)

        # ---- TTL editor ----
        ttl_frame = ttk.LabelFrame(
            root,
            text="API cache TTLs (hours) \u2014 edits settings_cache keys",
        )
        ttl_frame.pack(fill="x", padx=8, pady=4)

        self.ttl_details_var = tk.StringVar(value="")
        self.ttl_similar_var = tk.StringVar(value="")
        self.ttl_negative_var = tk.StringVar(value="")

        def ttl_entry(col: int, label: str, var: tk.StringVar):
            ttk.Label(ttl_frame, text=label).grid(
                row=0, column=col, sticky="w", padx=(6, 2), pady=2
            )
            ttk.Entry(ttl_frame, textvariable=var, width=8).grid(
                row=0, column=col + 1, sticky="w", padx=(0, 12)
            )

        ttl_entry(0, "Details:", self.ttl_details_var)
        ttl_entry(2, "Similar/Bundle:", self.ttl_similar_var)
        ttl_entry(4, "Negative:", self.ttl_negative_var)

        self.save_ttls_btn = ttk.Button(
            ttl_frame, text="Save TTLs", command=self.on_save_ttls
        )
        self.save_ttls_btn.grid(row=0, column=6, sticky="w", padx=6)

        ttk.Label(
            ttl_frame,
            text="Tip: change TTLs then Clear Cache to fully apply.",
            foreground="gray",
        ).grid(row=0, column=7, sticky="w", padx=6)

        # ---- Buttons frame ----
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=8, pady=4)

        self.load_btn = ttk.Button(
            btn_frame, text="Load Mappings", command=self.on_load
        )
        self.load_btn.pack(side="left", padx=4)

        self.tmdb_to_wm_btn = ttk.Button(
            btn_frame, text="TMDB \u2192 WM", command=self.on_tmdb_to_wm
        )
        self.tmdb_to_wm_btn.pack(side="left", padx=4)

        self.wm_to_tmdb_btn = ttk.Button(
            btn_frame, text="WM \u2192 TMDB", command=self.on_wm_to_tmdb
        )
        self.wm_to_tmdb_btn.pack(side="left", padx=4)

        self.watchmode_btn = ttk.Button(
            btn_frame, text="Run watchmode", command=self.on_watchmode
        )
        self.watchmode_btn.pack(side="left", padx=4)

        self.reload_btn = ttk.Button(
            btn_frame, text="Reload", command=self.on_reload
        )
        self.reload_btn.pack(side="left", padx=4)

        self.clear_btn = ttk.Button(
            btn_frame, text="Clear Cache", command=self.on_clear
        )
        self.clear_btn.pack(side="left", padx=4)

        self.info_btn = ttk.Button(
            btn_frame, text="Info", command=self.on_info
        )
        self.info_btn.pack(side="left", padx=4)

        # ---- Status bar ----
        status_frame = ttk.Frame(root)
        status_frame.pack(fill="x", padx=8, pady=(4, 0))
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(status_frame, text="Status:").pack(side="left")
        ttk.Label(
            status_frame, textvariable=self.status_var, foreground="blue"
        ).pack(side="left", padx=(4, 20))
        self.meta_label = ttk.Label(
            status_frame, text="Meta: not loaded", foreground="gray"
        )
        self.meta_label.pack(side="left")

        # ---- Result box ----
        self.result_box = ScrolledText(
            root, wrap="none", font=("Courier", 11)
        )
        self.result_box.pack(fill="both", expand=True, padx=8, pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.dataset_loaded = False

        self.refresh_cache_labels()
        self.refresh_ttl_fields_from_api()

        threading.Thread(target=self.background_load, daemon=True).start()

    # ---------------- UI helpers ----------------

    def _ui(self, fn, *args, **kwargs):
        if threading.current_thread() is self._main_thread:
            return fn(*args, **kwargs)
        self.root.after(0, lambda: fn(*args, **kwargs))

    def _toggle_key_visibility(self):
        self.api_key_entry.config(
            show="" if self.show_key_var.get() else "*"
        )

    def set_status(self, s: str):
        def _do():
            self.status_var.set(s)
            self.root.update_idletasks()
        self._ui(_do)

    def set_meta(self, text: str):
        def _do():
            self.meta_label.config(text=text)
            self.root.update_idletasks()
        self._ui(_do)

    def set_result(self, text: str):
        def _do():
            if len(text) > 2_000_000:
                text2 = text[:2_000_000] + "\n\n[truncated large output]"
            else:
                text2 = text
            self.result_box.delete("1.0", tk.END)
            self.result_box.insert(tk.END, text2)
            self.root.update_idletasks()
        self._ui(_do)

    def _set_buttons_state(self, enabled: bool = True):
        def _do():
            state = "normal" if enabled else "disabled"
            for btn in (
                self.load_btn,
                self.tmdb_to_wm_btn,
                self.wm_to_tmdb_btn,
                self.watchmode_btn,
                self.reload_btn,
                self.clear_btn,
                self.info_btn,
                self.save_key_btn,
                self.save_ttls_btn,
            ):
                try:
                    btn.config(state=state)
                except Exception:
                    pass
            self.root.update_idletasks()
        self._ui(_do)

    def refresh_cache_labels(self):
        try:
            v = None
            if hasattr(self.api, "main_cache"):
                v = self.api.main_cache.get(  # type: ignore[attr-defined]
                    "watchmode:cache_version"
                )
            if v is None:
                v = "?"
            self.cache_version_var.set(f"cache_version: {v}")
        except Exception:
            self.cache_version_var.set("cache_version: ?")

    def refresh_ttl_fields_from_api(self):
        if sc_get_setting is None:
            return
        try:
            self.ttl_details_var.set(
                sc_get_setting(
                    "fenlight.watchmode.cache.details_hours", "168"
                )
            )
            self.ttl_similar_var.set(
                sc_get_setting(
                    "fenlight.watchmode.cache.similar_hours", "24"
                )
            )
            self.ttl_negative_var.set(
                sc_get_setting(
                    "fenlight.watchmode.cache.negative_hours", "24"
                )
            )
        except Exception:
            pass

    # ---------------- Events ----------------

    def on_close(self):
        try:
            if hasattr(self.api, "close_dataset"):
                self.api.close_dataset()
        except Exception:
            pass
        self.root.destroy()

    def on_save_key(self):
        key = self.api_key_var.get().strip()
        try:
            if hasattr(self.api, "set_api_key"):
                ok = bool(self.api.set_api_key(key))
                if ok:
                    self.refresh_cache_labels()
                    messagebox.showinfo(
                        "Saved",
                        "API key saved successfully (API cache invalidated).",
                    )
                else:
                    messagebox.showerror("Error", "Failed to save API key.")
            else:
                messagebox.showwarning(
                    "Warning",
                    "watchmode_api has no set_api_key function.",
                )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_save_ttls(self):
        if sc_set_setting is None:
            messagebox.showerror(
                "Unavailable",
                "settings_cache.set_setting is not available in this environment.",
            )
            return

        def norm_hours(v: str) -> str:
            v = (v or "").strip()
            try:
                n = int(v)
                return str(max(1, n))
            except Exception:
                return v or ""

        try:
            sc_set_setting(
                "fenlight.watchmode.cache.details_hours",
                norm_hours(self.ttl_details_var.get()),
            )
            sc_set_setting(
                "fenlight.watchmode.cache.similar_hours",
                norm_hours(self.ttl_similar_var.get()),
            )
            sc_set_setting(
                "fenlight.watchmode.cache.negative_hours",
                norm_hours(self.ttl_negative_var.get()),
            )
            messagebox.showinfo(
                "Saved",
                "TTL settings saved. "
                "(Existing cached items keep old expiry until cleared.)",
            )
            self.on_info()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save TTLs: {e}")

    def on_mode_selected(self, event=None):
        new_mode = self.mode_var.get()
        persisted = False
        try:
            if hasattr(self.api, "set_setting_mode"):
                persisted = bool(self.api.set_setting_mode(new_mode))
        except Exception:
            persisted = False

        if persisted:
            if messagebox.askyesno(
                "Mode changed",
                f"Mode set to '{new_mode}'. Reload mappings now to apply?",
            ):
                self.on_reload()
        else:
            messagebox.showwarning(
                "Persist failed", "Could not persist mode selection."
            )

    def background_load(self):
        try:
            self.set_status("background loading...")
            self._set_buttons_state(False)
            self._ui(lambda: self.mode_combo.config(state="disabled"))
            mode = self.mode_var.get()
            self.api.ensure_loaded(mode=mode)
            self.dataset_loaded = True
            runtime = (
                self.api.get_runtime_mode()
                if hasattr(self.api, "get_runtime_mode")
                else None
            )
            info = (
                self.api.dataset_info()
                if hasattr(self.api, "dataset_info")
                else {}
            )
            self.refresh_cache_labels()
            self.refresh_ttl_fields_from_api()
            self.set_meta(f"Loaded. runtime_mode={runtime or 'unknown'}")
            self.set_result(
                json.dumps(info, indent=2, ensure_ascii=False, default=str)
            )
            self.set_status("idle")
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.set_meta(f"Load error: {e}")
            self.set_result(
                f"Load error:\n{e}\n\n{traceback.format_exc()}"
            )
            self.set_status("error")
        finally:
            self._ui(lambda: self.mode_combo.config(state="readonly"))
            self._set_buttons_state(True)

    def on_load(self):
        def task():
            try:
                self.set_status("loading...")
                self._set_buttons_state(False)
                mode = self.mode_var.get()
                self.api.ensure_loaded(mode=mode)
                self.dataset_loaded = True
                runtime = (
                    self.api.get_runtime_mode()
                    if hasattr(self.api, "get_runtime_mode")
                    else None
                )
                info = (
                    self.api.dataset_info()
                    if hasattr(self.api, "dataset_info")
                    else {}
                )
                self.refresh_cache_labels()
                self.refresh_ttl_fields_from_api()
                self.set_meta(
                    f"Loaded. runtime_mode={runtime or 'unknown'}"
                )
                self.set_result(
                    json.dumps(
                        info, indent=2, ensure_ascii=False, default=str
                    )
                )
                self.set_status("idle")
            except Exception as e:
                self.set_meta(f"Load error: {e}")
                self.set_result(f"Load error:\n{e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=task, daemon=True).start()

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
                runtime = (
                    self.api.get_runtime_mode()
                    if hasattr(self.api, "get_runtime_mode")
                    else None
                )
                info = (
                    self.api.dataset_info()
                    if hasattr(self.api, "dataset_info")
                    else {}
                )
                self.refresh_cache_labels()
                self.refresh_ttl_fields_from_api()
                self.set_meta(
                    f"Reloaded. runtime_mode={runtime or 'unknown'}"
                )
                self.set_result(
                    json.dumps(
                        info, indent=2, ensure_ascii=False, default=str
                    )
                )
                self.set_status("idle")
            except Exception as e:
                self.set_meta(f"Reload error: {e}")
                self.set_result(f"Reload error:\n{e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=task, daemon=True).start()

    def on_clear(self):
        try:
            if hasattr(self.api, "clear_watchmode_cache"):
                self.api.clear_watchmode_cache()

            cache_dir = compute_watchmode_cache_dir()
            if os.path.isdir(cache_dir):
                import shutil

                shutil.rmtree(cache_dir, ignore_errors=True)

            self.refresh_cache_labels()
            self.set_meta(
                "Cache cleared (cache_version bumped + mapping files deleted)."
            )
            self.set_result(
                json.dumps(
                    {
                        "cache_cleared": True,
                        "mapping_cache_dir_deleted": cache_dir,
                    },
                    indent=2,
                )
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear cache: {e}")

    def on_info(self):
        try:
            info = (
                self.api.dataset_info()
                if hasattr(self.api, "dataset_info")
                else {"error": "no dataset_info"}
            )
            self.refresh_cache_labels()
            self.refresh_ttl_fields_from_api()
            self.set_result(
                json.dumps(info, indent=2, ensure_ascii=False, default=str)
            )
        except Exception as e:
            self.set_result(f"Error: {e}")

    def on_tmdb_to_wm(self):
        tmdb_id = safe_int(self.tmdb_id_var.get())
        if tmdb_id is None:
            messagebox.showerror("Invalid", "Enter a numeric TMDB ID.")
            return
        kind = self.type_var.get()

        def task():
            try:
                self.set_status("looking up TMDB \u2192 WM (local)...")
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
                self.set_meta(
                    f"TMDB {tmdb_id} \u2192 WM "
                    f"{wm_id or 'not found'} ({elapsed_ms:.3f} ms)"
                )

                if wm_id is not None:
                    self.wm_id_var.set(str(wm_id))

                self.refresh_cache_labels()
                self.set_status("idle")
            except Exception as e:
                self.set_result(f"Error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=task, daemon=True).start()

    def on_wm_to_tmdb(self):
        wm_id = safe_int(self.wm_id_var.get())
        if wm_id is None:
            messagebox.showerror(
                "Invalid", "Enter a numeric Watchmode ID."
            )
            return

        def task():
            try:
                self.set_status("looking up WM \u2192 TMDB (local)...")
                self._set_buttons_state(False)
                self.api.ensure_loaded(mode=self.mode_var.get())

                t0 = time.perf_counter()
                packed = self.api.lookup_watchmode_to_tmdb_packed(wm_id)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if packed is not None:
                    tmdb_id = packed >> 1
                    media_type = "tv" if (packed & 1) else "movie"
                else:
                    tmdb_id = None
                    media_type = None

                out = {
                    "watchmode_id": wm_id,
                    "packed_value": packed,
                    "tmdb_id": tmdb_id,
                    "media_type": media_type,
                    "found": packed is not None,
                    "lookup_ms": round(elapsed_ms, 3),
                }
                self.set_result(json.dumps(out, indent=2))

                if packed is not None:
                    self.tmdb_id_var.set(str(tmdb_id))
                    self.type_var.set(media_type)
                    self.set_meta(
                        f"WM {wm_id} \u2192 TMDB {tmdb_id} "
                        f"({media_type}) ({elapsed_ms:.3f} ms)"
                    )
                else:
                    self.set_meta(
                        f"WM {wm_id} \u2192 not found "
                        f"({elapsed_ms:.3f} ms)"
                    )

                self.refresh_cache_labels()
                self.set_status("idle")
            except Exception as e:
                self.set_result(f"Error: {e}")
                self.set_status("error")
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=task, daemon=True).start()

    # ---- watchmode pipeline ----
    def on_watchmode(self):
        tmdb_id = safe_int(self.tmdb_id_var.get())
        if tmdb_id is None:
            messagebox.showerror("Invalid", "Enter a numeric TMDB ID.")
            return
        kind = self.type_var.get()
        fmt = (self.watchmode_format_var.get() or "bundle").strip()

        def task():
            try:
                self.set_status("running watchmode pipeline...")
                self._set_buttons_state(False)
                self.api.ensure_loaded(mode=self.mode_var.get())

                api_key = (
                    self.api_key_var.get().strip() or self.api.get_api_key()
                )
                if not api_key:
                    self.set_result(
                        "Error: No API key set. Save your API key first."
                    )
                    self.set_status("error")
                    return

                bypass = bool(self.bypass_api_cache_var.get())

                t0 = time.perf_counter()
                result: Any

                if fmt == "bundle":
                    if not hasattr(self.api, "get_watchmode_bundle_for_addon"):
                        raise RuntimeError(
                            "watchmode_api.get_watchmode_bundle_for_addon "
                            "not found (API mismatch)"
                        )
                    result = self.api.get_watchmode_bundle_for_addon(
                        tmdb_id, kind,
                        api_key=api_key, bypass_cache=bypass,
                    )

                elif fmt == "addon":
                    if not hasattr(self.api, "get_watchmode_for_addon"):
                        raise RuntimeError(
                            "watchmode_api.get_watchmode_for_addon "
                            "not found (API mismatch)"
                        )
                    result = self.api.get_watchmode_for_addon(
                        tmdb_id, kind,
                        api_key=api_key, bypass_cache=bypass,
                    )

                elif fmt == "extras":
                    if not hasattr(self.api, "get_watchmode_extras"):
                        raise RuntimeError(
                            "watchmode_api.get_watchmode_extras "
                            "not found (API mismatch)"
                        )
                    result = self.api.get_watchmode_extras(
                        tmdb_id, kind,
                        api_key=api_key, bypass_cache=bypass,
                    )

                elif fmt == "pairs":
                    if not hasattr(self.api, "query_watchmode_pairs"):
                        raise RuntimeError(
                            "watchmode_api.query_watchmode_pairs "
                            "not found (API mismatch)"
                        )
                    result = self.api.query_watchmode_pairs(
                        tmdb_id, kind,
                        api_key=api_key, bypass_cache=bypass,
                    )

                elif fmt == "packed":
                    if not hasattr(self.api, "query_watchmode_packed"):
                        raise RuntimeError(
                            "watchmode_api.query_watchmode_packed "
                            "not found (API mismatch)"
                        )
                    packed = self.api.query_watchmode_packed(
                        tmdb_id, kind,
                        api_key=api_key, bypass_cache=bypass,
                    )
                    decoded = [
                        {
                            "id": (p >> 1),
                            "media_type": "tv" if (p & 1) else "movie",
                        }
                        for p in packed
                    ]
                    result = {
                        "count": len(packed),
                        "packed": packed,
                        "decoded": decoded,
                    }

                else:
                    raise ValueError(f"Unknown format: {fmt}")

                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if isinstance(result, dict):
                    result["tester_total_ms"] = round(elapsed_ms, 3)
                    result["bypass_cache"] = bypass
                    result["format"] = fmt

                self.set_result(
                    json.dumps(
                        result, indent=2,
                        ensure_ascii=False, default=str,
                    )
                )

                # meta line
                count = None
                if isinstance(result, dict):
                    count = (
                        result.get("total_results")
                        or result.get("count")
                    )

                # Show extras summary in meta if bundle format
                extras_hint = ""
                if fmt == "bundle" and isinstance(result, dict):
                    extras = result.get("watchmode_extras", {})
                    if isinstance(extras, dict):
                        has = [
                            k for k in (
                                "will_you_like_this",
                                "review_summary",
                                "user_rating",
                                "critic_score",
                            )
                            if extras.get(k) is not None
                        ]
                        extras_hint = (
                            f" \u2014 extras: {', '.join(has)}"
                            if has else " \u2014 extras: (none)"
                        )
                elif fmt == "extras" and isinstance(result, dict):
                    has = [
                        k for k in result
                        if result.get(k) is not None
                        and k not in (
                            "tester_total_ms",
                            "bypass_cache",
                            "format",
                        )
                    ]
                    extras_hint = (
                        f" \u2014 fields: {', '.join(has)}"
                        if has else " \u2014 fields: (none)"
                    )

                self.set_meta(
                    f"watchmode: TMDB {tmdb_id} ({kind}) "
                    f"\u2014 format={fmt} \u2014 count={count}"
                    f"{extras_hint} ({elapsed_ms:.1f} ms)"
                )

                self.refresh_cache_labels()
                self.refresh_ttl_fields_from_api()
                self.set_status("idle")
            except Exception as e:
                import traceback

                self.set_result(
                    f"Error:\n{e}\n\n{traceback.format_exc()}"
                )
                self.set_status("error")
            finally:
                self._set_buttons_state(True)

        threading.Thread(target=task, daemon=True).start()


def main():
    root = tk.Tk()
    _app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()