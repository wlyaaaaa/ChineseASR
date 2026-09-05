"""Small Windows desktop host for local Win+H dictation.

The host intentionally owns only desktop interaction.  It does not record audio,
load an ASR model, retain transcripts on disk, or alter Windows-wide shortcuts.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import queue
import sys
import threading
import time
from typing import Callable, Protocol


_ULONG_PTR = ctypes.c_size_t
_WPARAM = ctypes.c_size_t
_LPARAM = ctypes.c_ssize_t
_LRESULT = ctypes.c_ssize_t
_HWND = ctypes.c_void_p
_HHOOK = ctypes.c_void_p
_HANDLE = ctypes.c_void_p
_BOOL = ctypes.c_int
_HOOKPROC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    _LRESULT, ctypes.c_int, _WPARAM, _LPARAM
)

_WH_KEYBOARD_LL = 13
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101
_WM_SYSKEYDOWN = 0x0104
_WM_SYSKEYUP = 0x0105
_VK_ESCAPE = 0x1B
_VK_H = 0x48
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C
_VK_LSHIFT = 0xA0
_VK_RSHIFT = 0xA1
_VK_LCONTROL = 0xA2
_VK_RCONTROL = 0xA3
_VK_LMENU = 0xA4
_VK_RMENU = 0xA5
_VK_MENU_MASK = 0xE8
_MENU_MASK_EXTRA_INFO = 0x43415352  # "CASR": only this host's inert marker.
_LLKHF_LOWER_IL_INJECTED = 0x00000002
_LLKHF_INJECTED = 0x00000010
_GA_ROOT = 2
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020
_SWP_SHOWWINDOW = 0x0040
_ERROR_ALREADY_EXISTS = 183
_EVENT_MODIFY_STATE = 0x0002
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0

_MUTEX_NAME = r"Local\ChineseASR.DictationHost.v1"
_QUIT_EVENT_NAME = r"Local\ChineseASR.DictationHost.Quit.v1"
_MODIFIER_KEYS = (
    _VK_LSHIFT,
    _VK_RSHIFT,
    _VK_LCONTROL,
    _VK_RCONTROL,
    _VK_LMENU,
    _VK_RMENU,
    _VK_LWIN,
    _VK_RWIN,
)
_WIN_KEYS = (_VK_LWIN, _VK_RWIN)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", _HWND),
        ("hwndFocus", _HWND),
        ("hwndCapture", _HWND),
        ("hwndMenuOwner", _HWND),
        ("hwndMoveSize", _HWND),
        ("hwndCaret", _HWND),
        ("rcCaret", _RECT),
    ]


@dataclass(frozen=True)
class TargetWindow:
    """The focused control which was active when one dictation session started."""

    root: int
    focus: int


@dataclass(frozen=True)
class KeyboardEvent:
    vk_code: int
    message: int
    injected: bool = False


def is_available() -> bool:
    """Return whether this process can use the Windows desktop APIs."""

    return sys.platform == "win32" and hasattr(ctypes, "WinDLL")


def _handle_value(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return int(getattr(value, "value", 0) or 0)


def _utf16_units(text: str) -> list[int]:
    data = text.encode("utf-16-le", "surrogatepass")
    return [int.from_bytes(data[index : index + 2], "little") for index in range(0, len(data), 2)]


class _Platform(Protocol):
    available: bool

    def create_mutex(self, name: str) -> tuple[int, bool]: ...
    def close_handle(self, handle: int) -> None: ...
    def create_quit_event(self, name: str) -> int: ...
    def signal_existing_event(self, name: str) -> bool: ...
    def named_mutex_exists(self, name: str) -> bool: ...
    def event_is_signaled(self, handle: int) -> bool: ...
    def install_keyboard_hook(self, callback: object) -> int: ...
    def uninstall_keyboard_hook(self, handle: int) -> None: ...
    def call_next_hook(self, handle: int, n_code: int, w_param: int, l_param: int) -> int: ...
    def get_foreground_window(self) -> int: ...
    def get_root_window(self, hwnd: int) -> int: ...
    def get_focus_window(self, foreground: int) -> int: ...
    def wait_for_modifiers_released(self, timeout: float) -> bool: ...
    def send_unicode_text(self, text: str) -> bool: ...
    def send_menu_mask(self) -> bool: ...
    def make_window_nonactivating(self, hwnd: int, show: bool = True) -> None: ...


class _WinApi:
    """Narrow ctypes wrapper, kept separate so host logic has a fakeable platform."""

    def __init__(self) -> None:
        self.available = is_available()
        if not self.available:
            return
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self.user32.SetWindowsHookExW.argtypes = [ctypes.c_int, _HOOKPROC, _HWND, wintypes.DWORD]
        self.user32.SetWindowsHookExW.restype = _HHOOK
        self.user32.UnhookWindowsHookEx.argtypes = [_HHOOK]
        self.user32.UnhookWindowsHookEx.restype = _BOOL
        self.user32.CallNextHookEx.argtypes = [_HHOOK, ctypes.c_int, _WPARAM, _LPARAM]
        self.user32.CallNextHookEx.restype = _LRESULT
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = _HWND
        self.user32.GetAncestor.argtypes = [_HWND, wintypes.UINT]
        self.user32.GetAncestor.restype = _HWND
        self.user32.GetWindowThreadProcessId.argtypes = [_HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(_GUITHREADINFO)]
        self.user32.GetGUIThreadInfo.restype = _BOOL
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT
        self.user32.SetWindowPos.argtypes = [
            _HWND,
            _HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self.user32.SetWindowPos.restype = _BOOL
        self._get_window_long = getattr(self.user32, "GetWindowLongPtrW", self.user32.GetWindowLongW)
        self._get_window_long.argtypes = [_HWND, ctypes.c_int]
        self._get_window_long.restype = ctypes.c_ssize_t
        self._set_window_long = getattr(self.user32, "SetWindowLongPtrW", self.user32.SetWindowLongW)
        self._set_window_long.argtypes = [_HWND, ctypes.c_int, ctypes.c_ssize_t]
        self._set_window_long.restype = ctypes.c_ssize_t

        self.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetModuleHandleW.restype = _HWND
        self.kernel32.CreateMutexW.argtypes = [_HWND, _BOOL, wintypes.LPCWSTR]
        self.kernel32.CreateMutexW.restype = _HANDLE
        self.kernel32.OpenMutexW.argtypes = [wintypes.DWORD, _BOOL, wintypes.LPCWSTR]
        self.kernel32.OpenMutexW.restype = _HANDLE
        self.kernel32.CreateEventW.argtypes = [_HWND, _BOOL, _BOOL, wintypes.LPCWSTR]
        self.kernel32.CreateEventW.restype = _HANDLE
        self.kernel32.OpenEventW.argtypes = [wintypes.DWORD, _BOOL, wintypes.LPCWSTR]
        self.kernel32.OpenEventW.restype = _HANDLE
        self.kernel32.SetEvent.argtypes = [_HANDLE]
        self.kernel32.SetEvent.restype = _BOOL
        self.kernel32.WaitForSingleObject.argtypes = [_HANDLE, wintypes.DWORD]
        self.kernel32.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel32.CloseHandle.argtypes = [_HANDLE]
        self.kernel32.CloseHandle.restype = _BOOL

    def create_mutex(self, name: str) -> tuple[int, bool]:
        ctypes.set_last_error(0)
        handle = _handle_value(self.kernel32.CreateMutexW(None, False, name))
        return handle, ctypes.get_last_error() == _ERROR_ALREADY_EXISTS

    def close_handle(self, handle: int) -> None:
        if handle:
            self.kernel32.CloseHandle(_HANDLE(handle))

    def create_quit_event(self, name: str) -> int:
        return _handle_value(self.kernel32.CreateEventW(None, False, False, name))

    def signal_existing_event(self, name: str) -> bool:
        handle = _handle_value(self.kernel32.OpenEventW(_EVENT_MODIFY_STATE, False, name))
        if not handle:
            return False
        try:
            return bool(self.kernel32.SetEvent(_HANDLE(handle)))
        finally:
            self.close_handle(handle)

    def named_mutex_exists(self, name: str) -> bool:
        handle = _handle_value(self.kernel32.OpenMutexW(_SYNCHRONIZE, False, name))
        if not handle:
            return False
        self.close_handle(handle)
        return True

    def event_is_signaled(self, handle: int) -> bool:
        return bool(handle) and self.kernel32.WaitForSingleObject(_HANDLE(handle), 0) == _WAIT_OBJECT_0

    def install_keyboard_hook(self, callback: object) -> int:
        module = self.kernel32.GetModuleHandleW(None)
        return _handle_value(self.user32.SetWindowsHookExW(_WH_KEYBOARD_LL, callback, module, 0))

    def uninstall_keyboard_hook(self, handle: int) -> None:
        if handle:
            self.user32.UnhookWindowsHookEx(_HHOOK(handle))

    def call_next_hook(self, handle: int, n_code: int, w_param: int, l_param: int) -> int:
        return int(self.user32.CallNextHookEx(_HHOOK(handle), n_code, w_param, l_param))

    def get_foreground_window(self) -> int:
        return _handle_value(self.user32.GetForegroundWindow())

    def get_root_window(self, hwnd: int) -> int:
        if not hwnd:
            return 0
        return _handle_value(self.user32.GetAncestor(_HWND(hwnd), _GA_ROOT)) or hwnd

    def get_focus_window(self, foreground: int) -> int:
        if not foreground:
            return 0
        thread_id = self.user32.GetWindowThreadProcessId(_HWND(foreground), None)
        info = _GUITHREADINFO()
        info.cbSize = ctypes.sizeof(_GUITHREADINFO)
        if not self.user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
            return 0
        return _handle_value(info.hwndFocus)

    def wait_for_modifiers_released(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if not any(int(self.user32.GetAsyncKeyState(key)) & 0x8000 for key in _MODIFIER_KEYS):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def send_unicode_text(self, text: str) -> bool:
        units = _utf16_units(text)
        if not units:
            return True
        inputs = (_INPUT * (len(units) * 2))()
        for index, unit in enumerate(units):
            for key_up in (False, True):
                item = inputs[index * 2 + int(key_up)]
                item.type = _INPUT_KEYBOARD
                item.union.ki.wVk = 0
                item.union.ki.wScan = unit
                item.union.ki.dwFlags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if key_up else 0)
                item.union.ki.time = 0
                item.union.ki.dwExtraInfo = 0
        sent = self.user32.SendInput(len(inputs), inputs, ctypes.sizeof(_INPUT))
        return int(sent) == len(inputs)

    def send_menu_mask(self) -> bool:
        """Mark a consumed Win gesture without producing text or swallowing Win-up."""

        inputs = (_INPUT * 2)()
        for index, key_up in enumerate((False, True)):
            inputs[index].type = _INPUT_KEYBOARD
            inputs[index].union.ki.wVk = _VK_MENU_MASK
            inputs[index].union.ki.wScan = 0
            inputs[index].union.ki.dwFlags = _KEYEVENTF_KEYUP if key_up else 0
            inputs[index].union.ki.time = 0
            inputs[index].union.ki.dwExtraInfo = _MENU_MASK_EXTRA_INFO
        return int(self.user32.SendInput(2, inputs, ctypes.sizeof(_INPUT))) == 2

    def make_window_nonactivating(self, hwnd: int, show: bool = True) -> None:
        if not hwnd:
            return
        # Tk's ``winfo_id()`` is often an inner child HWND.  Window styles and
        # activation semantics belong to its GA_ROOT wrapper, which is the one
        # Windows actually maps and brings to the foreground.
        top_level = self.get_root_window(hwnd) or hwnd
        handle = _HWND(top_level)
        style = int(self._get_window_long(handle, _GWL_EXSTYLE))
        self._set_window_long(handle, _GWL_EXSTYLE, style | _WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE)
        flags = _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_FRAMECHANGED
        if show:
            flags |= _SWP_SHOWWINDOW
        self.user32.SetWindowPos(
            handle,
            _HWND(-1),
            0,
            0,
            0,
            0,
            flags,
        )


class AlreadyRunningError(RuntimeError):
    """Raised by a context-managed guard when another host owns this session."""


class SingleInstanceGuard:
    """A same-session mutex guard for acquiring the host before engine warm-up."""

    def __init__(self, api: _Platform | None = None, name: str = _MUTEX_NAME) -> None:
        self._api = api or _WinApi()
        self._name = name
        self._handle = 0
        self._already_running = False

    @property
    def already_running(self) -> bool:
        return self._already_running

    @property
    def acquired(self) -> bool:
        return bool(self._handle)

    def acquire(self) -> bool:
        if self._handle:
            return True
        if not self._api.available:
            return False
        handle, exists = self._api.create_mutex(self._name)
        if not handle:
            return False
        if exists:
            self._api.close_handle(handle)
            self._already_running = True
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle:
            self._api.close_handle(self._handle)
            self._handle = 0

    def __enter__(self) -> "SingleInstanceGuard":
        if not self.acquire():
            raise AlreadyRunningError("ChineseASR dictation is already running in this Windows session.")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def request_existing_quit(api: _Platform | None = None) -> bool:
    """Ask an already-running host in this Windows session to invoke ``on_quit``.

    It is intentionally a named kernel event, not a socket or background service.
    Calling it does not create a host and returns ``False`` when no host has exposed
    the event yet.
    """

    platform = api or _WinApi()
    return bool(platform.available and platform.signal_existing_event(_QUIT_EVENT_NAME))


def is_running(api: _Platform | None = None) -> bool:
    """Return whether the same-session dictation mutex is currently held."""

    platform = api or _WinApi()
    return bool(platform.available and platform.named_mutex_exists(_MUTEX_NAME))


class WindowsHost:
    """Thread-safe tray/overlay shell for a caller-owned dictation controller."""

    is_available = staticmethod(is_available)

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_cancel: Callable[[], None],
        on_quit: Callable[[], None],
        *,
        api: _Platform | None = None,
        instance_guard: SingleInstanceGuard | None = None,
        tk_module: object | None = None,
        tray_factory: Callable[["WindowsHost"], object] | None = None,
    ) -> None:
        self.on_toggle = on_toggle
        self.on_cancel = on_cancel
        self.on_quit = on_quit
        self._api = api or _WinApi()
        self._guard = instance_guard or SingleInstanceGuard(self._api)
        self._tk_module = tk_module
        self._tray_factory = tray_factory
        self._lock = threading.RLock()
        self._events: queue.Queue[str] = queue.Queue()
        self._status = "中文听写正在启动"
        self._detail = ""
        self._recording = False
        self._busy = False
        self._error = False
        self._last_text = ""
        self._copy_requested = ""
        self._visible_until = time.monotonic() + 3.0
        self._overlay_visible = False
        self._shortcut_released = False
        self._win_keys: set[int] = set()
        self._suppress_h = False
        self._suppress_escape = False
        self._hook_proc: object | None = None
        self._hook_handle = 0
        self._quit_event_handle = 0
        self._quit_notified = False
        self._running = False
        self._close_requested = False
        self._finalized = False
        self._root = None
        self._overlay = None
        self._status_var = None
        self._detail_var = None
        self._tray = None

    @property
    def shortcut_released(self) -> bool:
        with self._lock:
            return self._shortcut_released

    @property
    def latest_text(self) -> str:
        with self._lock:
            return self._last_text

    def acquire_single_instance(self) -> bool:
        """Acquire the mutex and quit event before a controller warms an engine."""

        if not self._guard.acquire():
            return False
        if not self._quit_event_handle:
            self._quit_event_handle = self._api.create_quit_event(_QUIT_EVENT_NAME)
        if self._quit_event_handle:
            return True
        self._guard.release()
        return False

    def __enter__(self) -> "WindowsHost":
        if not self.acquire_single_instance():
            raise AlreadyRunningError("ChineseASR dictation is already running in this Windows session.")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def show(self, status: str, detail: str = "", recording: bool = False, error: bool = False) -> None:
        """Update the compact overlay from any thread without moving focus."""

        with self._lock:
            self._status = str(status)
            self._detail = str(detail)
            self._recording = bool(recording)
            self._error = bool(error)
            self._visible_until = (
                None if (self._error or self._recording or self._busy) else time.monotonic() + 3.0
            )
        self._events.put("refresh")

    def set_busy(self, busy: bool) -> None:
        """Control Esc interception independently from the recording visual state."""

        with self._lock:
            self._busy = bool(busy)
            self._visible_until = (
                None if (self._busy or self._recording or self._error) else time.monotonic() + 3.0
            )
        self._events.put("refresh")

    def set_last_text(self, text: str) -> None:
        """Retain recognized text in memory for explicit tray/overlay copying only."""

        with self._lock:
            self._last_text = str(text)

    def copy_text(self, text: str | None = None) -> bool:
        """Queue an explicit clipboard write on the UI thread; never auto-copies text."""

        with self._lock:
            value = self._last_text if text is None else str(text)
            if not value or self._root is None:
                return False
            self._copy_requested = value
        self._events.put("copy")
        return True

    def set_shortcut_released(self, released: bool) -> None:
        """Temporarily pass Win+H to Windows while keeping the tray host alive."""

        with self._lock:
            self._shortcut_released = bool(released)
            self._suppress_h = False
        self.show(
            "Win+H 已交还系统" if released else "Win+H 已由中文听写接管",
            "可从托盘随时切换",
        )

    def capture_target(self) -> TargetWindow:
        foreground = self._api.get_foreground_window()
        if not foreground:
            return TargetWindow(0, 0)
        root = self._api.get_root_window(foreground)
        focus = self._api.get_focus_window(foreground) or foreground
        return TargetWindow(root, focus)

    def insert_text(self, text: str, target: TargetWindow) -> bool:
        """Insert Unicode text only while the original foreground/focus pair remains."""

        if not text:
            return True
        if not self._target_is_current(target):
            return False
        if not self._api.wait_for_modifiers_released(0.5):
            return False
        # Recheck after the trigger modifiers are physically released and directly
        # before SendInput.  The function never calls SetForegroundWindow.
        if not self._target_is_current(target):
            return False
        return self._api.send_unicode_text(text)

    def _target_is_current(self, target: TargetWindow) -> bool:
        if not target.root or not target.focus:
            return False
        foreground = self._api.get_foreground_window()
        if not foreground or self._api.get_root_window(foreground) != target.root:
            return False
        focus = self._api.get_focus_window(foreground) or foreground
        return focus == target.focus

    def run(self) -> bool:
        """Run the desktop event loop.  Returns ``False`` if no primary host exists."""

        if not self._api.available or not self.acquire_single_instance():
            return False
        try:
            self._create_overlay()
            self._running = True
            self._install_hook()
            self._start_tray()
            self._poll()
            self._root.mainloop()
            return True
        finally:
            self._finalize_close()

    def close(self) -> None:
        """Request a safe UI shutdown; it is safe for controller worker threads."""

        self._close_requested = True
        self._events.put("close")
        if not self._running:
            self._finalize_close()

    def _create_overlay(self) -> None:
        if self._tk_module is None:
            import tkinter as tk

            self._tk_module = tk
        tk = self._tk_module
        self._root = tk.Tk()
        self._root.withdraw()
        self._overlay = tk.Toplevel(self._root)
        self._overlay.overrideredirect(True)
        self._overlay.attributes("-topmost", True)
        self._overlay.configure(bg="#202124")
        frame = tk.Frame(self._overlay, bg="#202124", padx=12, pady=8)
        frame.pack(fill="both", expand=True)
        self._status_var = tk.StringVar(value=self._status)
        self._detail_var = tk.StringVar(value=self._detail)
        tk.Label(
            frame,
            textvariable=self._status_var,
            anchor="w",
            justify="left",
            fg="#f8f9fa",
            bg="#202124",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(fill="x")
        tk.Label(
            frame,
            textvariable=self._detail_var,
            anchor="w",
            justify="left",
            wraplength=336,
            fg="#d0d7de",
            bg="#202124",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", pady=(3, 0))
        width, height = 360, 78
        x = max(0, (self._overlay.winfo_screenwidth() - width) // 2)
        y = max(0, self._overlay.winfo_screenheight() - height - 72)
        self._overlay.geometry(f"{width}x{height}+{x}+{y}")
        self._overlay.protocol("WM_DELETE_WINDOW", lambda: self._events.put("quit"))
        # Apply WS_EX_NOACTIVATE before the first show, not only after it.
        self._overlay.withdraw()
        self._api.make_window_nonactivating(int(self._overlay.winfo_id()), show=False)
        self._render_overlay()

    def _install_hook(self) -> None:
        def callback(n_code: int, w_param: int, l_param: int) -> int:
            try:
                if n_code >= 0:
                    raw = ctypes.cast(l_param, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                    event = KeyboardEvent(
                        vk_code=int(raw.vkCode),
                        message=int(w_param),
                        injected=bool(raw.flags & (_LLKHF_INJECTED | _LLKHF_LOWER_IL_INJECTED)),
                    )
                    if self._handle_keyboard_event(event):
                        return 1
            except Exception:
                # A hook must fail open; the tray remains available for recovery.
                pass
            return self._api.call_next_hook(self._hook_handle, n_code, w_param, l_param)

        self._hook_proc = _HOOKPROC(callback)
        self._hook_handle = self._api.install_keyboard_hook(self._hook_proc)
        if not self._hook_handle:
            self.show("Win+H 快捷键不可用", "可通过托盘开始或停止听写", error=True)

    def _handle_keyboard_event(self, event: KeyboardEvent) -> bool:
        """Return whether a low-level keyboard event must be suppressed."""

        if event.injected:
            return False
        down = event.message in (_WM_KEYDOWN, _WM_SYSKEYDOWN)
        up = event.message in (_WM_KEYUP, _WM_SYSKEYUP)
        with self._lock:
            if event.vk_code in _WIN_KEYS:
                if down:
                    self._win_keys.add(event.vk_code)
                    return False
                if up:
                    self._win_keys.discard(event.vk_code)
                    return False

            if event.vk_code == _VK_H:
                if self._shortcut_released:
                    return False
                if down and self._win_keys:
                    if not self._suppress_h:
                        self._suppress_h = True
                        # H is swallowed, so mark the Win gesture with a private,
                        # non-text virtual key.  Win-up still reaches Windows, which
                        # avoids leaving later ordinary keys in a stuck Win state.
                        try:
                            self._api.send_menu_mask()
                        except Exception:
                            pass
                        self._events.put("toggle")
                    return True
                if up and self._suppress_h:
                    self._suppress_h = False
                    return True

            if event.vk_code == _VK_ESCAPE and self._busy:
                if down:
                    if not self._suppress_escape:
                        self._suppress_escape = True
                        self._events.put("cancel")
                    return True
                if up and self._suppress_escape:
                    self._suppress_escape = False
                    return True
        return False

    def _poll(self) -> None:
        if self._close_requested:
            self._finalize_close()
            return
        if self._quit_event_handle and self._api.event_is_signaled(self._quit_event_handle):
            self._events.put("quit")
        self._dispatch_pending_events()
        if self._close_requested:
            self._finalize_close()
            return
        self._render_overlay()
        if self._running and not self._close_requested:
            self._root.after(25, self._poll)

    def _dispatch_pending_events(self) -> None:
        while True:
            try:
                action = self._events.get_nowait()
            except queue.Empty:
                return
            if action == "close":
                self._close_requested = True
                continue
            if action == "refresh":
                continue
            if action == "toggle":
                self._invoke_callback(self.on_toggle, "切换听写失败")
            elif action == "cancel":
                self._invoke_callback(self.on_cancel, "取消听写失败")
            elif action == "copy":
                self._copy_on_ui_thread()
            elif action == "release":
                self.set_shortcut_released(not self.shortcut_released)
            elif action == "quit" and not self._quit_notified:
                self._quit_notified = True
                self._invoke_callback(self.on_quit, "退出听写失败", close_on_error=True)

    def _invoke_callback(self, callback: Callable[[], None], failure_status: str, *, close_on_error: bool = False) -> None:
        try:
            callback()
        except Exception:
            self.show(failure_status, "可从托盘退出后重试", error=True)
            if close_on_error:
                self.close()

    def _copy_on_ui_thread(self) -> None:
        with self._lock:
            text = self._copy_requested
            self._copy_requested = ""
        if not text or self._root is None:
            return
        try:
            self._root.clipboard_clear()
            self._root.clipboard_append(text)
            self._root.update()
            self.show("已复制识别文字", "可粘贴到任意输入框")
        except Exception:
            self.show("无法复制文字", "请在托盘重试", error=True)

    def _render_overlay(self) -> None:
        if self._overlay is None:
            return
        with self._lock:
            status = self._status
            detail = self._detail
            error = self._error
            visible_until = self._visible_until
            should_show = self._busy or self._recording or error or (
                visible_until is not None and time.monotonic() < visible_until
            )
        if len(detail) > 150:
            detail = detail[:147] + "…（托盘可复制全文）"
        try:
            self._status_var.set(status)
            self._detail_var.set(detail)
            self._overlay.configure(bg="#6e1f1f" if error else "#202124")
            if should_show and not self._overlay_visible:
                self._overlay.deiconify()
                self._api.make_window_nonactivating(int(self._overlay.winfo_id()))
                self._overlay_visible = True
                self._schedule_overlay_style_reapply()
            elif not should_show and self._overlay_visible:
                self._overlay.withdraw()
                self._overlay_visible = False
        except Exception:
            pass

    def _schedule_overlay_style_reapply(self) -> None:
        """Let Tk finish wrapping a Toplevel before styling its actual GA_ROOT."""

        if self._root is None:
            return
        for delay_ms in (0, 75, 250):
            try:
                self._root.after(delay_ms, self._reapply_overlay_nonactivation)
            except Exception:
                return

    def _reapply_overlay_nonactivation(self) -> None:
        if self._overlay is None or not self._overlay_visible:
            return
        try:
            # ``show=False`` only refreshes the real wrapper's style; it never
            # reveals an overlay that the normal visibility policy has hidden.
            self._api.make_window_nonactivating(int(self._overlay.winfo_id()), show=False)
        except Exception:
            pass

    def _start_tray(self) -> None:
        try:
            self._tray = self._tray_factory(self) if self._tray_factory else self._build_pystray_icon()
            self._tray.run_detached()
        except Exception:
            self._tray = None
            self.show("托盘图标不可用", "Win+H 仍可开始或停止听写", error=True)

    def _build_pystray_icon(self) -> object:
        import pystray
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((8, 5, 56, 59), radius=13, fill=(38, 132, 255, 255))
        draw.ellipse((25, 13, 39, 37), fill=(255, 255, 255, 255))
        draw.rectangle((29, 35, 35, 48), fill=(255, 255, 255, 255))
        menu = pystray.Menu(
            pystray.MenuItem("开始/停止听写", lambda *_: self._events.put("toggle")),
            pystray.MenuItem("取消本次听写", lambda *_: self._events.put("cancel")),
            pystray.MenuItem("复制最近识别文字", lambda *_: self.copy_text()),
            pystray.MenuItem("暂时释放/恢复 Win+H", lambda *_: self._events.put("release")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出中文听写", lambda *_: self._events.put("quit")),
        )
        return pystray.Icon("ChineseASRDictation", image, "中文听写", menu)

    def _finalize_close(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._running = False
        if self._hook_handle:
            self._api.uninstall_keyboard_hook(self._hook_handle)
            self._hook_handle = 0
        if self._tray is not None:
            try:
                self._tray.stop()
            except Exception:
                pass
            self._tray = None
        if self._overlay is not None:
            try:
                self._overlay.destroy()
            except Exception:
                pass
            self._overlay = None
        self._overlay_visible = False
        if self._root is not None:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass
            self._root = None
        if self._quit_event_handle:
            self._api.close_handle(self._quit_event_handle)
            self._quit_event_handle = 0
        self._guard.release()
