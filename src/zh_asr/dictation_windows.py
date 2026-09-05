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
_WM_QUIT = 0x0012
_VK_ESCAPE = 0x1B
_VK_H = 0x48
_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C
_VK_LSHIFT = 0xA0
_VK_RSHIFT = 0xA1
_VK_LCONTROL = 0xA2
_VK_RCONTROL = 0xA3
_VK_LMENU = 0xA4
_VK_RMENU = 0xA5
_VK_MENU_MASK = 0xE8
_HOST_INPUT_EXTRA_INFO = 0x43415352  # "CASR": only this host's injected input.
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
_PM_NOREMOVE = 0

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
_CTRL_KEYS = (_VK_CONTROL, _VK_LCONTROL, _VK_RCONTROL)
_ALT_KEYS = (_VK_MENU, _VK_LMENU, _VK_RMENU)
_SHIFT_KEYS = (_VK_SHIFT, _VK_LSHIFT, _VK_RSHIFT)


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


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", _HWND),
        ("message", wintypes.UINT),
        ("wParam", _WPARAM),
        ("lParam", _LPARAM),
        ("time", wintypes.DWORD),
        ("pt", _POINT),
        ("lPrivate", wintypes.DWORD),
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
    extra_info: int = 0


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
    def current_thread_id(self) -> int: ...
    def ensure_message_queue(self) -> None: ...
    def pump_messages(self) -> None: ...
    def post_thread_quit(self, thread_id: int) -> bool: ...
    def get_foreground_window(self) -> int: ...
    def get_root_window(self, hwnd: int) -> int: ...
    def get_window_process_id(self, hwnd: int) -> int: ...
    def current_process_id(self) -> int: ...
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
        self.user32.PeekMessageW.argtypes = [
            ctypes.POINTER(_MSG),
            _HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.user32.PeekMessageW.restype = _BOOL
        self.user32.GetMessageW.argtypes = [ctypes.POINTER(_MSG), _HWND, wintypes.UINT, wintypes.UINT]
        self.user32.GetMessageW.restype = ctypes.c_int
        self.user32.TranslateMessage.argtypes = [ctypes.POINTER(_MSG)]
        self.user32.TranslateMessage.restype = _BOOL
        self.user32.DispatchMessageW.argtypes = [ctypes.POINTER(_MSG)]
        self.user32.DispatchMessageW.restype = _LRESULT
        self.user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, _WPARAM, _LPARAM]
        self.user32.PostThreadMessageW.restype = _BOOL
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
        self.kernel32.GetCurrentThreadId.argtypes = []
        self.kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        self.kernel32.GetCurrentProcessId.argtypes = []
        self.kernel32.GetCurrentProcessId.restype = wintypes.DWORD
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

    def current_thread_id(self) -> int:
        return int(self.kernel32.GetCurrentThreadId())

    def ensure_message_queue(self) -> None:
        message = _MSG()
        self.user32.PeekMessageW(ctypes.byref(message), None, 0, 0, _PM_NOREMOVE)

    def pump_messages(self) -> None:
        message = _MSG()
        while True:
            result = int(self.user32.GetMessageW(ctypes.byref(message), None, 0, 0))
            if result <= 0:
                return
            self.user32.TranslateMessage(ctypes.byref(message))
            self.user32.DispatchMessageW(ctypes.byref(message))

    def post_thread_quit(self, thread_id: int) -> bool:
        return bool(self.user32.PostThreadMessageW(thread_id, _WM_QUIT, 0, 0))

    def get_foreground_window(self) -> int:
        return _handle_value(self.user32.GetForegroundWindow())

    def get_root_window(self, hwnd: int) -> int:
        if not hwnd:
            return 0
        return _handle_value(self.user32.GetAncestor(_HWND(hwnd), _GA_ROOT)) or hwnd

    def get_window_process_id(self, hwnd: int) -> int:
        if not hwnd:
            return 0
        process_id = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(_HWND(hwnd), ctypes.byref(process_id))
        return int(process_id.value)

    def current_process_id(self) -> int:
        return int(self.kernel32.GetCurrentProcessId())

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
                item.union.ki.dwExtraInfo = _HOST_INPUT_EXTRA_INFO
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
            inputs[index].union.ki.dwExtraInfo = _HOST_INPUT_EXTRA_INFO
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
        on_hide: Callable[[], None] | None = None,
        on_device_change: Callable[[str | None], None] | None = None,
        on_refresh_devices: Callable[[], None] | None = None,
        api: _Platform | None = None,
        instance_guard: SingleInstanceGuard | None = None,
        tk_module: object | None = None,
        tray_factory: Callable[["WindowsHost"], object] | None = None,
    ) -> None:
        self.on_toggle = on_toggle
        self.on_cancel = on_cancel
        self.on_quit = on_quit
        self.on_hide = on_hide or (lambda: None)
        self.on_device_change = on_device_change or (lambda _value: None)
        self.on_refresh_devices = on_refresh_devices or (lambda: None)
        self._api = api or _WinApi()
        self._process_id = self._api.current_process_id()
        self._guard = instance_guard or SingleInstanceGuard(self._api)
        self._tk_module = tk_module
        self._tray_factory = tray_factory
        self._lock = threading.RLock()
        self._events: queue.Queue[str] = queue.Queue()
        self._ui_calls: queue.Queue[Callable[[], None]] = queue.Queue()
        self._status = "中文听写正在启动"
        self._detail = ""
        self._recording = False
        self._busy = False
        self._error = False
        self._last_text = ""
        self._copy_requested = ""
        self._overlay_visible = False
        self._panel_open = False
        self._shortcut_released = False
        self._win_keys: set[int] = set()
        self._ctrl_keys: set[int] = set()
        self._alt_keys: set[int] = set()
        self._shift_keys: set[int] = set()
        self._suppress_h = False
        self._suppress_escape = False
        self._hook_proc: object | None = None
        self._hook_handle = 0
        self._hook_thread: threading.Thread | None = None
        self._hook_thread_id = 0
        self._hook_ready = threading.Event()
        self._hook_stop = threading.Event()
        self._hook_error = ""
        self._quit_event_handle = 0
        self._quit_notified = False
        self._running = False
        self._close_requested = False
        self._finalized = False
        self._root = None
        self._overlay = None
        self._ui_thread_id: int | None = None
        self._own_window_roots: set[int] = set()
        self._last_external_target: TargetWindow | None = None
        self._status_var = None
        self._detail_var = None
        self._device_var = None
        self._device_menu = None
        self._record_canvas = None
        self._status_label = None
        self._microphones: list[dict[str, str | None]] = []
        self._selected_microphone: str | None = None
        self._drag_offset: tuple[int, int] | None = None
        self._tray = None
        self._tray_thread: threading.Thread | None = None

    @property
    def shortcut_released(self) -> bool:
        with self._lock:
            return self._shortcut_released

    @property
    def latest_text(self) -> str:
        with self._lock:
            return self._last_text

    def post_to_ui(self, callback: Callable[[], None]) -> None:
        """Run a small callback on Tk's thread without making Tk cross-thread calls."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        self._ui_calls.put(callback)
        self._events.put("ui")

    def open_panel(self) -> None:
        """Show the compact panel; background status updates never do this implicitly."""

        with self._lock:
            self._panel_open = True
        if self._is_ui_thread():
            self._render_overlay()
        else:
            self._events.put("open_panel")

    def hide_panel(self) -> None:
        """Immediately hide the panel while the tray host and hotkeys remain available."""

        with self._lock:
            self._panel_open = False
        if self._is_ui_thread():
            self._hide_panel_ui()
        else:
            self._events.put("hide_panel")

    def set_microphones(
        self,
        microphones: list[dict[str, str | None]],
        selected: str | None = None,
    ) -> None:
        """Replace panel device choices without firing a device-change callback."""

        choices: list[dict[str, str | None]] = []
        for item in microphones:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            normalized_value = None if value is None else str(value)
            label = str(item.get("label") or normalized_value or "Windows 默认麦克风")
            choices.append({"value": normalized_value, "label": label})
        with self._lock:
            self._microphones = choices
            self._selected_microphone = None if selected is None else str(selected)
        if self._is_ui_thread():
            self._refresh_microphone_menu_ui()
        else:
            self._events.put("microphones")

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
        """Update panel content from any thread without reopening a hidden panel."""

        with self._lock:
            self._status = str(status)
            self._detail = str(detail)
            self._recording = bool(recording)
            self._error = bool(error)
        self._events.put("refresh")

    def set_busy(self, busy: bool) -> None:
        """Control Esc interception independently from the recording visual state."""

        with self._lock:
            self._busy = bool(busy)
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
            "Win+H、Ctrl+Win+H 已交还系统" if released else "Win+H、Ctrl+Win+H 已由中文听写接管",
            "可从托盘随时切换",
        )

    def capture_target(self) -> TargetWindow:
        target = self._capture_external_target()
        if target is not None:
            return target
        with self._lock:
            return self._last_external_target or TargetWindow(0, 0)

    def _capture_external_target(self) -> TargetWindow | None:
        foreground = self._api.get_foreground_window()
        if not foreground or self._is_own_window(foreground):
            return None
        root = self._api.get_root_window(foreground)
        focus = self._api.get_focus_window(foreground) or foreground
        if not root or not focus:
            return None
        target = TargetWindow(root, focus)
        with self._lock:
            self._last_external_target = target
        return target

    def _remember_external_target(self) -> None:
        self._capture_external_target()

    def _is_own_window(self, hwnd: int) -> bool:
        if not hwnd:
            return False
        with self._lock:
            own_roots = set(self._own_window_roots)
        if hwnd in own_roots:
            return True
        root = self._api.get_root_window(hwnd)
        if root and root in own_roots:
            return True
        return self._api.get_window_process_id(hwnd) == self._process_id

    def insert_text(self, text: str, target: TargetWindow) -> bool:
        """Insert Unicode text only while the original foreground/focus pair remains."""

        if not text:
            return True
        if self._is_own_window(target.root):
            return False
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
        self._ui_thread_id = threading.get_ident()
        self._root.withdraw()
        self._overlay = tk.Toplevel(self._root)
        self._overlay.withdraw()
        self._overlay.overrideredirect(True)
        self._overlay.attributes("-topmost", True)
        self._overlay.configure(bg="#ffffff", highlightthickness=1, highlightbackground="#d8e3da")

        header = tk.Frame(self._overlay, bg="#f3f8f4", height=34)
        header.pack(fill="x")
        header.pack_propagate(False)
        title = tk.Label(
            header,
            text="中文听写",
            anchor="w",
            fg="#166534",
            bg="#f3f8f4",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        title.pack(side="left", fill="both", expand=True, padx=(12, 0))
        close = tk.Label(
            header,
            text="×",
            width=3,
            fg="#64748b",
            bg="#f3f8f4",
            font=("Segoe UI", 15),
            takefocus=False,
        )
        close.pack(side="right", fill="y")
        for widget in (header, title):
            widget.bind("<ButtonPress-1>", self._begin_drag)
            widget.bind("<B1-Motion>", self._drag_panel)
        close.bind("<ButtonRelease-1>", lambda _event: self._hide_from_panel())

        frame = tk.Frame(self._overlay, bg="#ffffff", padx=14, pady=7)
        frame.pack(fill="both", expand=True)
        self._status_var = tk.StringVar(value=self._status)
        self._detail_var = tk.StringVar(value=self._detail)
        self._status_label = tk.Label(
            frame,
            textvariable=self._status_var,
            anchor="center",
            justify="center",
            fg="#166534",
            bg="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._status_label.pack(fill="x")

        self._record_canvas = tk.Canvas(
            frame,
            width=62,
            height=62,
            bg="#ffffff",
            highlightthickness=0,
            takefocus=False,
            cursor="hand2",
        )
        self._record_canvas.pack(pady=(2, 1))
        self._record_canvas.bind("<ButtonRelease-1>", lambda _event: self._toggle_from_panel())

        tk.Label(
            frame,
            textvariable=self._detail_var,
            anchor="center",
            justify="center",
            wraplength=276,
            fg="#64748b",
            bg="#ffffff",
            font=("Microsoft YaHei UI", 8),
        ).pack(fill="x", pady=(0, 4))

        device_row = tk.Frame(frame, bg="#ffffff")
        device_row.pack(fill="x")
        tk.Label(
            device_row,
            text="麦克风",
            fg="#64748b",
            bg="#ffffff",
            font=("Microsoft YaHei UI", 8),
        ).pack(side="left")
        self._device_var = tk.StringVar(value="选择麦克风")
        selector = tk.Menubutton(
            device_row,
            width=20,
            textvariable=self._device_var,
            anchor="w",
            relief="flat",
            bd=0,
            padx=7,
            pady=2,
            fg="#334155",
            bg="#f1f5f2",
            activebackground="#e2f2e7",
            font=("Microsoft YaHei UI", 9),
            takefocus=False,
        )
        selector.pack(side="left", fill="x", expand=True, padx=(7, 5))
        self._device_menu = tk.Menu(selector, tearoff=False, bg="#ffffff", activebackground="#dcfce7")
        selector.configure(menu=self._device_menu)
        refresh = tk.Label(
            device_row,
            text="刷新",
            fg="#16803b",
            bg="#ffffff",
            cursor="hand2",
            font=("Microsoft YaHei UI", 9),
            takefocus=False,
        )
        refresh.pack(side="right")
        refresh.bind("<ButtonRelease-1>", lambda _event: self._refresh_devices_from_panel())

        width, height = 304, 180
        x = max(0, (self._overlay.winfo_screenwidth() - width) // 2)
        y = max(0, self._overlay.winfo_screenheight() - height - 84)
        self._overlay.geometry(f"{width}x{height}+{x}+{y}")
        self._overlay.protocol("WM_DELETE_WINDOW", self._hide_from_panel)
        # Apply WS_EX_NOACTIVATE before the first show, not only after it.
        self._overlay.withdraw()
        self._api.make_window_nonactivating(int(self._overlay.winfo_id()), show=False)
        self._refresh_own_window_roots_ui()
        self._refresh_microphone_menu_ui()
        self._paint_record_button_ui()

    def _is_ui_thread(self) -> bool:
        return self._ui_thread_id is not None and self._ui_thread_id == threading.get_ident()

    def _refresh_own_window_roots_ui(self) -> None:
        roots: set[int] = set()
        for widget in (self._root, self._overlay):
            if widget is None:
                continue
            try:
                root = self._api.get_root_window(int(widget.winfo_id()))
            except Exception:
                continue
            if root:
                roots.add(root)
        with self._lock:
            self._own_window_roots = roots

    def _begin_drag(self, event) -> None:
        if self._overlay is None:
            return
        self._drag_offset = (event.x_root - self._overlay.winfo_x(), event.y_root - self._overlay.winfo_y())

    def _drag_panel(self, event) -> None:
        if self._overlay is None or self._drag_offset is None:
            return
        offset_x, offset_y = self._drag_offset
        self._overlay.geometry(f"+{max(0, event.x_root - offset_x)}+{max(0, event.y_root - offset_y)}")

    def _toggle_from_panel(self) -> None:
        self._remember_external_target()
        self._invoke_callback(self.on_toggle, "切换听写失败")

    def _hide_from_panel(self) -> None:
        self._hide_panel_ui()
        self._invoke_callback(self.on_hide, "收起听写窗口失败")

    def _open_panel_ui(self) -> None:
        with self._lock:
            self._panel_open = True
        self._render_overlay()

    def _hide_panel_ui(self) -> None:
        with self._lock:
            self._panel_open = False
        if self._overlay is not None and self._overlay_visible:
            try:
                self._overlay.withdraw()
            except Exception:
                pass
        self._overlay_visible = False

    def _refresh_devices_from_panel(self) -> None:
        self._invoke_callback(self.on_refresh_devices, "刷新麦克风失败")

    def _select_microphone(self, value: str | None) -> None:
        with self._lock:
            self._selected_microphone = value
        self._refresh_microphone_menu_ui()
        try:
            self.on_device_change(value)
        except Exception:
            self.show("切换麦克风失败", "可重新选择设备", error=True)

    def _refresh_microphone_menu_ui(self) -> None:
        if self._device_menu is None or self._device_var is None:
            return
        with self._lock:
            choices = list(self._microphones)
            selected = self._selected_microphone
        self._device_menu.delete(0, "end")
        selected_label = None
        for item in choices:
            value = item["value"]
            label = str(item["label"])
            if value == selected:
                selected_label = label
            self._device_menu.add_command(label=label, command=lambda item_value=value: self._select_microphone(item_value))
        if not choices:
            self._device_menu.add_command(label="未发现可用麦克风", state="disabled")
        display = selected_label or ("Windows 默认麦克风" if selected is None else "选择麦克风")
        self._device_var.set(display if len(display) <= 24 else display[:21] + "…")

    def _paint_record_button_ui(self) -> None:
        if self._record_canvas is None:
            return
        with self._lock:
            active = self._recording
        canvas = self._record_canvas
        canvas.delete("all")
        fill = "#16a34a" if active else "#ecfdf3"
        ink = "#ffffff" if active else "#15803d"
        canvas.create_oval(4, 4, 58, 58, fill=fill, outline="#22a65a", width=2)
        canvas.create_oval(25, 14, 37, 34, fill=ink, outline=ink)
        canvas.create_line(23, 31, 23, 36, 41, 36, 41, 31, fill=ink, width=2)
        canvas.create_line(32, 36, 32, 44, fill=ink, width=2)
        canvas.create_line(25, 45, 39, 45, fill=ink, width=2)

    def _install_hook(self) -> None:
        """Install the low-level hook on its own message-pump thread.

        Tk callbacks may briefly block while opening an input stream.  Keeping the
        hook off that thread prevents Windows from timing out and silently removing
        it, while the callback itself still only puts a short action into a queue.
        """

        if self._hook_thread is not None and self._hook_thread.is_alive():
            return
        self._hook_stop.clear()
        self._hook_ready.clear()
        self._hook_error = ""
        self._hook_handle = 0
        self._hook_thread_id = 0
        self._hook_thread = threading.Thread(
            target=self._hook_worker,
            name="chineseasr-win-hotkey",
            daemon=True,
        )
        self._hook_thread.start()
        if not self._hook_ready.wait(1.5):
            self._hook_error = "keyboard hook worker did not become ready"
            self._stop_hook_worker()
        if not self._hook_handle:
            self.show("听写快捷键不可用", "可通过托盘开始或停止听写", error=True)

    def _hook_worker(self) -> None:
        handle = 0
        try:
            self._api.ensure_message_queue()
            with self._lock:
                self._hook_thread_id = self._api.current_thread_id()
            self._hook_proc = _HOOKPROC(self._low_level_hook_callback)
            handle = self._api.install_keyboard_hook(self._hook_proc)
            with self._lock:
                self._hook_handle = handle
            if not handle:
                self._hook_error = "SetWindowsHookExW returned no hook"
        except Exception as exc:
            self._hook_error = f"{type(exc).__name__} while installing keyboard hook"
        finally:
            self._hook_ready.set()

        if not handle:
            return
        try:
            if not self._hook_stop.is_set():
                self._api.pump_messages()
        finally:
            self._api.uninstall_keyboard_hook(handle)
            with self._lock:
                if self._hook_handle == handle:
                    self._hook_handle = 0
                self._hook_thread_id = 0

    def _stop_hook_worker(self) -> None:
        self._hook_stop.set()
        with self._lock:
            thread = self._hook_thread
            thread_id = self._hook_thread_id
        if thread_id:
            try:
                self._api.post_thread_quit(thread_id)
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread is not None and not thread.is_alive():
            self._hook_thread = None

    def _low_level_hook_callback(self, n_code: int, w_param: int, l_param: int) -> int:
        try:
            if n_code >= 0:
                raw = ctypes.cast(l_param, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                event = KeyboardEvent(
                    vk_code=int(raw.vkCode),
                    message=int(w_param),
                    injected=bool(raw.flags & (_LLKHF_INJECTED | _LLKHF_LOWER_IL_INJECTED)),
                    extra_info=int(raw.dwExtraInfo),
                )
                if self._handle_keyboard_event(event):
                    return 1
        except Exception:
            # A hook must fail open; the tray remains available for recovery.
            pass
        return self._api.call_next_hook(self._hook_handle, n_code, w_param, l_param)

    def _handle_keyboard_event(self, event: KeyboardEvent) -> bool:
        """Return whether a low-level keyboard event must be suppressed."""

        if event.injected and event.extra_info == _HOST_INPUT_EXTRA_INFO:
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

            if event.vk_code in _CTRL_KEYS:
                if down:
                    self._ctrl_keys.add(event.vk_code)
                elif up:
                    self._ctrl_keys.discard(event.vk_code)
                return False

            if event.vk_code in _ALT_KEYS:
                if down:
                    self._alt_keys.add(event.vk_code)
                elif up:
                    self._alt_keys.discard(event.vk_code)
                return False

            if event.vk_code in _SHIFT_KEYS:
                if down:
                    self._shift_keys.add(event.vk_code)
                elif up:
                    self._shift_keys.discard(event.vk_code)
                return False

            if event.vk_code == _VK_H:
                if self._shortcut_released:
                    return False
                if down and self._win_keys and not self._alt_keys and not self._shift_keys:
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
        self._drain_ui_calls()
        if self._quit_event_handle and self._api.event_is_signaled(self._quit_event_handle):
            self._events.put("quit")
        self._dispatch_pending_events()
        if self._close_requested:
            self._finalize_close()
            return
        self._render_overlay()
        if self._running and not self._close_requested:
            self._root.after(25, self._poll)

    def _drain_ui_calls(self) -> None:
        while True:
            try:
                callback = self._ui_calls.get_nowait()
            except queue.Empty:
                return
            try:
                callback()
            except Exception:
                self.show("界面更新失败", "可继续使用托盘快捷键", error=True)

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
            if action == "ui":
                continue
            if action == "open_panel":
                self._open_panel_ui()
                continue
            if action == "hide_panel":
                self._hide_panel_ui()
                continue
            if action == "microphones":
                self._refresh_microphone_menu_ui()
                continue
            if action == "toggle":
                self._remember_external_target()
                self._open_panel_ui()
                self._invoke_callback(self.on_toggle, "切换听写失败")
            elif action == "cancel":
                self._invoke_callback(self.on_cancel, "取消听写失败")
            elif action == "copy":
                self._copy_on_ui_thread()
            elif action == "release":
                self.set_shortcut_released(not self.shortcut_released)
            elif action == "quit" and not self._quit_notified:
                self._quit_notified = True
                self._hide_panel_ui()
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
            should_show = self._panel_open
        if len(detail) > 92:
            detail = detail[:89] + "…（托盘可复制全文）"
        try:
            self._status_var.set(status)
            self._detail_var.set(detail)
            if self._status_label is not None:
                self._status_label.configure(fg="#b91c1c" if error else "#166534")
            self._paint_record_button_ui()
            if should_show and not self._overlay_visible:
                self._overlay.deiconify()
                self._api.make_window_nonactivating(int(self._overlay.winfo_id()))
                self._overlay_visible = True
                self._refresh_own_window_roots_ui()
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
            self._refresh_own_window_roots_ui()
        except Exception:
            pass

    def _start_tray(self) -> None:
        try:
            self._tray = self._tray_factory(self) if self._tray_factory else self._build_pystray_icon()
            self._tray_thread = threading.Thread(
                target=self._tray.run,
                name="chineseasr-tray",
                daemon=True,
            )
            self._tray_thread.start()
        except Exception:
            self._tray = None
            self._tray_thread = None
            self.show("托盘图标不可用", "Win+H 或 Ctrl+Win+H 仍可开始或停止听写", error=True)

    def _build_pystray_icon(self) -> object:
        import pystray
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((8, 5, 56, 59), radius=13, fill=(38, 132, 255, 255))
        draw.ellipse((25, 13, 39, 37), fill=(255, 255, 255, 255))
        draw.rectangle((29, 35, 35, 48), fill=(255, 255, 255, 255))
        menu = pystray.Menu(
            pystray.MenuItem("显示并开始/暂停听写", lambda *_: self._events.put("toggle")),
            pystray.MenuItem("取消本次听写", lambda *_: self._events.put("cancel")),
            pystray.MenuItem("复制最近识别文字", lambda *_: self.copy_text()),
            pystray.MenuItem("暂时释放/恢复两组快捷键", lambda *_: self._events.put("release")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出中文听写", lambda *_: self._events.put("quit")),
        )
        return pystray.Icon("ChineseASRDictation", image, "中文听写", menu)

    def _finalize_close(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._running = False
        # Remove the user-visible panel first.  Controller/model cleanup may take
        # a while, but it must never leave an "exiting" strip in front of the user.
        if self._overlay is not None:
            try:
                self._overlay.withdraw()
                self._overlay.destroy()
            except Exception:
                pass
            self._overlay = None
        self._overlay_visible = False
        with self._lock:
            self._panel_open = False
            self._own_window_roots.clear()
        if self._root is not None:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass
            self._root = None
        self._ui_thread_id = None
        self._stop_hook_worker()
        if self._tray is not None:
            try:
                self._tray.stop()
            except Exception:
                pass
            self._tray = None
        if self._tray_thread is not None and self._tray_thread is not threading.current_thread():
            self._tray_thread.join(timeout=0.5)
        self._tray_thread = None
        if self._quit_event_handle:
            self._api.close_handle(self._quit_event_handle)
            self._quit_event_handle = 0
        self._guard.release()
