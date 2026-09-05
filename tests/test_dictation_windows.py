from __future__ import annotations

import unittest
import time

from zh_asr.dictation_windows import (
    KeyboardEvent,
    WindowsHost,
    _VK_ESCAPE,
    _VK_H,
    _VK_LWIN,
    _WM_KEYDOWN,
    _WM_KEYUP,
    _utf16_units,
    is_running,
    request_existing_quit,
)


class FakeWindowsApi:
    available = True

    def __init__(self) -> None:
        self.foreground = 101
        self.focus = 201
        self.root = 1001
        self.modifiers_released = True
        self.wait_calls = 0
        self.sent_text: list[str] = []
        self.mutex_held = False
        self.mutex_handle = 0
        self.next_handle = 10
        self.event_handles: dict[int, str] = {}
        self.exposed_events: set[str] = set()
        self.signaled_events: set[str] = set()
        self.hook_callback = None
        self.unhooked: list[int] = []
        self.menu_masks = 0
        self.nonactivation_calls: list[tuple[int, bool]] = []

    def _new_handle(self) -> int:
        self.next_handle += 1
        return self.next_handle

    def create_mutex(self, _name: str) -> tuple[int, bool]:
        handle = self._new_handle()
        exists = self.mutex_held
        if not exists:
            self.mutex_held = True
            self.mutex_handle = handle
        return handle, exists

    def close_handle(self, handle: int) -> None:
        if handle == self.mutex_handle:
            self.mutex_held = False
            self.mutex_handle = 0
        name = self.event_handles.pop(handle, None)
        if name:
            self.exposed_events.discard(name)
            self.signaled_events.discard(name)

    def create_quit_event(self, name: str) -> int:
        handle = self._new_handle()
        self.event_handles[handle] = name
        self.exposed_events.add(name)
        return handle

    def signal_existing_event(self, name: str) -> bool:
        if name not in self.exposed_events:
            return False
        self.signaled_events.add(name)
        return True

    def named_mutex_exists(self, _name: str) -> bool:
        return self.mutex_held

    def event_is_signaled(self, handle: int) -> bool:
        name = self.event_handles.get(handle)
        if not name or name not in self.signaled_events:
            return False
        self.signaled_events.discard(name)  # auto-reset event semantics
        return True

    def install_keyboard_hook(self, callback: object) -> int:
        self.hook_callback = callback
        return 701

    def uninstall_keyboard_hook(self, handle: int) -> None:
        self.unhooked.append(handle)

    def call_next_hook(self, *_args: object) -> int:
        return 0

    def get_foreground_window(self) -> int:
        return self.foreground

    def get_root_window(self, hwnd: int) -> int:
        return self.root if hwnd == self.foreground else 0

    def get_focus_window(self, foreground: int) -> int:
        return self.focus if foreground == self.foreground else 0

    def wait_for_modifiers_released(self, _timeout: float) -> bool:
        self.wait_calls += 1
        return self.modifiers_released

    def send_unicode_text(self, text: str) -> bool:
        self.sent_text.append(text)
        return True

    def send_menu_mask(self) -> bool:
        self.menu_masks += 1
        return True

    def make_window_nonactivating(self, _hwnd: int, show: bool = True) -> None:
        self.nonactivation_calls.append((_hwnd, show))


class FakeVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class FakeOverlay:
    def __init__(self) -> None:
        self.deiconify_count = 0
        self.withdraw_count = 0

    def configure(self, **_kwargs: object) -> None:
        return None

    def deiconify(self) -> None:
        self.deiconify_count += 1

    def withdraw(self) -> None:
        self.withdraw_count += 1

    def winfo_id(self) -> int:
        return 900


class FakeScheduledRoot:
    def __init__(self) -> None:
        self.scheduled: list[tuple[int, object]] = []

    def after(self, delay_ms: int, callback: object) -> None:
        self.scheduled.append((delay_ms, callback))


class WindowsHostTests(unittest.TestCase):
    def make_host(self):
        self.api = FakeWindowsApi()
        self.calls: list[str] = []
        return WindowsHost(
            on_toggle=lambda: self.calls.append("toggle"),
            on_cancel=lambda: self.calls.append("cancel"),
            on_quit=lambda: self.calls.append("quit"),
            api=self.api,
        )

    def test_win_h_consumes_h_uses_menu_mask_and_leaves_windows_up_for_system(self):
        host = self.make_host()

        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN)))
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN)))
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYUP)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN)))
        host._dispatch_pending_events()

        self.assertEqual(["toggle"], self.calls)
        self.assertEqual(1, self.api.menu_masks)

    def test_injected_and_released_shortcut_events_are_not_taken_over(self):
        host = self.make_host()

        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN, injected=True)))
        host.set_shortcut_released(True)
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYUP)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP)))
        host._dispatch_pending_events()

        self.assertEqual([], self.calls)

    def test_escape_is_only_taken_while_controller_marks_host_busy(self):
        host = self.make_host()
        host.show("正在聆听", recording=True)

        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_ESCAPE, _WM_KEYDOWN)))
        host.set_busy(True)
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_ESCAPE, _WM_KEYDOWN)))
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_ESCAPE, _WM_KEYUP)))
        host._dispatch_pending_events()
        host.set_busy(False)

        self.assertEqual(["cancel"], self.calls)
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_ESCAPE, _WM_KEYDOWN)))

    def test_target_guard_rejects_changed_focus_and_waits_for_shortcut_modifiers(self):
        host = self.make_host()
        target = host.capture_target()

        self.assertTrue(host.insert_text("中文😀", target))
        self.assertEqual(["中文😀"], self.api.sent_text)
        self.assertEqual([0x4E2D, 0x6587, 0xD83D, 0xDE00], _utf16_units("中文😀"))
        self.assertEqual(1, self.api.wait_calls)

        self.api.focus = 202
        self.assertFalse(host.insert_text("不应输入", target))
        self.assertEqual(["中文😀"], self.api.sent_text)

        self.api.focus = 201
        self.api.modifiers_released = False
        self.assertFalse(host.insert_text("仍不应输入", target))
        self.assertEqual(["中文😀"], self.api.sent_text)

    def test_single_instance_quit_event_and_memory_text_are_independent_of_clipboard(self):
        host = self.make_host()
        self.assertTrue(host.acquire_single_instance())
        self.assertTrue(is_running(self.api))

        other = WindowsHost(lambda: None, lambda: None, lambda: None, api=self.api)
        self.assertFalse(other.acquire_single_instance())
        self.assertTrue(request_existing_quit(self.api))
        host._poll()
        self.assertEqual(["quit"], self.calls)

        host.set_last_text("累计识别全文")
        host.show("输入位置已改变", "错误说明", error=True)
        self.assertEqual("累计识别全文", host.latest_text)
        self.assertFalse(host.copy_text())  # no UI means no implicit clipboard mutation

        host.close()
        self.assertFalse(is_running(self.api))
        self.assertFalse(request_existing_quit(self.api))

    def test_overlay_hides_normal_notices_but_keeps_busy_and_errors_visible(self):
        host = self.make_host()
        overlay = FakeOverlay()
        host._overlay = overlay
        host._status_var = FakeVar()
        host._detail_var = FakeVar()

        host.show("已输入", "普通通知")
        host._render_overlay()
        host._render_overlay()
        self.assertEqual(1, overlay.deiconify_count)

        host._visible_until = time.monotonic() - 0.01
        host._render_overlay()
        host._render_overlay()
        self.assertEqual(1, overlay.withdraw_count)

        host.set_busy(True)
        host._render_overlay()
        host._render_overlay()
        self.assertEqual(2, overlay.deiconify_count)
        host.set_busy(False)
        host.show("识别失败", "错误详情", error=True)
        self.assertEqual("", host.latest_text)
        host._render_overlay()
        host._render_overlay()
        self.assertEqual(2, overlay.deiconify_count)

    def test_overlay_reapplies_nonactivation_after_tk_finishes_wrapping_window(self):
        host = self.make_host()
        root = FakeScheduledRoot()
        host._root = root
        host._overlay = FakeOverlay()
        host._overlay_visible = True

        host._schedule_overlay_style_reapply()

        self.assertEqual([0, 75, 250], [delay for delay, _callback in root.scheduled])
        for _delay, callback in root.scheduled:
            callback()
        self.assertEqual([(900, False), (900, False), (900, False)], self.api.nonactivation_calls)


if __name__ == "__main__":
    unittest.main()
