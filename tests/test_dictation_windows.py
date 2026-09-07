from __future__ import annotations

import time
import threading
import unittest

from zh_asr.dictation_windows import (
    KeyboardEvent,
    TargetWindow,
    WindowsHost,
    _DisplayMonitor,
    _HOST_INPUT_EXTRA_INFO,
    _OverlayPanel,
    _VK_ESCAPE,
    _VK_H,
    _VK_LCONTROL,
    _VK_LMENU,
    _VK_LSHIFT,
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
        self.move_calls: list[tuple[int, int, int]] = []
        self.own_process_windows: set[int] = set()
        self.message_queue_ready = threading.Event()
        self.message_quit = threading.Event()
        self.posted_thread_quits: list[int] = []
        self.monitors: list[_DisplayMonitor] = []

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

    def current_thread_id(self) -> int:
        return threading.get_ident()

    def ensure_message_queue(self) -> None:
        self.message_queue_ready.set()

    def pump_messages(self) -> None:
        self.message_quit.wait(1)

    def post_thread_quit(self, thread_id: int) -> bool:
        self.posted_thread_quits.append(thread_id)
        self.message_quit.set()
        return True

    def get_foreground_window(self) -> int:
        return self.foreground

    def get_root_window(self, hwnd: int) -> int:
        if hwnd == self.foreground:
            return self.root
        return hwnd if hwnd in (self.root, 2001) else 0

    def get_window_process_id(self, hwnd: int) -> int:
        return 42 if hwnd in self.own_process_windows else 99

    def current_process_id(self) -> int:
        return 42

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

    def list_monitors(self) -> list[_DisplayMonitor]:
        return list(self.monitors)

    def move_window(self, hwnd: int, x: int, y: int) -> None:
        self.move_calls.append((hwnd, x, y))


class FakeVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class FakeOverlay:
    def __init__(self, window_id: int = 900) -> None:
        self.window_id = window_id
        self.deiconify_count = 0
        self.withdraw_count = 0
        self.destroy_count = 0

    def configure(self, **_kwargs: object) -> None:
        return None

    def deiconify(self) -> None:
        self.deiconify_count += 1

    def withdraw(self) -> None:
        self.withdraw_count += 1

    def destroy(self) -> None:
        self.destroy_count += 1

    def winfo_id(self) -> int:
        return self.window_id

    def update_idletasks(self) -> None:
        pass

    def winfo_screenwidth(self) -> int:
        return 1920

    def winfo_screenheight(self) -> int:
        return 1080


class FakeScheduledRoot:
    def __init__(self) -> None:
        self.scheduled: list[tuple[int, object]] = []

    def after(self, delay_ms: int, callback: object) -> None:
        self.scheduled.append((delay_ms, callback))

    def update_idletasks(self) -> None:
        pass


class FakeTray:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()

    def run(self) -> None:
        self.started.set()
        self.stopped.wait(1)

    def stop(self) -> None:
        self.stopped.set()


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

    def test_low_level_hook_uses_a_separate_message_pump_and_stops_cleanly(self):
        host = self.make_host()
        host._install_hook()
        try:
            self.assertTrue(self.api.message_queue_ready.wait(0.2))
            self.assertEqual(701, host._hook_handle)
            self.assertIsNotNone(host._hook_thread)
        finally:
            host._stop_hook_worker()

        self.assertEqual([701], self.api.unhooked)
        self.assertEqual(0, host._hook_handle)
        self.assertEqual(1, len(self.api.posted_thread_quits))

    def test_tray_uses_our_daemon_thread_and_can_be_stopped_boundedly(self):
        tray = FakeTray()
        host = WindowsHost(
            on_toggle=lambda: None,
            on_cancel=lambda: None,
            on_quit=lambda: None,
            api=FakeWindowsApi(),
            tray_factory=lambda _host: tray,
        )
        host._start_tray()
        try:
            self.assertTrue(tray.started.wait(0.2))
            self.assertTrue(host._tray_thread.daemon)
        finally:
            tray.stop()
            host._tray_thread.join(timeout=0.2)

    def test_only_our_exact_marked_injected_events_are_ignored(self):
        host = self.make_host()

        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN, injected=True)))
        self.assertFalse(
            host._handle_keyboard_event(
                KeyboardEvent(_VK_H, _WM_KEYDOWN, injected=True, extra_info=_HOST_INPUT_EXTRA_INFO)
            )
        )
        self.assertFalse(
            host._handle_keyboard_event(
                KeyboardEvent(_VK_H, _WM_KEYUP, injected=True, extra_info=_HOST_INPUT_EXTRA_INFO)
            )
        )
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP, injected=True)))
        host._dispatch_pending_events()
        self.assertEqual([], self.calls)

    def test_externally_injected_win_h_is_taken_over_like_a_physical_hotkey(self):
        host = self.make_host()

        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN, injected=True)))
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN, injected=True)))
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYUP, injected=True)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP, injected=True)))
        host._dispatch_pending_events()

        self.assertEqual(["toggle"], self.calls)
        self.assertEqual(1, self.api.menu_masks)

    def test_ctrl_win_h_is_supported_for_physical_and_externally_injected_input(self):
        for injected in (False, True):
            with self.subTest(injected=injected):
                host = self.make_host()
                self.assertFalse(
                    host._handle_keyboard_event(KeyboardEvent(_VK_LCONTROL, _WM_KEYDOWN, injected=injected))
                )
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN, injected=injected)))
                self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN, injected=injected)))
                self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYUP, injected=injected)))
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP, injected=injected)))
                self.assertFalse(
                    host._handle_keyboard_event(KeyboardEvent(_VK_LCONTROL, _WM_KEYUP, injected=injected))
                )
                host._dispatch_pending_events()

                self.assertEqual(["toggle"], self.calls)
                self.assertEqual(1, self.api.menu_masks)

    def test_win_alt_h_and_win_shift_h_are_left_entirely_to_windows(self):
        for modifier in (_VK_LMENU, _VK_LSHIFT):
            with self.subTest(modifier=modifier):
                host = self.make_host()
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(modifier, _WM_KEYDOWN)))
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN)))
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN)))
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYUP)))
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP)))
                self.assertFalse(host._handle_keyboard_event(KeyboardEvent(modifier, _WM_KEYUP)))
                host._dispatch_pending_events()

                self.assertEqual([], self.calls)
                self.assertEqual(0, self.api.menu_masks)

    def test_released_shortcut_events_are_not_taken_over(self):
        host = self.make_host()

        host.set_shortcut_released(True)
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYUP)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP)))
        host._dispatch_pending_events()

        self.assertEqual([], self.calls)

    def test_hotkey_and_record_button_have_separate_actions(self):
        calls = []
        host = WindowsHost(on_toggle=lambda: calls.append("record"), on_cancel=lambda: None,
                           on_quit=lambda: None, on_hotkey=lambda: calls.append("visibility"),
                           api=FakeWindowsApi())
        host._events.put("toggle")
        host._dispatch_pending_events()
        self.assertEqual(["visibility"], calls)
        self.assertFalse(host.panel_visible)
        host._toggle_from_panel()
        self.assertEqual(["visibility", "record"], calls)

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

    def test_show_does_not_reopen_a_panel_after_the_user_hides_it(self):
        host = self.make_host()
        overlay = FakeOverlay()
        host._overlay = overlay
        host._status_var = FakeVar()
        host._detail_var = FakeVar()

        host.show("已输入", "普通通知")
        host._render_overlay()
        self.assertEqual(0, overlay.deiconify_count)

        host.open_panel()
        host._dispatch_pending_events()
        self.assertEqual(1, overlay.deiconify_count)

        host.hide_panel()
        host._dispatch_pending_events()
        self.assertEqual(1, overlay.withdraw_count)
        host.show("识别失败", "错误详情", error=True)
        host._render_overlay()
        self.assertEqual(1, overlay.deiconify_count)

    def test_configured_pnp_panels_follow_topology_without_using_tur_or_reopening_hidden_ui(self):
        api = FakeWindowsApi()
        calls: list[str] = []
        physical = _DisplayMonitor("MONITOR\\PHLC34B\\001", "Philips", -1920, -200, 1920, 1080)
        vdd = _DisplayMonitor("MONITOR\\MTT1337\\002", "MTT1337", 0, 0, 2880, 1740, primary=True)
        tur = _DisplayMonitor("MONITOR\\TUR0000\\003", "TUR", 2880, 0, 2288, 1048)
        api.monitors = [physical, vdd, tur]
        host = WindowsHost(
            on_toggle=lambda: calls.append("toggle"),
            on_cancel=lambda: None,
            on_quit=lambda: None,
            monitor_ids=["PHLC34B", "MTT1337"],
            api=api,
        )
        host._root = FakeScheduledRoot()
        host._topology_initialized = True
        created: list[_OverlayPanel] = []

        def create_panel(monitor: _DisplayMonitor | None) -> _OverlayPanel:
            panel = _OverlayPanel(monitor=monitor, overlay=FakeOverlay(900 + len(created)))
            created.append(panel)
            return panel

        host._create_overlay_panel = create_panel  # type: ignore[method-assign]
        host._sync_display_topology_ui(force=True)

        self.assertEqual([physical, vdd], [panel.monitor for panel in host._panels])
        host._panel_open = True
        host._render_overlay()
        self.assertEqual([1, 1], [panel.overlay.deiconify_count for panel in host._panels])
        self.assertEqual([(900, True), (901, True)], api.nonactivation_calls)

        host._hide_panel_ui()
        api.monitors = [vdd, tur]
        host._sync_display_topology_ui(force=True)
        host.show("后台状态更新", "不会重新显示")
        host._render_overlay()
        self.assertEqual([vdd], [panel.monitor for panel in host._panels])
        self.assertEqual(0, host._panels[0].overlay.deiconify_count)

        api.monitors = [tur]
        host._sync_display_topology_ui(force=True)
        host._render_overlay()
        self.assertEqual([], host._panels)
        self.assertIsNone(host._overlay)

        api.monitors = [physical, vdd, tur]
        host._sync_display_topology_ui(force=True)
        host.show("后台状态更新", "仍保持隐藏")
        host._render_overlay()
        self.assertEqual([physical, vdd], [panel.monitor for panel in host._panels])
        self.assertEqual([0, 0], [panel.overlay.deiconify_count for panel in host._panels])
        self.assertEqual([], calls)

    def test_monitor_geometry_preserves_negative_coordinates_and_small_work_areas(self):
        host = self.make_host()
        monitor = _DisplayMonitor("MONITOR\\PHLC34B\\001", "Philips", -1920, -200, 120, 50)
        panel = _OverlayPanel(monitor=monitor, overlay=FakeOverlay())

        self.assertEqual((-1920, -200), host._panel_position(panel))
        host._panels = [panel]
        host._panel_open = True
        host._render_overlay()
        host._reapply_overlay_nonactivation(panel, finish_position=True)
        self.assertEqual([(900, -1920, -200)], self.api.move_calls)
        host._hide_panel_ui()
        host._panel_open = True
        host._render_overlay()
        self.assertEqual([(900, -1920, -200)], self.api.move_calls)

    def test_hotkey_dispatches_one_controller_toggle_for_multiple_panels(self):
        host = self.make_host()
        host._panels = [
            _OverlayPanel(monitor=None, overlay=FakeOverlay(900)),
            _OverlayPanel(monitor=None, overlay=FakeOverlay(901)),
        ]

        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYDOWN)))
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYDOWN)))
        self.assertTrue(host._handle_keyboard_event(KeyboardEvent(_VK_H, _WM_KEYUP)))
        self.assertFalse(host._handle_keyboard_event(KeyboardEvent(_VK_LWIN, _WM_KEYUP)))
        host._dispatch_pending_events()

        self.assertEqual(["toggle"], self.calls)

    def test_panel_callbacks_devices_and_external_target_fallback(self):
        calls: list[object] = []
        host = WindowsHost(
            on_toggle=lambda: calls.append("toggle"),
            on_cancel=lambda: None,
            on_quit=lambda: None,
            on_hide=lambda: calls.append("hide"),
            on_device_change=lambda value: calls.append(("device", value)),
            on_refresh_devices=lambda: calls.append("refresh"),
            api=FakeWindowsApi(),
        )
        host.set_microphones(
            [{"value": "dji", "label": "DJI Mic Mini"}, {"value": None, "label": "Windows 默认麦克风"}],
            selected="dji",
        )
        self.assertEqual([], calls)
        self.assertEqual("dji", host._selected_microphone)
        host._select_microphone(None)
        host._refresh_devices_from_panel()
        self.assertEqual([("device", None), "refresh"], calls)

        host._own_window_roots = {1001}
        host._last_external_target = TargetWindow(2001, 2002)
        host._api.foreground = 101
        host._api.root = 1001
        self.assertEqual(TargetWindow(2001, 2002), host.capture_target())
        self.assertFalse(host.insert_text("不应输入", TargetWindow(1001, 201)))

        host._own_window_roots.clear()
        host._api.own_process_windows.add(101)
        self.assertEqual(TargetWindow(2001, 2002), host.capture_target())

    def test_post_to_ui_and_x_hide_before_notifying_controller(self):
        state: list[object] = []
        host = WindowsHost(
            on_toggle=lambda: None,
            on_cancel=lambda: None,
            on_quit=lambda: None,
            on_hide=lambda: state.append(host._panel_open),
            api=FakeWindowsApi(),
        )
        overlay = FakeOverlay()
        host._overlay = overlay
        host._overlay_visible = True
        host._panel_open = True
        host.post_to_ui(lambda: state.append("ui"))
        host._drain_ui_calls()
        host._hide_from_panel()

        self.assertEqual(["ui", False], state)
        self.assertEqual(1, overlay.withdraw_count)

    def test_close_arriving_during_render_does_not_strand_tk_mainloop(self):
        host = self.make_host()
        root = FakeScheduledRoot()
        host._root = root
        host._running = True
        host._render_overlay = host.close
        host._poll()
        self.assertFalse(host._finalized)
        self.assertEqual(1, len(root.scheduled))
        _delay, next_turn = root.scheduled.pop()
        next_turn()
        self.assertTrue(host._finalized)
        self.assertFalse(host._running)

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
