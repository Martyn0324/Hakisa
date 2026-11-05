import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hakisa.config import load_game_config
from hakisa.thalamus.capture import CallableSource, resolve_capture_region, sub_regions
from hakisa.thalamus.control import INPUT, Controller, LoggingBackend, make_key_input
from hakisa.thalamus.keys import normalize_pynput_key, scan_code
from hakisa.thalamus.tracker import InputTracker
from hakisa.types import DesiredInput, InputEvent, Region, Tap


def test_controller_diff_and_taps():
    be = LoggingBackend()
    c = Controller(be)
    c.apply(DesiredInput(keys=frozenset({"w", "shift"})))
    assert ("key_down", "shift") in be.calls and ("key_down", "w") in be.calls
    # modifiers are pressed first
    assert be.calls.index(("key_down", "shift")) < be.calls.index(("key_down", "w"))
    be.calls.clear()
    c.apply(DesiredInput(keys=frozenset({"w"}), taps=[Tap(keys=("x",))]))
    assert be.calls == [("key_up", "shift"), ("key_down", "x")]
    be.calls.clear()
    c.apply(DesiredInput(keys=frozenset({"w"})))  # pending tap released on the next tick
    assert be.calls == [("key_up", "x")]
    be.calls.clear()
    c.apply(DesiredInput(keys=frozenset({"w"})))  # nothing changes -> nothing sent
    assert be.calls == []


def test_controller_cursor_and_button_tap():
    be = LoggingBackend()
    c = Controller(be)
    c.apply(DesiredInput(buttons=frozenset({"left"}), cursor=(10, 20), taps=[Tap(button="right")]))
    assert be.calls[0] == ("move_to", (10, 20))
    assert ("button_down", "left") in be.calls and ("button_down", "right") in be.calls
    be.calls.clear()
    c.apply(DesiredInput(buttons=frozenset({"left"}), cursor=(10, 20)))
    assert be.calls == [("button_up", "right")]


def test_controller_tap_with_cursor_and_modifier():
    be = LoggingBackend()
    c = Controller(be)
    c.apply(DesiredInput(taps=[Tap(keys=("ctrl", "q"), cursor=(5, 6))]))
    assert be.calls == [("move_to", (5, 6)), ("key_down", "ctrl"), ("key_down", "q")]
    be.calls.clear()
    c.apply(DesiredInput())
    assert be.calls == [("key_up", "q"), ("key_up", "ctrl")]


def test_release_all_survives_backend_errors():
    be = LoggingBackend()
    c = Controller(be)
    c.apply(DesiredInput(keys=frozenset({"w", "z"}), buttons=frozenset({"left"})))
    be.fail_on = "key_up"  # first key_up raises
    errors = c.release_all()
    assert len(errors) == 1
    assert ("button_up", "left") in be.calls
    assert not c.held_keys and not c.held_buttons


def test_sendinput_struct_layout_matches_win64():
    assert ctypes.sizeof(INPUT) == 40
    inp = make_key_input("up", up=True)
    assert inp.type == 1
    assert inp.u.ki.wScan == 0x48
    assert inp.u.ki.dwFlags & 0x0001 and inp.u.ki.dwFlags & 0x0002 and inp.u.ki.dwFlags & 0x0008


def test_scan_codes():
    assert scan_code("w") == (0x11, False)
    assert scan_code("Key.shift_r") == (0x2A, False)  # merged into "shift"
    assert scan_code("left") == (0x4B, True)
    with pytest.raises(KeyError):
        scan_code("nope")


def test_pynput_key_normalisation():
    assert normalize_pynput_key(SimpleNamespace(vk=0x51, char="\x11")) == "q"  # Ctrl+Q
    assert normalize_pynput_key(SimpleNamespace(vk=None, char="W")) == "w"
    assert normalize_pynput_key(SimpleNamespace(name="shift_r")) == "shift"
    assert normalize_pynput_key(SimpleNamespace(name="ctrl_l")) == "ctrl"
    assert normalize_pynput_key(SimpleNamespace(name="f12")) == "f12"
    assert normalize_pynput_key(SimpleNamespace(vk=0x31, char="!")) == "1"


def test_tracker_state_events_and_hotkeys():
    t = [0.0]
    tr = InputTracker(clock=lambda: t[0], move_hz=10)
    fired = []
    tr.on_hotkey("f12", lambda: fired.append(True))
    tr.feed(InputEvent(0.0, "key", "w", True))
    tr.feed(InputEvent(0.01, "key", "w", True))  # auto-repeat ignored
    tr.feed(InputEvent(0.02, "button", "left", True, 5, 5))
    tr.feed(InputEvent(0.03, "move", "", True, 6, 6))
    tr.feed(InputEvent(0.04, "move", "", True, 7, 7))  # coalesced (10 Hz), but cursor updated
    tr.feed(InputEvent(0.05, "key", "f12", True))
    snap = tr.snapshot()
    assert snap.keys == frozenset({"w", "f12"}) and snap.buttons == frozenset({"left"})
    assert snap.cursor == (7, 7)
    events = tr.drain_events()
    assert [e.kind for e in events] == ["key", "button", "move", "key"]
    assert fired == [True]
    assert tr.drain_events() == []


def test_capture_regions():
    cfg = load_game_config("lol")
    cap = resolve_capture_region(cfg, window_lookup=lambda title: Region(100, 50, 1920, 1080))
    assert cap == Region(100, 50, 1920, 1080)
    obs, mm = sub_regions(cfg, cap)
    assert obs == cap
    assert mm == Region(100 + 1645, 50 + 805, 270, 270)
    dojo = load_game_config("dojo_keyboard")
    obs, mm = sub_regions(dojo, Region(0, 0, 544, 448))
    assert obs == Region(0, 0, 384, 448) and mm is None


def test_callable_source():
    frames = iter([np.zeros((4, 5, 3), np.uint8), np.ones((4, 5, 3), np.uint8)])
    src = CallableSource(lambda: next(frames))
    assert src.region == Region(0, 0, 5, 4)
    assert src.grab().sum() == 0 and src.grab().sum() == 60
