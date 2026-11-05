"""Key names, normalisation of OS/pynput key objects and Windows scan codes.

Games that read DirectInput / raw input ignore virtual-key events, so Hakisa injects *scan codes*
(PS/2 set 1).  Extended keys (arrows, right ctrl/alt, ...) need ``KEYEVENTF_EXTENDEDKEY``.
"""

from __future__ import annotations

SCAN_CODES: dict[str, int] = {
    "esc": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06, "6": 0x07, "7": 0x08,
    "8": 0x09, "9": 0x0A, "0": 0x0B, "-": 0x0C, "=": 0x0D, "backspace": 0x0E, "tab": 0x0F,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14, "y": 0x15, "u": 0x16, "i": 0x17,
    "o": 0x18, "p": 0x19, "[": 0x1A, "]": 0x1B, "enter": 0x1C, "ctrl": 0x1D, "a": 0x1E, "s": 0x1F,
    "d": 0x20, "f": 0x21, "g": 0x22, "h": 0x23, "j": 0x24, "k": 0x25, "l": 0x26, ";": 0x27,
    "'": 0x28, "`": 0x29, "shift": 0x2A, "\\": 0x2B, "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F,
    "b": 0x30, "n": 0x31, "m": 0x32, ",": 0x33, ".": 0x34, "/": 0x35, "shift_r": 0x36,
    "alt": 0x38, "space": 0x39, "caps_lock": 0x3A,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F, "f6": 0x40, "f7": 0x41,
    "f8": 0x42, "f9": 0x43, "f10": 0x44, "f11": 0x57, "f12": 0x58,
}

EXTENDED_SCAN_CODES: dict[str, int] = {
    "up": 0x48, "down": 0x50, "left": 0x4B, "right": 0x4D, "insert": 0x52, "delete": 0x53,
    "home": 0x47, "end": 0x4F, "page_up": 0x49, "page_down": 0x51, "ctrl_r": 0x1D, "alt_r": 0x38,
}

# pynput Key enum names -> Hakisa names (left/right variants of modifiers are merged)
_PYNPUT_ALIASES = {
    "shift": "shift", "shift_l": "shift", "shift_r": "shift",
    "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt",
    "cmd": "cmd", "cmd_l": "cmd", "cmd_r": "cmd",
    "esc": "esc", "enter": "enter", "space": "space", "tab": "tab", "backspace": "backspace",
    "caps_lock": "caps_lock", "up": "up", "down": "down", "left": "left", "right": "right",
    "insert": "insert", "delete": "delete", "home": "home", "end": "end",
    "page_up": "page_up", "page_down": "page_down",
}


def normalize_key_name(name: str) -> str:
    name = name.strip().lower()
    if name.startswith("key."):
        name = name[4:]
    return _PYNPUT_ALIASES.get(name, name)


def normalize_pynput_key(key) -> str | None:
    """Turn a pynput ``Key``/``KeyCode`` (duck-typed) into a Hakisa key name.

    Uses the virtual-key code for letters/digits because, with Ctrl held, ``KeyCode.char`` becomes a
    control character (Ctrl+Q -> ``'\\x11'``).
    """
    vk = getattr(key, "vk", None)
    if vk is not None:
        if 0x41 <= vk <= 0x5A:
            return chr(vk).lower()
        if 0x30 <= vk <= 0x39:
            return chr(vk)
        if 0x60 <= vk <= 0x69:
            return f"num{vk - 0x60}"
    char = getattr(key, "char", None)
    if char:
        if len(char) == 1 and ord(char) < 32:  # control character, e.g. Ctrl+letter on some platforms
            return chr(ord(char) + 96)
        return char.lower()
    name = getattr(key, "name", None)
    if name:
        name = normalize_key_name(name)
        if name.startswith("f") and name[1:].isdigit():
            return name
        return name
    return None


def scan_code(name: str) -> tuple[int, bool]:
    """Scan code and "extended" flag for a Hakisa key name."""
    name = normalize_key_name(name)
    if name in SCAN_CODES:
        return SCAN_CODES[name], False
    if name in EXTENDED_SCAN_CODES:
        return EXTENDED_SCAN_CODES[name], True
    raise KeyError(f"no scan code for key {name!r}")
