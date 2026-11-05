"""Thalamus: the relay between Hakisa and the machine (screen in, keys/mouse out, human input recording).

"Many afferences and efferences pass through the thalamus ... sometimes the information can be
preprocessed before being sent to the cortex" — Hakisa 1.0 notes (Meneses, Neuroanatomia Aplicada).
"""

from hakisa.thalamus.capture import CallableSource, make_dpi_aware, resolve_capture_region, sub_regions
from hakisa.thalamus.control import Controller, LoggingBackend
from hakisa.thalamus.tracker import InputTracker

__all__ = [
    "CallableSource",
    "Controller",
    "InputTracker",
    "LoggingBackend",
    "make_dpi_aware",
    "resolve_capture_region",
    "sub_regions",
]
