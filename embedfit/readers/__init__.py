"""Instrument readers.  A reader returns a :class:`~embedfit.data.Spectrum` and
records what the instrument did (detector correction, reference conversion,
dark subtraction) so that it can be put into the model.

Register a new reader with :func:`register`; call it with :func:`read`.
"""
from __future__ import annotations

from typing import Callable

from .generic import from_arrays, read_table
from .riken_ac import read_ac, read_ac_folder

_READERS: dict = {}


def register(name: str) -> Callable:
    def deco(fn):
        _READERS[name.lower()] = fn
        return fn
    return deco


def read(kind: str, path: str, **kw):
    try:
        return _READERS[kind.lower()](path, **kw)
    except KeyError:
        raise KeyError(f"no reader '{kind}'; available: {sorted(_READERS)}") from None


def available() -> list:
    return sorted(_READERS)


register("riken_ac")(read_ac)
register("table")(read_table)

__all__ = ["read", "register", "available", "read_ac", "read_ac_folder", "read_table", "from_arrays"]
