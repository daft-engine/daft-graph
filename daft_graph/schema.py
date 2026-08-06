"""Column name constants and shared type aliases for daft-graph."""

from __future__ import annotations

from typing import Literal

# Vertex and edge column conventions, mirroring GraphFrames (id, src, dst).
ID = "id"
SRC = "src"
DST = "dst"

# Algorithm output columns.
COMPONENT = "component"
LABEL = "label"
RANK = "rank"

# Internal working columns used by edge utilities and the star algorithms.
U = "u"
V = "v"
REP = "rep"

# Connected components execution strategy.
Strategy = Literal["auto", "distributed", "local"]

__all__ = [
    "COMPONENT",
    "DST",
    "ID",
    "LABEL",
    "RANK",
    "REP",
    "SRC",
    "Strategy",
    "U",
    "V",
]
