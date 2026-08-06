"""Guarded imports for the optional ``local`` extra.

The distributed algorithms are pure Daft, so ``numpy`` and ``scipy`` are not core
runtime dependencies. They back only the single node solves, namely the
``strategy="local"`` path of connected components and the ``svd_plus_plus``
factorization.

These helpers check availability and raise an actionable message. The caller then
does an ordinary ``import numpy as np`` inside the function body, which keeps the
import lazy while leaving the module properly typed for mypy.
"""

from __future__ import annotations

_EXTRA = "daft-graph[local]"

_HINT = (
    "{package} is required for {feature}, but it is not installed. It ships in the "
    "optional 'local' extra, so install {extra} (for example 'uv sync --extra local' "
    "or 'pip install {extra}'). The distributed code paths do not need it."
)


def require_numpy(feature: str) -> None:
    """Check that ``numpy`` is importable.

    Args:
        feature: Human readable name of the caller, used in the error message.

    Raises:
        ImportError: If ``numpy`` is not installed.
    """
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(_HINT.format(package="numpy", feature=feature, extra=_EXTRA)) from exc


def require_scipy(feature: str) -> None:
    """Check that ``scipy`` and the sparse graph routines are importable.

    Args:
        feature: Human readable name of the caller, used in the error message.

    Raises:
        ImportError: If ``scipy`` is not installed.
    """
    try:
        import scipy.sparse.csgraph  # noqa: F401
    except ImportError as exc:
        raise ImportError(_HINT.format(package="scipy", feature=feature, extra=_EXTRA)) from exc


def has_local_extra() -> bool:
    """Whether the optional ``local`` extra is installed.

    Lets ``strategy="auto"`` mean the best *available* strategy, so a core only
    install silently prefers the distributed path instead of raising. An explicit
    ``strategy="local"`` still raises, because the caller asked for it by name.

    Returns:
        True when both ``numpy`` and ``scipy`` can be imported.
    """
    from importlib.util import find_spec

    # A never-raise probe. `find_spec` on a dotted path imports the parent
    # packages, so a broken (not merely absent) scipy install can raise arbitrary
    # exceptions; treat any of them as "not available" so auto still falls back.
    try:
        return find_spec("numpy") is not None and find_spec("scipy.sparse.csgraph") is not None
    except Exception:  # noqa: BLE001 - availability probe must never raise
        return False
