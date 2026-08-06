"""Tests for the optional `local` extra guards in daft_graph._optional."""

from __future__ import annotations

import builtins

import pytest

from daft_graph._optional import has_local_extra, require_numpy, require_scipy


def _block(monkeypatch: pytest.MonkeyPatch, blocked: str) -> None:
    """Make importing `blocked` (and its submodules) raise ModuleNotFoundError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked or name.startswith(blocked + "."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_require_numpy_passes_when_present() -> None:
    # numpy is in the dev group, so this must not raise
    require_numpy("a feature")


def test_require_scipy_passes_when_present() -> None:
    require_scipy("a feature")


def test_require_numpy_raises_guided_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _block(monkeypatch, "numpy")
    with pytest.raises(ImportError, match=r"daft-graph\[local\]") as exc:
        require_numpy("the local connected components solve")
    assert "numpy" in str(exc.value)
    assert "the local connected components solve" in str(exc.value)


def test_require_scipy_raises_guided_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _block(monkeypatch, "scipy")
    with pytest.raises(ImportError, match=r"daft-graph\[local\]") as exc:
        require_scipy("svd_plus_plus")
    assert "scipy" in str(exc.value)


def test_require_numpy_catches_plain_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken (not merely absent) install raises ImportError, not just ModuleNotFoundError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy":
            raise ImportError("DLL load failed")  # simulate a broken build
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"daft-graph\[local\]"):
        require_numpy("x")


def test_has_local_extra_true_when_present() -> None:
    assert has_local_extra() is True


def test_has_local_extra_false_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # a find_spec that blows up must be treated as "not available", not propagated
    import daft_graph._optional as opt

    def boom(_name: str) -> object:
        raise RuntimeError("broken import machinery")

    monkeypatch.setattr(opt, "find_spec", boom, raising=False)
    # find_spec is imported inside the function, so patch importlib.util too
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert has_local_extra() is False
