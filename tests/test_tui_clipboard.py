"""Clipboard backend selection and the copy path.

`detect_backend` takes the platform and a `which` so the whole matrix — Windows,
macOS, Wayland, X11, WSL, and a box with none of them — is testable from one
machine. That injectability is the only reason these cases can be covered at all.
"""

from __future__ import annotations

from tui.clipboard import copy, copy_via_backend, detect_backend


def _which(*available: str):
    """A stub `shutil.which` that only knows about `available`."""
    present = set(available)
    return lambda binary: f"/usr/bin/{binary}" if binary in present else None


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def test_windows_uses_clip_exe():
    assert detect_backend("win32", _which("clip.exe")) == ["clip.exe"]


def test_macos_uses_pbcopy():
    assert detect_backend("darwin", _which("pbcopy")) == ["pbcopy"]


def test_linux_prefers_wayland_over_x11():
    """On a Wayland session xclip often exists but talks to an XWayland
    clipboard nothing else reads."""
    backend = detect_backend("linux", _which("wl-copy", "xclip"))
    assert backend == ["wl-copy"]


def test_linux_falls_back_to_xclip():
    assert detect_backend("linux", _which("xclip")) == [
        "xclip",
        "-selection",
        "clipboard",
    ]


def test_linux_falls_back_to_xsel():
    assert detect_backend("linux", _which("xsel"))[0] == "xsel"


def test_wsl_reaches_the_windows_clipboard():
    """WSL reports platform 'linux' but has no X server; clip.exe is the only
    thing that reaches the clipboard the user can actually paste from."""
    assert detect_backend("linux", _which("clip.exe")) == ["clip.exe"]


def test_no_helper_means_no_backend():
    assert detect_backend("linux", _which()) is None


def test_a_missing_helper_on_windows_means_no_backend():
    assert detect_backend("win32", _which()) is None


def test_an_unknown_platform_tries_the_linux_candidates():
    assert detect_backend("freebsd", _which("xclip")) is not None


# --------------------------------------------------------------------------
# Piping to the backend
# --------------------------------------------------------------------------


def test_a_missing_binary_is_reported_not_raised():
    """A clipboard helper failing must never take the session with it."""
    assert copy_via_backend("x", ["definitely-not-a-real-binary-zzz"]) is False


# --------------------------------------------------------------------------
# The copy path
# --------------------------------------------------------------------------


def _sender(*, works: bool = True):
    """A stub OSC-52 sender. Passed in rather than pulled off an app object so
    `copy` can sit underneath `App.copy_to_clipboard`."""
    sent: list[str] = []

    def send(text: str) -> None:
        if not works:
            raise RuntimeError("no OSC 52 here")
        sent.append(text)

    send.sent = sent
    return send


def test_osc52_is_always_attempted(monkeypatch):
    """It is free and it is the only path that works over SSH."""
    monkeypatch.setattr("tui.clipboard.detect_backend", lambda *_a, **_k: None)
    send = _sender()
    copy(send, "hello")
    assert send.sent == ["hello"]


def test_the_message_names_the_backend_that_worked(monkeypatch):
    monkeypatch.setattr("tui.clipboard.detect_backend", lambda *_a, **_k: ["pbcopy"])
    monkeypatch.setattr("tui.clipboard.copy_via_backend", lambda *_a: True)
    assert "pbcopy" in copy(_sender(), "hello")


def test_it_falls_back_to_the_terminal_when_the_helper_fails(monkeypatch):
    monkeypatch.setattr("tui.clipboard.detect_backend", lambda *_a, **_k: ["pbcopy"])
    monkeypatch.setattr("tui.clipboard.copy_via_backend", lambda *_a: False)
    assert "terminal" in copy(_sender(), "hello")


def test_the_native_helper_runs_even_when_osc52_works(monkeypatch):
    """Windows Terminal accepts OSC 52 but a plain conhost does not, and there
    is no way to tell which one you are in — so do both."""
    monkeypatch.setattr("tui.clipboard.detect_backend", lambda *_a, **_k: ["clip.exe"])
    calls: list = []
    monkeypatch.setattr(
        "tui.clipboard.copy_via_backend", lambda *a: calls.append(a) or True
    )
    send = _sender()
    copy(send, "hello")
    assert send.sent == ["hello"] and len(calls) == 1


def test_a_total_failure_says_so(monkeypatch):
    """A copy you cannot tell has failed is worse than no copy button."""
    monkeypatch.setattr("tui.clipboard.detect_backend", lambda *_a, **_k: None)
    assert "failed" in copy(_sender(works=False), "hello")


def test_copying_nothing_says_so(monkeypatch):
    monkeypatch.setattr("tui.clipboard.detect_backend", lambda *_a, **_k: None)
    assert copy(_sender(), "") == "nothing to copy"


def test_the_message_counts_lines(monkeypatch):
    monkeypatch.setattr("tui.clipboard.detect_backend", lambda *_a, **_k: None)
    assert "3 lines" in copy(_sender(), "a\nb\nc")
