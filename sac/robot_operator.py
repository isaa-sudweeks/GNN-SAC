"""Terminal pause/resume and latched operator stop; no motor I/O in signal handlers."""

from __future__ import annotations

import os
import select
import sys


class OperatorEmergencyStop(RuntimeError):
    pass


class OperatorControls:
    """Ctrl+P toggles pause; E or the preview stop button latches emergency stop.

    Ctrl+C keeps the terminal's normal SIGINT behavior. Commands remain zero
    throughout pause; returning to running requires a newly built observation.
    """

    def __init__(self, *, enabled: bool = True, stream=None) -> None:
        self.paused = False
        self.stopped = False
        self.revision = 0
        self.enabled = enabled
        self.stream = stream or sys.stdin
        self.fd = None
        self.saved = None
        self.events = []
        self.external_events = None
        self.windows_console = False

    def __enter__(self) -> OperatorControls:
        if self.enabled and os.name == "nt":
            self.windows_console = True
        elif self.enabled and self.stream.isatty():
            import termios
            self.fd = self.stream.fileno()
            self.saved = termios.tcgetattr(self.fd)
            changed = termios.tcgetattr(self.fd)
            changed[3] &= ~(termios.ICANON | termios.ECHO)
            changed[6][termios.VMIN] = 0
            changed[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, changed)
        return self

    def __exit__(self, *_) -> None:
        if self.saved is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)

    def request(self, event: str) -> None:
        if event == "stop":
            self.stopped = True
            self.paused = True
        elif event == "toggle_pause" and not self.stopped:
            self.paused = not self.paused
        else:
            return
        self.revision += 1
        self.events.append("stopped" if self.stopped else "paused" if self.paused else "running")

    def feed(self, value: bytes) -> None:
        for character in value:
            if character == 16:
                self.request("toggle_pause")
            elif character in (ord("e"), ord("E")):
                self.request("stop")
            elif character == 3:
                raise KeyboardInterrupt

    def poll(self) -> bool:
        if self.windows_console:
            import msvcrt
            while msvcrt.kbhit():
                self.feed(msvcrt.getwch().encode("utf-8"))
        if self.fd is not None and select.select([self.fd], [], [], 0)[0]:
            self.feed(os.read(self.fd, 128))
        if self.external_events is not None:
            from queue import Empty
            while True:
                try:
                    self.request(self.external_events.get_nowait())
                except Empty:
                    break
        if self.stopped:
            raise OperatorEmergencyStop("operator emergency stop")
        return self.paused

    def drain_events(self) -> list[str]:
        events, self.events = self.events, []
        return events
