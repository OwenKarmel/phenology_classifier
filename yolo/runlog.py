"""Log a pipeline command to yolo/logs/ so the viewer's Model tab can follow it.

The command runs again as a child process. This parent copies everything the
child prints to the terminal unchanged, and to yolo/logs/<stamp>_<command>_<experiment>.log
as plain text: colours removed, and progress-bar redraws (lines rewritten
with \\r) kept to one every few seconds. <same name>.json holds the status:

    {"command", "argv", "experiment", "pid", "started", "ended", "exit_code",
     "state": "running" | "finished" | "failed" | "stopped"}

The run ignores SIGHUP, so it keeps going if the terminal closes; Ctrl+C or
`kill <pid>` stops it.
"""
import codecs
import datetime as dt
import json
import os
import re
import select
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
CHILD_ENV = "YOLO_PIPELINE_LOG"  # set in the child: the log it is writing to
ANSI = re.compile(r"\x1b(\[[0-?]*[ -/]*[@-~]|\][^\x07]*\x07|[@-Z\\-_])")
BREAKS = re.compile(r"(\r\n|\r|\n)")


def clean(text):
    return ANSI.sub("", text)


class LogWriter:
    """Plain-text log of a terminal stream.

    A progress bar redraws its line with \\r many times a second. Only the
    newest frame is written, at most every `interval` seconds, as
    "\\r<frame>" (a reader shows the text after a line's last \\r).
    """

    def __init__(self, fh, interval=2.0):
        self.fh = fh
        self.interval = interval
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.cur = ""        # text since the last \r or \n
        self.frame = None    # newest finished frame of this line, not written yet
        self.shown = None    # frame last written for this line
        self.last = 0.0

    def feed(self, data):
        for part in BREAKS.split(self.decoder.decode(data)):
            if part in ("\n", "\r\n"):
                self.end_line()
            elif part == "\r":
                if self.cur:
                    self.frame, self.cur = self.cur, ""
            elif part:
                self.cur += part
        self.tick()

    def end_line(self):
        text = self.cur or self.frame or self.shown or ""
        self.fh.write(("\r" if self.shown is not None else "") + clean(text) + "\n")
        self.cur, self.frame, self.shown = "", None, None
        self.fh.flush()

    def tick(self):
        if self.frame is None or time.monotonic() - self.last < self.interval:
            return
        self.fh.write(("\r" if self.shown is not None else "") + clean(self.frame))
        self.shown, self.frame, self.last = self.frame, None, time.monotonic()
        self.fh.flush()

    def close(self):
        if self.cur or self.frame or self.shown is not None:
            self.end_line()


def write_status(path, status):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(status, fh, indent=1)
    os.replace(tmp, path)


def run_logged(command, experiment, argv, script):
    """Run `script argv` as a child, logging it; returns the child's exit code."""
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(LOG_DIR, f"{stamp}_{command}_{experiment}")
    log_path, status_path = base + ".log", base + ".json"
    status = {"command": command, "argv": argv, "experiment": experiment, "pid": os.getpid(),
              "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "ended": None,
              "exit_code": None, "state": "running", "log": os.path.basename(log_path)}
    write_status(status_path, status)

    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # survive a closed terminal (inherited by the child)
    env = dict(os.environ, PYTHONUNBUFFERED="1", **{CHILD_ENV: log_path})
    child = subprocess.Popen([sys.executable, script, *argv], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env)
    # Set after starting the child, which must keep Python's Ctrl+C handling:
    # Ctrl+C reaches it directly (same process group) and it stops cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: child.terminate())

    term = sys.stdout.buffer
    with open(log_path, "w", buffering=1) as fh:
        writer = LogWriter(fh)
        fh.write(f"$ {' '.join(['pipeline.py', *argv])}\n")
        fd = child.stdout.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                writer.tick()
                continue
            data = os.read(fd, 65536)
            if not data:
                break
            if term is not None:
                try:
                    term.write(data)
                    term.flush()
                except OSError:  # terminal gone: keep logging
                    term = None
            writer.feed(data)
        code = child.wait()
        writer.close()
        state = "finished" if code == 0 else "stopped" if code in (-signal.SIGINT, -signal.SIGTERM, 130) else "failed"
        fh.write(f"\n[{state}, exit code {code}, {dt.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
    status.update(ended=dt.datetime.now().astimezone().isoformat(timespec="seconds"), exit_code=code, state=state)
    write_status(status_path, status)
    return code
