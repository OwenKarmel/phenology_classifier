import io
import json
import os
import sys
import textwrap

import pytest

import runlog


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(runlog.time, "monotonic", c)
    return c


def test_plain_lines_pass_through_without_colours(clock):
    out = io.StringIO()
    w = runlog.LogWriter(out)
    w.feed(b"\x1b[1mhello\x1b[0m world\nsecond ")
    w.feed(b"line\n")
    w.close()
    assert out.getvalue() == "hello world\nsecond line\n"


def test_progress_redraws_are_thinned(clock):
    out = io.StringIO()
    w = runlog.LogWriter(out, interval=2.0)
    w.feed(b"\r\x1b[K 1/10")
    assert out.getvalue() == ""   # a frame is finished only when the next \r comes
    w.feed(b"\r\x1b[K 2/10")
    assert out.getvalue() == " 1/10"  # first frame: written at once, no \r in front
    w.feed(b"\r\x1b[K 3/10")      # within the interval: held back
    assert out.getvalue() == " 1/10"
    clock.t += 3
    w.feed(b"\r\x1b[K 4/10")
    assert out.getvalue() == " 1/10\r 3/10"  # only the newest finished frame
    clock.t += 3
    w.tick()                      # quiet stream: the held frame goes out on the timer
    assert out.getvalue() == " 1/10\r 3/10"  # 4/10 isn't finished yet
    w.feed(b"\r\x1b[K10/10\n")    # the finished bar replaces the last frame
    w.feed(b"next\n")
    assert out.getvalue() == " 1/10\r 3/10\r10/10\nnext\n"


def test_reader_view_matches_terminal(clock):
    """What the Model tab shows: per line, the text after the last \\r."""
    out = io.StringIO()
    w = runlog.LogWriter(out, interval=0)
    for i in range(1, 6):
        w.feed(f"\repoch 1: {i}/5".encode())
        clock.t += 1
    w.feed(b"\nDone\n")
    shown = [ln.split("\r")[-1] for ln in out.getvalue().split("\n")]
    assert shown == ["epoch 1: 5/5", "Done", ""]


def test_utf8_split_across_reads(clock):
    out = io.StringIO()
    w = runlog.LogWriter(out)
    data = "✅ ok ━━\n".encode()
    for i in range(len(data)):
        w.feed(data[i:i + 1])
    assert out.getvalue() == "✅ ok ━━\n"


def test_newline_after_shown_frame_keeps_it(clock):
    out = io.StringIO()
    w = runlog.LogWriter(out, interval=0)
    w.feed(b"\rframe A\r")
    clock.t += 1
    w.tick()
    w.feed(b"\n")
    assert out.getvalue().split("\n")[0].split("\r")[-1] == "frame A"


def test_run_logged_records_output_and_status(tmp_path, monkeypatch, capsysbinary):
    monkeypatch.setattr(runlog, "LOG_DIR", str(tmp_path / "logs"))
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys
        assert os.environ["{runlog.CHILD_ENV}"].endswith(".log")
        print("args", sys.argv[1:])
        sys.stdout.write("\\r 1/2\\r 2/2\\n")
        print("err", file=sys.stderr)
        sys.exit(3)
    """))
    code = runlog.run_logged("train", "exp1", ["train", "--x"], str(script))
    assert code == 3
    (status_file,) = [p for p in os.listdir(tmp_path / "logs") if p.endswith(".json")]
    st = json.loads((tmp_path / "logs" / status_file).read_text())
    assert st["state"] == "failed" and st["exit_code"] == 3 and st["experiment"] == "exp1"
    assert st["started"][-6] in "+-" and st["ended"]  # timestamps carry the UTC offset
    log = (tmp_path / "logs" / st["log"]).read_text()
    assert "args ['train', '--x']" in log and "err" in log and " 2/2" in log and "[failed, exit code 3" in log
    assert b"args ['train', '--x']" in capsysbinary.readouterr().out  # terminal copy, unchanged


def test_run_logged_success(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "LOG_DIR", str(tmp_path / "logs"))
    script = tmp_path / "ok.py"
    script.write_text("print('fine')\n")
    assert runlog.run_logged("test", "e", ["test"], str(script)) == 0
    (status_file,) = [p for p in os.listdir(tmp_path / "logs") if p.endswith(".json")]
    assert json.loads((tmp_path / "logs" / status_file).read_text())["state"] == "finished"


def test_kill_parent_stops_child_and_records_it(tmp_path):
    """`kill <pid>` (the pid in the Model tab) stops the run and marks it stopped."""
    import signal
    import subprocess
    import time as _time

    child = tmp_path / "child.py"
    child.write_text("import time\nprint('ready', flush=True)\ntime.sleep(60)\n")
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.path.dirname(runlog.__file__)!r})
        import runlog
        runlog.LOG_DIR = {str(tmp_path / "logs")!r}
        sys.exit(runlog.run_logged("train", "e", ["train"], {str(child)!r}))
    """))
    p = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert p.stdout.readline().strip() == b"ready"
    t0 = _time.time()
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=20)
    assert _time.time() - t0 < 10  # the child didn't sleep out its 60 s
    (status_file,) = [f for f in os.listdir(tmp_path / "logs") if f.endswith(".json")]
    st = json.loads((tmp_path / "logs" / status_file).read_text())
    assert st["state"] == "stopped" and st["exit_code"] == -signal.SIGTERM
