"""The viewer's Model-tab endpoints (visualization/server.py), on fake logs."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "visualization"))
import server  # noqa: E402


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_LOGS", str(tmp_path))
    return tmp_path


def add_run(d, name, text, **status):
    (d / f"{name}.log").write_bytes(text.encode() if isinstance(text, str) else text)
    (d / f"{name}.json").write_text(json.dumps({"state": "finished", "pid": os.getpid(), **status}))


def test_runs_newest_first_and_dead_running_is_lost(logs):
    add_run(logs, "20261001-100000_train_a", "x\n")
    add_run(logs, "20261002-100000_train_b", "y\n", state="running", pid=2**22 + 12345)
    add_run(logs, "20261003-100000_cv_c", "z\n", state="running")  # this process: alive
    runs = server.model_runs()["runs"]
    assert [r["name"] for r in runs] == ["20261003-100000_cv_c.log", "20261002-100000_train_b.log",
                                         "20261001-100000_train_a.log"]
    assert [r["state"] for r in runs] == ["running", "lost", "finished"]
    assert runs[0]["size"] == 2


def test_missing_log_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_LOGS", str(tmp_path / "nope"))
    assert server.model_runs()["runs"] == []


def test_incremental_reads(logs):
    add_run(logs, "r", "line 1\nline 2\n")
    first = server.model_log("r.log", 0)
    assert first["text"] == "line 1\nline 2\n" and first["next"] == 14
    with open(logs / "r.log", "a") as fh:
        fh.write("\r 1/3\r 2/3")
    more = server.model_log("r.log", first["next"])
    assert more["offset"] == 14 and more["text"] == "\r 1/3\r 2/3"
    assert server.model_log("r.log", more["next"])["text"] == ""


def test_tail_starts_at_a_line(logs, monkeypatch):
    monkeypatch.setattr(server, "LOG_TAIL", 20)
    add_run(logs, "big", "".join(f"line {i:03d}\n" for i in range(10)))  # 9 bytes a line
    r = server.model_log("big.log", -1)
    assert r["text"] == "line 008\nline 009\n" and r["skipped"] == r["offset"] == 72


def test_utf8_cut_is_held_back(logs, monkeypatch):
    monkeypatch.setattr(server, "LOG_CHUNK", 4)
    add_run(logs, "u", "ab✅")  # ✅ is 3 bytes: the first read must not split it
    r = server.model_log("u.log", 0)
    assert r["text"] == "ab" and r["next"] == 2
    r = server.model_log("u.log", r["next"])
    assert r["text"] == "✅"


@pytest.mark.parametrize("name", ["../x.log", "a/b.log", "x.json", "", ".."])
def test_log_names_are_checked(logs, name):
    with pytest.raises((ValueError, FileNotFoundError)):
        server.model_log(name, 0)


def test_utf8_prefix():
    assert server.utf8_prefix(b"abc") == b"abc"
    assert server.utf8_prefix("é".encode()) == "é".encode()
    assert server.utf8_prefix("é".encode()[:1]) == b""
    assert server.utf8_prefix(b"a" + "✅".encode()[:2]) == b"a"
    assert server.utf8_prefix("✅".encode()) == "✅".encode()
    assert server.utf8_prefix("😀".encode()[:3]) == b""
    assert server.utf8_prefix(b"") == b""
