"""Experiment bookkeeping in pipeline.py (no training)."""
import os

import pytest

import config
import pipeline
import splits
from config import ConfigError


def load(project, *overrides):
    return config.load(project["config"], list(overrides))


def fake_started(cfg):
    os.makedirs(os.path.join(cfg["run_dir"], "cv", "CAC_Camera2", "weights"))


def test_ensure_data_reuses_up_to_date_datasets(project):
    cfg = load(project)
    a = pipeline.ensure_data(cfg)
    b = pipeline.ensure_data(cfg)
    assert a["created"] == b["created"]


def test_changed_annotations_rebuild_before_training(project):
    cfg = load(project)
    a = pipeline.ensure_data(cfg)
    ann = project["ann"] / "annotations.json"
    ann.write_text(ann.read_text().replace('"status": "skipped"', '"status": "done"'))
    b = pipeline.ensure_data(cfg)
    assert b["fingerprint"] != a["fingerprint"]
    assert b["cameras"]["CAC/Camera2"]["frames"] == 3


def test_changed_annotations_are_refused_once_training_started(project):
    cfg = load(project)
    pipeline.ensure_data(cfg)
    fake_started(cfg)
    ann = project["ann"] / "annotations.json"
    ann.write_text(ann.read_text().replace('"status": "skipped"', '"status": "done"'))
    with pytest.raises(ConfigError, match="changed since"):
        pipeline.ensure_data(cfg)


def test_changed_settings_are_refused_once_training_started(project):
    cfg = load(project)
    pipeline.save_config(cfg)
    fake_started(cfg)
    pipeline.check_same_settings(load(project, "train.device=0", "train.workers=2"))  # runtime only: fine
    with pytest.raises(ConfigError, match=r"train.epochs: \d+ -> 120"):
        pipeline.check_same_settings(load(project, "train.epochs=120"))
    with pytest.raises(ConfigError, match="model.imgsz"):
        pipeline.check_same_settings(load(project, "model.imgsz=1280"))


def test_saved_config_round_trips(project):
    cfg = load(project, "train.epochs=5")
    pipeline.save_config(cfg)
    again = config.load(os.path.join(cfg["run_dir"], "config.yaml"))
    assert again["train"]["epochs"] == 5 and again["split"] == cfg["split"]


def test_smoke_and_set_parse():
    args = pipeline.parse(["train", "--smoke", "--set", "train.epochs=1", "experiment=z"])
    assert args.command == "train" and args.smoke and args.set == ["train.epochs=1", "experiment=z"]
    args = pipeline.parse(["cv", "--fold", "LR/Camera1", "FRC/Camera1"])
    assert args.fold == ["LR/Camera1", "FRC/Camera1"]
    args = pipeline.parse(["predict", "a.jpg", "dir/", "--weights", "w.pt"])
    assert args.source == ["a.jpg", "dir/"] and args.weights == "w.pt"


def test_prepare_command(project, capsys):
    assert pipeline.main(["prepare", "--config", project["config"]]) == 0
    cfg = load(project)
    assert splits.read_split(cfg)["folds"]
    assert "Annotated but not in split (not used): HC/Camera3 (1 frames)" in capsys.readouterr().out


def test_test_without_final_model_fails_cleanly(project, capsys):
    pipeline.main(["prepare", "--config", project["config"]])
    assert pipeline.main(["test", "--no-log", "--config", project["config"]]) == 2
    assert "final model isn't trained yet" in capsys.readouterr().out


def test_bad_config_exits_with_message(project):
    with pytest.raises(SystemExit, match="config: .*divisible by 32"):
        pipeline.main(["prepare", "--config", project["config"], "--set", "model.imgsz=1000"])
