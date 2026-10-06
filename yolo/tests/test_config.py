import os

import pytest
import yaml

import config
from config import ConfigError


def write(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def base():
    with open(config.DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


def test_repo_config_loads():
    cfg = config.load()
    assert cfg["split"]["test"] == ["CAC/Camera1"]
    assert cfg["split"]["cross_validation"] == ["CAC/Camera2", "FRC/Camera1", "LR/Camera1"]
    assert cfg["data"]["class_map"]["AI_grape_cluster"] == "grape_cluster"
    assert cfg["weights_path"].endswith("/weights/yolo26l.pt")
    assert cfg["run_dir"] == os.path.join(cfg["paths"]["output"], "runs", cfg["experiment"])
    assert cfg["image_cache"].endswith(f"image_cache/{cfg['model']['imgsz']}")
    assert os.path.isabs(cfg["paths"]["annotations"])


def test_overrides_are_yaml_typed():
    cfg = config.load(overrides=["train.epochs=7", "model.imgsz=1280", "split.test=[]", "train.cos_lr=true",
                                 "experiment=x1"])
    assert cfg["train"]["epochs"] == 7 and cfg["model"]["imgsz"] == 1280
    assert cfg["split"]["test"] == [] and cfg["train"]["cos_lr"] is True
    assert cfg["raw"]["train"]["epochs"] == 7  # saved with the run as given
    assert cfg["run_dir"].endswith("/runs/x1")


@pytest.mark.parametrize("override, msg", [
    ("split.test=[FRC/Camera1]", "both test and cross_validation"),
    ("split.cross_validation=[CAC_Camera2]", "FIELD/CameraN"),
    ("split.cross_validation=[]", "at least one camera"),
    ("split.test=[CAC/Camera1, CAC/Camera1]", "twice"),
    ("data.class_map={grape_cluster: flower}", "not in data.names"),
    ("data.names=[a, a]", "unique"),
    ("model.imgsz=1000", "divisible by 32"),
    ("train.imgsz=640", "set by the pipeline"),
    ("predict.source=x", "set by the pipeline"),
    ("train.val=false", "train.val"),
    ("experiment=has space", "experiment"),
    ("bogus.key=1", "unknown top-level"),
    ("nokey", "key=value"),
])
def test_bad_configs_are_refused(override, msg):
    with pytest.raises(ConfigError, match=msg):
        config.load(overrides=[override])


def test_no_camera_is_included_by_default(tmp_path):
    cfg = base()
    del cfg["split"]["cross_validation"]
    with pytest.raises(ConfigError, match="at least one camera"):
        config.load(write(tmp_path, cfg))


def test_relative_paths_resolve_from_project(tmp_path):
    cfg = config.load(write(tmp_path, base()))
    assert cfg["paths"]["annotations"] == os.path.join(config.PROJECT, "annotated_images")
