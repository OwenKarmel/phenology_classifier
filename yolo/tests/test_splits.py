import os

import cv2
import numpy as np
import pytest
import yaml

import config
import splits
from config import ConfigError


def load(project, *overrides):
    return config.load(project["config"], list(overrides))


def test_select_uses_only_listed_cameras_and_maps_classes(project):
    cfg = load(project)
    classes, records = splits.load_annotations(cfg["paths"]["annotations"])
    sel = splits.select(cfg, classes, records)
    assert set(sel["cameras"]) == {"CAC/Camera1", "CAC/Camera2", "FRC/Camera1", "LR/Camera1"}
    assert sel["unassigned"] == {"HC/Camera3": 1}
    assert sel["other_views"] == 1
    cac2 = {f["stem"]: f["labels"] for f in sel["cameras"]["CAC/Camera2"]}
    assert len(cac2) == 2  # the skipped frame is left out
    assert cac2["2025_CAC_Camera2_Across_20250611_073002"] == ["0 0.400000 0.400000 0.100000 0.100000"]  # AI -> 0
    assert cac2["2025_CAC_Camera2_Across_20250612_073002"] == []  # negative kept
    assert len(sel["cameras"]["LR/Camera1"]) == 2  # Flower view excluded


def test_null_class_map_drops_boxes(project):
    cfg = load(project, "data.class_map={grape_cluster: grape_cluster, AI_grape_cluster: null}")
    sel = splits.select(cfg, *splits.load_annotations(cfg["paths"]["annotations"]))
    assert sel["dropped_boxes"] == 2
    frc = {f["stem"]: f["labels"] for f in sel["cameras"]["FRC/Camera1"]}
    assert frc["2025_FRC_Camera1_Across_20250612_073002"] == ["0 0.500000 0.500000 0.100000 0.100000"]


def test_unmapped_label_is_an_error(project):
    cfg = load(project, "data.class_map={grape_cluster: grape_cluster}")
    with pytest.raises(ConfigError, match="AI_grape_cluster"):
        splits.select(cfg, *splits.load_annotations(cfg["paths"]["annotations"]))


def test_listed_camera_without_frames_is_an_error(project):
    cfg = load(project, "split.test=[CAT/Camera9]")
    with pytest.raises(ConfigError, match="CAT/Camera9"):
        splits.select(cfg, *splits.load_annotations(cfg["paths"]["annotations"]))


def test_views_filter_can_be_dropped(project):
    cfg = load(project, "split.views=null")
    sel = splits.select(cfg, *splits.load_annotations(cfg["paths"]["annotations"]))
    assert len(sel["cameras"]["LR/Camera1"]) == 3 and sel["other_views"] == 0


def test_shrink_keeps_aspect_and_never_enlarges(tmp_path):
    p = str(tmp_path / "a.jpg")
    cv2.imwrite(p, np.full((1000, 1600, 3), 128, np.uint8))
    assert splits.shrink_image(p, 640).shape == (400, 640, 3)
    assert splits.shrink_image(p, 1024).shape == (640, 1024, 3)
    assert splits.shrink_image(p, 3200).shape == (1000, 1600, 3)


def test_prepare_writes_folds_and_final(project):
    cfg = load(project)
    split = splits.prepare(cfg, log=lambda *a: None)
    data = os.path.join(cfg["run_dir"], "data")
    assert set(split["folds"]) == {"CAC/Camera2", "FRC/Camera1", "LR/Camera1"}

    test_dirs = {"cameras/CAC_Camera1/images"}
    for cam, path in split["folds"].items():
        with open(path) as fh:
            d = yaml.safe_load(fh)
        assert d["path"] == data and d["names"] == {0: "grape_cluster"}
        assert d["val"] == [f"cameras/{splits.slug(cam)}/images"]
        assert f"cameras/{splits.slug(cam)}/images" not in d["train"]  # leave one camera out
        assert not test_dirs & set(d["train"] + d["val"])  # test camera never seen
        assert len(d["train"]) == 2
    with open(split["final"]) as fh:
        final = yaml.safe_load(fh)
    assert final["test"] == ["cameras/CAC_Camera1/images"]
    assert len(final["train"]) == 3 and not test_dirs & set(final["train"])

    img_dir = os.path.join(data, "cameras", "CAC_Camera2", "images")
    lbl_dir = os.path.join(data, "cameras", "CAC_Camera2", "labels")
    assert sorted(os.listdir(img_dir)) == ["2025_CAC_Camera2_Across_20250611_073002.png",
                                           "2025_CAC_Camera2_Across_20250612_073002.png"]
    im = cv2.imread(os.path.join(img_dir, "2025_CAC_Camera2_Across_20250611_073002.png"))
    assert im.shape == (360, 640, 3)  # 1152x648 shrunk to imgsz 640
    with open(os.path.join(lbl_dir, "2025_CAC_Camera2_Across_20250612_073002.txt")) as fh:
        assert fh.read() == ""  # negative frame: empty label file
    assert split["cameras"]["CAC/Camera1"]["boxes"] == 3
    assert split["fingerprint"] == splits.current_fingerprint(cfg)


def test_prepare_rebuild_replaces_old_files(project):
    cfg = load(project)
    splits.prepare(cfg, log=lambda *a: None)
    cfg2 = load(project, "split.cross_validation=[FRC/Camera1, LR/Camera1]")
    split = splits.prepare(cfg2, log=lambda *a: None)
    data = os.path.join(cfg["run_dir"], "data")
    assert not os.path.exists(os.path.join(data, "cameras", "CAC_Camera2"))
    assert not os.path.exists(os.path.join(data, "cv_CAC_Camera2.yaml"))
    assert set(split["folds"]) == {"FRC/Camera1", "LR/Camera1"}


def test_single_cv_camera_has_no_folds(project):
    cfg = load(project, "split.cross_validation=[LR/Camera1]")
    split = splits.prepare(cfg, log=lambda *a: None)
    assert split["folds"] == {}
    assert os.path.exists(split["final"])


def test_fingerprint_follows_labels_and_size(project):
    cfg = load(project)
    fp = splits.current_fingerprint(cfg)
    assert splits.current_fingerprint(load(project, "train.epochs=3")) == fp  # training settings don't matter
    assert splits.current_fingerprint(load(project, "model.imgsz=1280")) != fp
    ann = project["ann"] / "annotations.json"
    ann.write_text(ann.read_text().replace("0.5, 0.5, 0.1, 0.1", "0.5, 0.5, 0.2, 0.1", 1))
    assert splits.current_fingerprint(cfg) != fp


def test_predict_names_frames_like_annotations(project, tmp_path):
    import engine

    raw = project["raw"]
    a = raw / "Raw_Data/2025/CAC/Camera1/Across/20250611_073002.jpg"
    b = raw / "Raw_Data/2025/LR/Camera1/Across/20250611_073002.jpg"  # same file name, other camera
    other = tmp_path / "loose" / "x.png"
    other.parent.mkdir()
    cv2.imwrite(str(other), np.zeros((10, 10, 3), np.uint8))
    files = engine.list_images([str(a), str(raw / "Raw_Data/2025/LR/Camera1/Across/*.jpg"), str(other.parent)],
                               str(raw))
    names = [n for _, n in files]
    assert names[0] == "2025_CAC_Camera1_Across_20250611_073002"
    assert "2025_LR_Camera1_Across_20250611_073002" in names and str(b) in [str(f) for f, _ in files]
    assert names[-1] == "loose_x"
    assert len(set(names)) == len(names)
    with pytest.raises(FileNotFoundError):
        engine.list_images([str(tmp_path / "nothing*.jpg")])
