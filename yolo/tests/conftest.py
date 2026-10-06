"""Fixtures: a small fake Annotate-tab dataset and a config pointing at it.

    yolo/.venv/bin/python -m pytest yolo/tests
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.dirname(HERE)
sys.path.insert(0, YOLO_DIR)

import config  # noqa: E402

W, H = 1152, 648  # 16:9 like the 9216x5184 trailcam frames


def make_frame(path, seed):
    rng = np.random.default_rng(seed)
    im = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, im)


def record(field, camera, ts, status="done", boxes=None, view="Across"):
    source = f"Raw_Data/2025/{field}/{camera}/{view}/{ts}.jpg"
    stem = f"2025_{field}_{camera}_{view}_{ts}"
    return stem, {"source": source, "image": f"images/{stem}.jpg", "label": f"labels/{stem}.txt",
                  "year": "2025", "field": field, "camera": camera, "view": view,
                  "taken": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}",
                  "width": W, "height": H, "split": "train", "status": status,
                  "boxes": boxes if boxes is not None else [[0, 0.5, 0.5, 0.1, 0.1]]}


@pytest.fixture
def project(tmp_path):
    """tmp project: raw_data with frames, annotated_images, output dir, config file."""
    raw = tmp_path / "raw_data"
    ann = tmp_path / "annotated_images"
    ann.mkdir()
    (ann / "classes.txt").write_text("grape_cluster\nAI_grape_cluster\n")
    recs = dict([
        record("CAC", "Camera1", "20250611_073002"),
        record("CAC", "Camera1", "20250612_073002", boxes=[[0, 0.2, 0.3, 0.05, 0.08], [0, 0.6, 0.4, 0.04, 0.05]]),
        record("CAC", "Camera2", "20250611_073002", boxes=[[1, 0.4, 0.4, 0.1, 0.1]]),
        record("CAC", "Camera2", "20250612_073002", boxes=[]),                      # negative
        record("CAC", "Camera2", "20250613_073002", status="skipped"),              # left out
        record("FRC", "Camera1", "20250611_073002"),
        record("FRC", "Camera1", "20250612_073002", boxes=[[1, 0.1, 0.1, 0.05, 0.05], [0, 0.5, 0.5, 0.1, 0.1]]),
        record("LR", "Camera1", "20250611_073002"),
        record("LR", "Camera1", "20250612_133001"),
        record("HC", "Camera3", "20250611_073002"),                                # not in the split
        record("LR", "Camera1", "20250611_073002", view="Flower"),                 # other view
    ])
    for i, r in enumerate(recs.values()):
        make_frame(str(raw / r["source"]), i)
    (ann / "annotations.json").write_text(json.dumps({"version": 1, "images": recs}))
    with open(config.DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    cfg["experiment"] = "unit"
    cfg["paths"] = {"annotations": str(ann), "raw_data": str(raw), "output": str(tmp_path / "out")}
    cfg["model"]["imgsz"] = 640
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return {"root": tmp_path, "config": str(path), "raw": raw, "ann": ann, "records": recs}
