"""Build this experiment's Ultralytics datasets from the Annotate tab's dataset.

    <run_dir>/data/
      cameras/<FIELD>_<Camera>/images/<stem>.png  frame shrunk to model.imgsz (link into the image cache)
      cameras/<FIELD>_<Camera>/labels/<stem>.txt  boxes, classes mapped by data.class_map
      cv_<FIELD>_<Camera>.yaml   fold: train on the other CV cameras, validate on this one
      final.yaml                 train on every CV camera; `test:` is the test cameras
      split.json                 every frame used, by camera, with counts and a fingerprint

Only frames with status "done" in annotations.json count (a frame submitted
with no boxes is a negative: an empty label file). Frames are shrunk once with
area interpolation and shared by experiments in <output>/image_cache/<imgsz>/,
because Ultralytics would shrink the 9216x5184 originals with INTER_LINEAR,
which aliases at this scale.
"""
import datetime as dt
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import yaml

from config import ConfigError

REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def slug(camera):
    """CAC/Camera1 -> CAC_Camera1 (folder and fold names)."""
    return camera.replace("/", "_")


def camera_of(rec):
    return f"{rec['field']}/{rec['camera']}" if rec.get("field") and rec.get("camera") else None


def load_annotations(ann_dir):
    """(class names by id, frame records) from the Annotate tab's folder."""
    manifest = os.path.join(ann_dir, "annotations.json")
    if not os.path.exists(manifest):
        raise ConfigError(f"no annotations.json in {ann_dir} (paths.annotations)")
    with open(manifest) as fh:
        images = json.load(fh).get("images", {})
    with open(os.path.join(ann_dir, "classes.txt")) as fh:
        classes = [ln.strip() for ln in fh if ln.strip()]
    return classes, [dict(r, stem=stem) for stem, r in sorted(images.items())]


def select(cfg, classes, records):
    """Assign "done" frames to the configured cameras and map their labels.

    Returns {"cameras": {camera: [frame]}, "unassigned": {camera: n}, "other_views": n,
    "dropped_boxes": n}; each frame is {stem, source, camera, taken, labels}.
    """
    sp = cfg["split"]
    listed = sp["test"] + sp["cross_validation"]
    views = sp.get("views")
    names = cfg["data"]["names"]
    cmap = cfg["data"]["class_map"]
    cameras = {c: [] for c in listed}
    unassigned, other_views, dropped = {}, 0, 0
    for r in records:
        cam = camera_of(r)
        if r.get("status") != "done" or cam is None:
            continue
        if views is not None and r.get("view") not in views:
            other_views += 1
            continue
        if cam not in cameras:
            unassigned[cam] = unassigned.get(cam, 0) + 1
            continue
        labels = []
        for c, x, y, w, h in r["boxes"]:
            label = classes[c] if 0 <= c < len(classes) else f"<class {c}>"
            if label not in cmap:
                raise ConfigError(f"annotation label {label!r} (on {r['stem']}) is not in data.class_map; "
                                  "map it to one of data.names, or to null to drop it")
            if cmap[label] is None:
                dropped += 1
                continue
            labels.append(f"{names.index(cmap[label])} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        cameras[cam].append({"stem": r["stem"], "source": r["source"], "camera": cam,
                             "taken": r.get("taken"), "labels": labels})
    empty = [c for c in listed if not cameras[c]]
    if empty:
        raise ConfigError(f"no finished annotations for {', '.join(empty)}"
                          f"{' in views ' + ', '.join(views) if views else ''}: check the camera names in split")
    return {"cameras": cameras, "unassigned": unassigned, "other_views": other_views, "dropped_boxes": dropped}


def fingerprint(cfg, sel):
    """Changes whenever the frames, their labels, the classes or the image size change."""
    h = hashlib.sha1()
    h.update(json.dumps([cfg["model"]["imgsz"], cfg["data"]["names"], cfg["split"]["test"],
                         cfg["split"]["cross_validation"]]).encode())
    for cam in sorted(sel["cameras"]):
        for f in sel["cameras"][cam]:
            h.update(json.dumps([cam, f["stem"], f["labels"]]).encode())
    return h.hexdigest()[:16]


def shrink_image(path, size):
    """Read `path` (BGR) with its long side shrunk to `size` px, never enlarged.

    JPEG decoding at 1/2, 1/4 or 1/8 scale is fast and itself anti-aliased;
    INTER_AREA takes it the rest of the way.
    """
    probe = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if probe is None:
        raise FileNotFoundError(f"cannot read image {path}")
    long8 = max(probe.shape[:2])  # ~ long side / 8
    factor = max((f for f in (8, 4, 2) if long8 * 8 // f >= size), default=1)
    im = probe if factor == 8 else cv2.imread(path, REDUCED[factor] if factor > 1 else cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    r = size / max(h, w)
    if r < 1:
        im = cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    return im


def cache_image(src, dst, size):
    """Shrunken PNG copy of `src` at `dst`, made once."""
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return False
    im = shrink_image(src, size)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.{os.getpid()}.tmp.png"
    if not cv2.imwrite(tmp, im, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f"could not write {tmp}")
    os.replace(tmp, dst)
    return True


def data_dir(cfg):
    return os.path.join(cfg["run_dir"], "data")


def read_split(cfg):
    path = os.path.join(data_dir(cfg), "split.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def describe(sel, cfg, log=print):
    sp = cfg["split"]
    log(f"{'Camera':<16}{'Role':<18}{'Frames':>7}{'Negatives':>11}{'Boxes':>7}")
    for cam in sp["test"] + sp["cross_validation"]:
        frames = sel["cameras"][cam]
        role = "test" if cam in sp["test"] else "cross-validation"
        neg = sum(1 for f in frames if not f["labels"])
        log(f"{cam:<16}{role:<18}{len(frames):>7}{neg:>11}{sum(len(f['labels']) for f in frames):>7}")
    if sel["unassigned"]:
        log("Annotated but not in split (not used): " +
            ", ".join(f"{c} ({n} frames)" for c, n in sorted(sel["unassigned"].items())))
    if sel["other_views"]:
        log(f"{sel['other_views']} annotated frames are in views other than {', '.join(sp['views'])} (not used)")
    if sel["dropped_boxes"]:
        log(f"{sel['dropped_boxes']} boxes dropped by data.class_map")


def write_yaml(path, comment, body):
    with open(path, "w") as fh:
        fh.write("".join(f"# {ln}\n" for ln in comment.splitlines()))
        yaml.safe_dump(body, fh, sort_keys=False)


def prepare(cfg, log=print, workers=16):
    """Write <run_dir>/data (see the module docstring). Returns its split.json dict."""
    classes, records = load_annotations(cfg["paths"]["annotations"])
    sel = select(cfg, classes, records)
    describe(sel, cfg, log)
    size = cfg["model"]["imgsz"]
    raw = cfg["paths"]["raw_data"]

    # Shrink every frame once (shared by experiments with the same imgsz).
    frames = [f for fs in sel["cameras"].values() for f in fs]
    jobs = [(os.path.join(raw, f["source"]), os.path.join(cfg["image_cache"], f["stem"] + ".png")) for f in frames]
    missing = [s for s, _ in jobs if not os.path.isfile(s)]
    if missing:
        raise ConfigError(f"{len(missing)} annotated frames are missing from {raw}, e.g. {missing[0]}")
    with ThreadPoolExecutor(max(1, min(workers, os.cpu_count() or 1))) as pool:
        made = sum(pool.map(lambda j: cache_image(j[0], j[1], size), jobs))
    log(f"Frames at {size} px: {made} shrunk now, {len(jobs) - made} already in {cfg['image_cache']}")

    # Build the dataset folder next to the old one, then swap it in.
    final_dir = data_dir(cfg)
    new = final_dir + ".new"
    shutil.rmtree(new, ignore_errors=True)
    for cam, fs in sel["cameras"].items():
        img_dir = os.path.join(new, "cameras", slug(cam), "images")
        lbl_dir = os.path.join(new, "cameras", slug(cam), "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)
        for f in fs:
            os.symlink(os.path.join(cfg["image_cache"], f["stem"] + ".png"), os.path.join(img_dir, f["stem"] + ".png"))
            with open(os.path.join(lbl_dir, f["stem"] + ".txt"), "w") as fh:
                fh.write("".join(ln + "\n" for ln in f["labels"]))

    names = dict(enumerate(cfg["data"]["names"]))
    cv, test = cfg["split"]["cross_validation"], cfg["split"]["test"]
    imgs = lambda cams: [f"cameras/{slug(c)}/images" for c in cams]  # noqa: E731
    folds = {}
    if len(cv) >= 2:
        for cam in cv:
            others = [c for c in cv if c != cam]
            path = os.path.join(new, f"cv_{slug(cam)}.yaml")
            write_yaml(path, f"Experiment {cfg['experiment']}, fold {cam}: train on {', '.join(others)}; "
                             f"validate on {cam}.\nWritten by yolo/pipeline.py prepare from {cfg['config_path']}.",
                       {"path": final_dir, "train": imgs(others), "val": imgs([cam]), "names": names})
            folds[cam] = os.path.join(final_dir, os.path.basename(path))
    body = {"path": final_dir, "train": imgs(cv),
            "val": imgs(cv)}  # Ultralytics needs one; the final model only re-scores its training frames on it
    if test:
        body["test"] = imgs(test)
    body["names"] = names
    write_yaml(os.path.join(new, "final.yaml"),
               f"Experiment {cfg['experiment']}, final model: train on every CV camera "
               f"({', '.join(cv)}).\n`val` repeats the training cameras (Ultralytics needs a val set); "
               f"`test` is held out: {', '.join(test) or 'none'}.\n"
               f"Written by yolo/pipeline.py prepare from {cfg['config_path']}.", body)

    split = {
        "experiment": cfg["experiment"],
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint(cfg, sel),
        "imgsz": size,
        "names": cfg["data"]["names"],
        "test": test,
        "cross_validation": cv,
        "folds": folds,
        "final": os.path.join(final_dir, "final.yaml"),
        "unassigned": sel["unassigned"],
        "cameras": {cam: {"frames": len(fs), "negatives": sum(1 for f in fs if not f["labels"]),
                          "boxes": sum(len(f["labels"]) for f in fs),
                          "images": [f["source"] for f in fs]}
                    for cam, fs in sel["cameras"].items()},
    }
    with open(os.path.join(new, "split.json"), "w") as fh:
        json.dump(split, fh, indent=1)
        fh.write("\n")

    old = final_dir + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final_dir):
        os.replace(final_dir, old)
    os.replace(new, final_dir)
    shutil.rmtree(old, ignore_errors=True)
    log(f"Datasets written to {final_dir}: {len(folds)} CV folds + final.yaml")
    return split


def current_fingerprint(cfg):
    """Fingerprint of what `prepare` would build now (no image work)."""
    classes, records = load_annotations(cfg["paths"]["annotations"])
    return fingerprint(cfg, select(cfg, classes, records))
