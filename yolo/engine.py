"""Training, evaluation and inference with Ultralytics: sections 5 and 6 of
Train_YOLO_Models.ipynb, for the datasets that splits.py builds.

    <run_dir>/
      config.yaml               the config (with --set overrides) this experiment ran with
      data/                     datasets and data.yaml files (splits.py)
      cv/<FIELD>_<Camera>/      one leave-one-camera-out fold: the Ultralytics run (weights/last.pt,
                                best.pt, results.csv, plots), eval_last/ and eval_best/ (scores on
                                the held-out camera) and done.json (those scores)
      final/                    the final model, trained on every CV camera: weights/last.pt
      test/                     eval/ (scores on the test cameras), predict/ (test frames with the
                                predicted boxes) and done.json
      summary.json              CV and test scores

A stage with done.json is finished and is skipped when `train` runs again;
an interrupted one resumes from its weights/last.pt.
"""
import csv
import datetime as dt
import glob
import json
import os
import re
import shutil
import statistics
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from splits import shrink_image, slug

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METRICS = ("precision", "recall", "mAP50", "mAP50-95")
_YOLO = None


def init(cfg):
    """Import Ultralytics with its settings kept under <output>/ultralytics:
    checkpoints in <output>/weights, analytics off."""
    global _YOLO
    out = cfg["paths"]["output"]
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(out, "ultralytics"))
    os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)  # else Ultralytics falls back to /tmp
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    import ultralytics.utils as uu
    from ultralytics import YOLO, settings

    settings.update({"weights_dir": cfg["weights_dir"], "runs_dir": os.path.join(out, "runs"),
                     "datasets_dir": os.path.join(out, "datasets"), "sync": False})
    uu.WEIGHTS_DIR = Path(cfg["weights_dir"])  # read at import; the AMP check's yolo26n.pt goes here
    _YOLO = YOLO
    return YOLO


def pull(cfg, names=None, log=print):
    """Download YOLO26 checkpoints (and yolo26n.pt, which the AMP check uses)."""
    from ultralytics.utils.downloads import attempt_download_asset

    for name in names or sorted({os.path.basename(cfg["weights_path"]), "yolo26n.pt"}):
        dest = os.path.join(cfg["weights_dir"], name)
        if not os.path.exists(dest):
            attempt_download_asset(dest)
        log(f"{name}: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def device_args(cfg, section):
    """`section` settings, on the training GPU unless the section names its own device."""
    args = dict(cfg[section])
    if "device" not in args and "device" in cfg["train"]:
        args["device"] = cfg["train"]["device"]
    return args


# ------------------------------------------------------------------ training

def finished_training(last):
    """True once Ultralytics has finalised last.pt (optimizer stripped, epoch = -1)."""
    from ultralytics.utils.patches import torch_load

    ckpt = torch_load(last, map_location="cpu")
    return ckpt.get("epoch", -1) == -1


def train(cfg, data_yaml, out_dir, val, log=print, extra=None, callbacks=None):
    """Train one model into out_dir (an Ultralytics project/name), resuming an
    interrupted run there. Returns weights/last.pt. `callbacks`: {event: fn}
    added to the Ultralytics model."""
    last = os.path.join(out_dir, "weights", "last.pt")
    if os.path.exists(last) and finished_training(last):
        log(f"{out_dir}: training already finished, using {last}")
        return last
    if os.path.exists(last):
        log(f"Resuming {last}")
        # Resume keeps the checkpoint's settings; the device and workers may change.
        keep = {k: cfg["train"][k] for k in ("device", "workers", "batch", "cache") if k in cfg["train"]}
        model = _YOLO(last)
        for event, fn in (callbacks or {}).items():
            model.add_callback(event, fn)
        model.train(resume=True, **keep)
    else:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)  # a run that died before its first checkpoint
        project, name = os.path.split(out_dir)
        args = {**cfg["train"], **(extra or {})}
        model = _YOLO(cfg["weights_path"])
        for event, fn in (callbacks or {}).items():
            model.add_callback(event, fn)
        model.train(data=data_yaml, imgsz=cfg["model"]["imgsz"], project=project, name=name,
                    exist_ok=True, val=val, **args)
    if not os.path.exists(last):
        raise RuntimeError(f"training finished without {last}")
    return last


def epoch_stats(run_dir):
    """Rows of results.csv as dicts of floats (keys stripped)."""
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [{k.strip(): float(v) for k, v in row.items() if v not in (None, "")} for row in csv.DictReader(fh)]


def best_epoch(rows):
    """Epoch with the best Ultralytics fitness (mAP50-95 here), from results.csv."""
    key = "metrics/mAP50-95(B)"
    rows = [r for r in rows if key in r]
    return int(max(rows, key=lambda r: r[key])["epoch"]) if rows else None


# ---------------------------------------------------------------- evaluation

def scores(m):
    b = m.box
    out = {"precision": float(b.mp), "recall": float(b.mr), "mAP50": float(b.map50), "mAP50-95": float(b.map)}
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    if len(b.ap_class_index) > 1:
        out["per_class"] = {names[int(c)]: {"precision": float(b.p[i]), "recall": float(b.r[i]),
                                            "mAP50": float(b.ap50[i]), "mAP50-95": float(b.ap[i])}
                            for i, c in enumerate(b.ap_class_index)}
    return out


def evaluate(cfg, weights, data_yaml, split, out_dir, name):
    """model.val() on one split of data_yaml; plots and predictions go to out_dir/name."""
    m = _YOLO(weights).val(data=data_yaml, split=split, imgsz=cfg["model"]["imgsz"], project=out_dir,
                           name=name, exist_ok=True, **device_args(cfg, "evaluate"))
    return scores(m)


def fmt_scores(s):
    return "  ".join(f"{k} {s[k]:.3f}" for k in METRICS)


# ----------------------------------------------------------------- inference

def raw_name(path, raw):
    """Name of a frame inside raw_data, as in annotated_images:
    raw_data/Raw_Data/2025/LR/Camera1/Across/20250620_133001.jpg -> 2025_LR_Camera1_Across_20250620_133001"""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(raw))
    if rel.startswith(".."):
        return None
    parts = Path(rel).with_suffix("").parts
    if parts[0] == "Raw_Data":
        parts = parts[1:]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(parts))


def list_images(sources, raw=None):
    """Image files under each source (a file, a folder searched recursively, or a glob),
    each with an output name that is unique within the run: frames in raw_data are
    named as in annotated_images, others by file name (with their folder for folders)."""
    out = []
    for src in sources:
        p = Path(src).expanduser()
        if p.is_file():
            out.append((p, p.stem))
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in IMAGE_EXT and f.is_file():
                    rel = f.relative_to(p).with_suffix("")
                    out.append((f, "_".join((p.name, *rel.parts))))
        else:
            files = [Path(f) for f in sorted(glob.glob(os.path.expanduser(src), recursive=True))
                     if os.path.isfile(f) and Path(f).suffix.lower() in IMAGE_EXT]
            if not files:
                raise FileNotFoundError(f"no images found at {src}")
            out.extend((f, f.stem) for f in files)
    if raw:
        out = [(f, raw_name(f, raw) or name) for f, name in out]
    seen = {}
    for i, (f, name) in enumerate(out):
        n = seen.get(name, 0)
        seen[name] = n + 1
        if n:
            out[i] = (f, f"{name}_{n}")
    return out


def predict(cfg, weights, sources, out_dir, log=print):
    """Detect clusters in every image under `sources` (sections 6 and 7 of the notebook).

    Each image is shrunk to model.imgsz exactly as the training frames were.
    Writes out_dir/images/<name>.jpg (boxes drawn), labels/<name>.txt
    (class x_center y_center width height confidence, normalised to the image,
    so they hold for the full-size frame too) and detections.csv.
    """
    files = list_images(sources, cfg["paths"]["raw_data"])
    if not files:
        raise FileNotFoundError("no images to predict on")
    args = device_args(cfg, "predict")
    batch = int(args.pop("batch", 4))
    save_img = args.pop("save", True)
    save_txt = args.pop("save_txt", True)
    save_conf = args.pop("save_conf", True)
    model = _YOLO(weights)
    names = model.names
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    size = cfg["model"]["imgsz"]
    log(f"Predicting on {len(files)} images with {weights} at {size} px -> {out_dir}")

    def load(item):
        return shrink_image(str(item[0]), size)

    rows, total, t0 = [], 0, time.time()
    with ThreadPoolExecutor(8) as pool, open(os.path.join(out_dir, "detections.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "source", "detections", *[f"n_{names[i]}" for i in sorted(names)], "max_conf"])
        chunks = deque(files[i:i + batch] for i in range(0, len(files), batch))
        ahead = deque()  # the next few batches load while the GPU works
        while chunks or ahead:
            while chunks and len(ahead) < 3:
                ch = chunks.popleft()
                ahead.append((ch, [pool.submit(load, it) for it in ch]))
            chunk, futures = ahead.popleft()
            results = model.predict([f.result() for f in futures], imgsz=size, verbose=False, **args)
            for (src, name), r in zip(chunk, results):
                cls = r.boxes.cls.int().tolist()
                conf = r.boxes.conf.tolist()
                if save_img:
                    cv2.imwrite(os.path.join(out_dir, "images", name + ".jpg"), r.plot(line_width=args.get("line_width")),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                if save_txt:
                    with open(os.path.join(out_dir, "labels", name + ".txt"), "w") as lf:
                        for c, xywh, p in zip(cls, r.boxes.xywhn.tolist(), conf):
                            lf.write(f"{c} " + " ".join(f"{v:.6f}" for v in xywh) + (f" {p:.4f}" if save_conf else "") + "\n")
                w.writerow([name, str(src), len(cls), *[cls.count(i) for i in sorted(names)],
                            f"{max(conf):.4f}" if conf else ""])
                rows.append(len(cls))
            total += len(chunk)
            log(f"  {total}/{len(files)} images, {sum(rows)} detections ({time.time() - t0:.0f} s)")
    return {"images": len(files), "detections": sum(rows), "out_dir": out_dir}


# ----------------------------------------------------------------- summaries

def cv_summary(folds):
    """Mean and standard deviation over folds, for last.pt and best.pt."""
    out = {}
    for which in ("last", "best"):
        vals = {k: [f[which][k] for f in folds.values() if f.get(which)] for k in METRICS}
        out[which] = {k: {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0}
                      for k, v in vals.items() if v}
    return out


def summarize(cfg, split, log=print):
    run = cfg["run_dir"]
    folds = {}
    for cam in split["folds"]:
        d = read_json(os.path.join(run, "cv", slug(cam), "done.json"))
        if d:
            folds[cam] = d
    final = read_json(os.path.join(run, "final", "done.json"))
    test = read_json(os.path.join(run, "test", "done.json"))
    summary = {"experiment": cfg["experiment"], "updated": dt.datetime.now().isoformat(timespec="seconds"),
               "model": cfg["model"], "epochs": cfg["train"].get("epochs"), "folds": folds,
               "cv": cv_summary(folds) if folds else None, "final": final, "test": test}
    write_json(os.path.join(run, "summary.json"), summary)

    log(f"\nExperiment {cfg['experiment']} ({run})")
    if folds:
        log(f"\nLeave-one-camera-out CV, scored with last.pt (best.pt in brackets: picked on the held-out camera, optimistic)")
        log(f"{'held-out camera':<18}{'frames':>7}" + "".join(f"{k:>18}" for k in METRICS) + f"{'best epoch':>12}")
        for cam, d in folds.items():
            cells = (f"{d['last'][k]:.3f} ({d['best'][k]:.3f})" for k in METRICS)
            log(f"{cam:<18}{d['val_frames']:>7}" + "".join(f"{c:>18}" for c in cells) + f"{d.get('best_epoch') or '-':>12}")
        s = summary["cv"]["last"]
        cells = (f"{s[k]['mean']:.3f} ± {s[k]['sd']:.3f}" for k in METRICS)
        log(f"{'mean ± sd':<25}" + "".join(f"{c:>18}" for c in cells))
        missing = [c for c in split["folds"] if c not in folds]
        if missing:
            log(f"Folds not finished: {', '.join(missing)}")
    if test:
        log(f"\nHeld-out test ({', '.join(split['test'])}, {test['frames']} frames), final model last.pt:")
        log("  " + fmt_scores(test["scores"]))
        log(f"  predictions: {test['predict_dir']}")
    elif final:
        log(f"\nFinal model: {final['weights']} (not tested yet: run `test`)")
    return summary
