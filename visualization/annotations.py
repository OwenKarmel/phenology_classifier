"""YOLO bounding-box annotations made in the viewer's Annotate tab.

Everything lives in one folder (default: <project>/annotated_images), laid out
like a Label Studio "YOLO" export plus what Ultralytics needs to train on it:

    classes.txt         class names, one per line; line n is class id n
    notes.json          the same classes, Label Studio style
    dataset.yaml        Ultralytics dataset config (train.txt / val.txt, names)
    train.txt, val.txt  the images of each split, one ./images/<stem> path per line
    images/<stem>.jpg   symlink to the original frame in raw_data
    labels/<stem>.txt   one line per box: class x_center y_center width height,
                        normalized to 0-1. An empty file = no clusters (a negative).
    annotations.json    every annotated or skipped frame: source path, camera,
                        capture time, image size, status, boxes and split

Only frames with status "done" get a label file and an image link. "skipped"
frames are remembered, with any boxes they had, but left out of the dataset.
The train/val split is fixed per camera folder and day (hash of folder + date),
so near-identical frames from one day never end up on both sides.
"""
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import threading

from PIL import Image

from build_index import TS_RE, trailcam_folder_info

DEFAULT_CLASSES = ["grape_cluster"]
VAL_PERCENT = 20
MAX_BOXES = 2000


def stem_for(source):
    """File stem for a raw_data-relative frame path, unique across cameras.

    Raw_Data/2025/CAT/Camera1/Across/20250707_133001.jpg
        -> 2025_CAT_Camera1_Across_20250707_133001
    """
    parts = source.split("/")
    if parts[0] == "Raw_Data":
        parts = parts[1:]
    parts[-1] = os.path.splitext(parts[-1])[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(parts))


def atomic_write(path, text):
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def remove(path):
    if os.path.lexists(path):
        os.remove(path)


class Store:
    def __init__(self, root, raw_root):
        self.root = os.path.abspath(root)
        # Link through raw_data itself (not its resolved target) so the links
        # stay relative inside the project.
        self.raw = os.path.abspath(raw_root)
        self.manifest = os.path.join(self.root, "annotations.json")
        self.lock = threading.Lock()
        self.images = {}
        if os.path.exists(self.manifest):
            with open(self.manifest) as fh:
                self.images = json.load(fh).get("images", {})

    # ---------------------------------------------------------------- read

    def classes(self):
        try:
            with open(os.path.join(self.root, "classes.txt")) as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
            return names or list(DEFAULT_CLASSES)
        except FileNotFoundError:
            return list(DEFAULT_CLASSES)

    def snapshot(self):
        with self.lock:
            return {
                "dir": self.root,
                "classes": self.classes(),
                "images": list(self.images.values()),
            }

    # --------------------------------------------------------------- write

    def save(self, source, path, status, boxes):
        """Set one frame's status ("done", "skipped" or "none") and boxes.

        Boxes are [class, x_center, y_center, width, height], normalized.
        Returns the stored record, or None once the frame is unlabeled again.
        """
        if status not in ("done", "skipped", "none"):
            raise ValueError(f"unknown status {status!r}")
        with self.lock:
            classes = self.classes()
            boxes = clean_boxes(boxes, len(classes))
            stem = stem_for(source)
            rec = self.images.get(stem)
            if status == "none":
                self.images.pop(stem, None)
                rec = None
            else:
                if rec is None or rec.get("source") != source:
                    rec = self.new_record(source, path, stem)
                rec.update(status=status, boxes=boxes,
                           updated=dt.datetime.now().isoformat(timespec="seconds"))
                self.images[stem] = rec
            self.write_files(stem, source, rec, path)
            self.write_dataset(classes)
            return rec

    def set_classes(self, names):
        """Rename or add classes. Ids are line numbers, so a class can only be
        removed from the end of the list, and only while no box uses it."""
        if not isinstance(names, list) or not names:
            raise ValueError("need a list of class names")
        names = [str(n).strip() for n in names]
        if any(not n or "\n" in n for n in names) or len(set(names)) != len(names):
            raise ValueError("class names must be unique and not empty")
        with self.lock:
            used = {b[0] for r in self.images.values() for b in r["boxes"]}
            if any(c >= len(names) for c in used):
                raise ValueError("a removed class is still used by some boxes")
            os.makedirs(self.root, exist_ok=True)
            atomic_write(os.path.join(self.root, "classes.txt"), "".join(n + "\n" for n in names))
            self.write_dataset(names)
            return names

    def new_record(self, source, path, stem):
        rel_dir, name = os.path.split(source)
        year, field, camera, view = trailcam_folder_info(rel_dir) \
            if source.startswith("Raw_Data/") and rel_dir.count("/") >= 2 else (None,) * 4
        m = TS_RE.match(name)
        taken = None
        if m:
            ts = m.group(1)
            taken = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        with Image.open(path) as im:
            width, height = im.size
            if im.getexif().get(274) in (5, 6, 7, 8):  # EXIF says rotated 90 degrees
                width, height = height, width
        # Same camera folder + same day -> same split.
        key = f"{rel_dir}/{(taken or name)[:10]}"
        val = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100 < VAL_PERCENT
        ext = os.path.splitext(name)[1].lower() or ".jpg"
        return {
            "source": source,
            "image": f"images/{stem}{ext}",
            "label": f"labels/{stem}.txt",
            "year": year, "field": field, "camera": camera, "view": view,
            "taken": taken,
            "width": width, "height": height,
            "split": "val" if val else "train",
        }

    def write_files(self, stem, source, rec, path):
        """Label file + image link for a "done" frame; neither for anything else."""
        os.makedirs(os.path.join(self.root, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "labels"), exist_ok=True)
        label = os.path.join(self.root, "labels", stem + ".txt")
        link = os.path.join(self.root, "images", stem + (os.path.splitext(source)[1].lower() or ".jpg"))
        if not rec or rec["status"] != "done":
            remove(label)
            remove(link)
        else:
            atomic_write(label, "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rec["boxes"]))
            target = os.path.relpath(os.path.join(self.raw, source), os.path.dirname(link))
            if not (os.path.islink(link) and os.readlink(link) == target):
                remove(link)
                try:
                    os.symlink(target, link)
                except OSError:  # no symlinks on this filesystem: copy instead
                    shutil.copy2(path, link)
        self.write_manifest()

    def write_manifest(self):
        # One record per line keeps the file readable and diff-friendly.
        lines = [f"    {json.dumps(k)}: {json.dumps(self.images[k])}" for k in sorted(self.images)]
        atomic_write(self.manifest, "{\n"
                     '  "version": 1,\n'
                     '  "about": "One record per frame annotated or skipped in the viewer. '
                     'Boxes are YOLO [class, x_center, y_center, width, height], normalized to the image. '
                     'source is relative to source_root.",\n'
                     f'  "source_root": {json.dumps(os.path.relpath(self.raw, self.root))},\n'
                     '  "images": {\n' + ",\n".join(lines) + ("\n" if lines else "") + "  }\n}\n")

    def write_dataset(self, classes):
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(os.path.join(self.root, "classes.txt")):
            atomic_write(os.path.join(self.root, "classes.txt"), "".join(n + "\n" for n in classes))
        atomic_write(os.path.join(self.root, "notes.json"), json.dumps({
            "categories": [{"id": i, "name": n} for i, n in enumerate(classes)],
            "info": {"year": dt.date.today().year, "version": "1.0",
                     "contributor": "Phenology viewer Annotate tab"},
        }, indent=2) + "\n")
        done = sorted((r for r in self.images.values() if r["status"] == "done"), key=lambda r: r["image"])
        for split in ("train", "val"):
            atomic_write(os.path.join(self.root, f"{split}.txt"),
                         "".join(f"./{r['image']}\n" for r in done if r["split"] == split))
        atomic_write(os.path.join(self.root, "dataset.yaml"),
                     "# Ultralytics YOLO dataset, written by the phenology viewer's Annotate tab.\n"
                     f"#   yolo detect train data={self.root}/dataset.yaml model=yolo11n.pt\n"
                     "# images/ holds symlinks into raw_data: copy with `rsync -aL` to take the\n"
                     "# real files along. Update `path` if you move this folder.\n"
                     f"path: {json.dumps(self.root)}\n"
                     "train: train.txt\n"
                     "val: val.txt\n"
                     "names:\n" + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(classes)))


def clean_boxes(boxes, n_classes):
    """Validate boxes and clip them to the image."""
    if boxes is None:
        return []
    if not isinstance(boxes, list) or len(boxes) > MAX_BOXES:
        raise ValueError("boxes must be a list")
    out = []
    for b in boxes:
        if not isinstance(b, list) or len(b) != 5:
            raise ValueError("each box is [class, x_center, y_center, width, height]")
        cls = b[0]
        if not isinstance(cls, int) or not 0 <= cls < n_classes:
            raise ValueError(f"unknown class id {cls!r}")
        cx, cy, w, h = (float(v) for v in b[1:])
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            raise ValueError("box coordinates must be numbers")
        x0, x1 = max(0.0, cx - abs(w) / 2), min(1.0, cx + abs(w) / 2)
        y0, y1 = max(0.0, cy - abs(h) / 2), min(1.0, cy + abs(h) / 2)
        if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
            continue
        out.append([cls, round((x0 + x1) / 2, 6), round((y0 + y1) / 2, 6), round(x1 - x0, 6), round(y1 - y0, 6)])
    return out
