"""Load and check yolo/config.yaml (see the comments in that file).

    cfg = load("yolo/config.yaml", ["train.epochs=5"])

Paths come back absolute, plus derived ones: cfg["run_dir"] (this
experiment's outputs), cfg["weights_path"] (the starting checkpoint) and
cfg["image_cache"] (frames shrunk to model.imgsz, shared by experiments).
"""
import copy
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DEFAULT_CONFIG = os.path.join(HERE, "config.yaml")

CAMERA_RE = re.compile(r"^[^/\s]+/[^/\s]+$")  # FIELD/CameraN
# Set by the pipeline for every run; refused in the train / evaluate / predict sections.
RESERVED = {"data", "imgsz", "model", "project", "name", "exist_ok", "resume", "source", "split"}
SECTIONS = ("paths", "split", "data", "model", "train", "evaluate", "predict")


class ConfigError(ValueError):
    pass


def apply_override(cfg, item):
    """`a.b.c=value` -> cfg["a"]["b"]["c"] = yaml-parsed value."""
    key, sep, value = item.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"--set expects key=value, got {item!r}")
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = yaml.safe_load(value) if value.strip() else None


def load(path=DEFAULT_CONFIG, overrides=()):
    path = os.path.abspath(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    raw = copy.deepcopy(cfg)
    for item in overrides:
        apply_override(cfg, item)
        apply_override(raw, item)
    check(cfg)

    def resolve(p):
        return os.path.normpath(os.path.join(PROJECT, os.path.expanduser(str(p))))

    for k in ("annotations", "raw_data", "output"):
        cfg["paths"][k] = resolve(cfg["paths"][k])
    out = cfg["paths"]["output"]
    cfg["config_path"] = path
    cfg["raw"] = raw  # as written (with overrides), saved next to each run
    cfg["run_dir"] = os.path.join(out, "runs", cfg["experiment"])
    cfg["weights_dir"] = os.path.join(out, "weights")
    w = os.path.expanduser(str(cfg["model"]["weights"]))
    cfg["weights_path"] = w if os.path.isabs(w) or os.sep in w else os.path.join(cfg["weights_dir"], w)
    cfg["image_cache"] = os.path.join(out, "image_cache", str(cfg["model"]["imgsz"]))
    return cfg


def check(cfg):
    if not isinstance(cfg, dict):
        raise ConfigError("the config must be a mapping")
    exp = cfg.get("experiment")
    if not isinstance(exp, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", exp):
        raise ConfigError("experiment must be a name made of letters, digits, _ . -")
    for s in SECTIONS:
        cfg.setdefault(s, {})
        if cfg[s] is None:
            cfg[s] = {}
        if not isinstance(cfg[s], dict):
            raise ConfigError(f"{s} must be a mapping")
    unknown = set(cfg) - set(SECTIONS) - {"experiment"}
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")

    for k in ("annotations", "raw_data", "output"):
        if not cfg["paths"].get(k):
            raise ConfigError(f"paths.{k} is required")

    # Split: cameras are listed explicitly; nothing is included by default.
    sp = cfg["split"]
    for k in ("test", "cross_validation"):
        v = sp.get(k)
        if v is None:
            sp[k] = []
        elif not isinstance(v, list) or not all(isinstance(c, str) and CAMERA_RE.match(c) for c in v):
            raise ConfigError(f"split.{k} must be a list of cameras written FIELD/CameraN (e.g. CAC/Camera1)")
        if len(set(sp[k])) != len(sp[k]):
            raise ConfigError(f"split.{k} lists a camera twice")
    both = set(sp["test"]) & set(sp["cross_validation"])
    if both:
        raise ConfigError(f"camera(s) in both test and cross_validation: {', '.join(sorted(both))}")
    if not sp["cross_validation"]:
        raise ConfigError("split.cross_validation must list at least one camera")
    views = sp.get("views")
    if views is not None and (not isinstance(views, list) or not all(isinstance(v, str) for v in views)):
        raise ConfigError("split.views must be a list of view names (or left out for every view)")

    # data.yaml classes
    names = cfg["data"].get("names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names) \
            or len(set(names)) != len(names):
        raise ConfigError("data.names must be a list of unique class names")
    cmap = cfg["data"].get("class_map")
    if cmap is None:
        cfg["data"]["class_map"] = cmap = {n: n for n in names}
    if not isinstance(cmap, dict):
        raise ConfigError("data.class_map must map annotation labels to data.names (or null)")
    for src, dst in cmap.items():
        if dst is not None and dst not in names:
            raise ConfigError(f"data.class_map: {src} -> {dst}, but {dst!r} is not in data.names")

    m = cfg["model"]
    if not m.get("weights"):
        raise ConfigError("model.weights is required (e.g. yolo26l.pt)")
    if not isinstance(m.get("imgsz"), int) or m["imgsz"] < 32 or m["imgsz"] % 32:
        raise ConfigError("model.imgsz must be a whole number of pixels divisible by 32")

    for s in ("train", "evaluate", "predict"):
        bad = RESERVED & set(cfg[s])
        if bad:
            raise ConfigError(f"{s}: {', '.join(sorted(bad))} is set by the pipeline (use model.imgsz / model.weights)")
    if "val" in cfg["train"]:
        raise ConfigError("train.val is set by the pipeline: on for CV folds, off for the final model")


def dump(cfg):
    """The config as written (plus --set overrides), for saving next to a run."""
    return yaml.safe_dump(cfg["raw"], sort_keys=False)
