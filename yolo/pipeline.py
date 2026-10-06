"""Grape-cluster YOLO26 detector: data split, leave-one-camera-out CV, final
model, test and inference, all set by yolo/config.yaml.

    yolo/.venv/bin/python yolo/pipeline.py <command> [--config FILE] [--set key=value ...] [--smoke]

Commands (sections of Train_YOLO_Models.ipynb in brackets):
  pull      download the YOLO26 checkpoint(s) named in the config        [3]
  prepare   build the datasets and data.yaml files from annotated_images [4]
  bench     time a few epochs of one fold and estimate how long `train` takes
  train     CV folds, then the final model, then the test set           [5, 6]
  cv        only the CV folds (--fold CAM to pick some)                  [5]
  final     only the final model (trained on every CV camera)            [5]
  test      score the final model on the test cameras, save predictions  [6]
  predict   run the final model (or --weights) on images or folders      [6]
  summary   print the CV and test scores

bench, train, cv, final, test and predict are logged to yolo/logs/, which the
viewer's Model tab shows live. Stopping (Ctrl+C) and running `train` again
resumes: finished stages are skipped and an interrupted one continues from its
last checkpoint. --fresh starts the experiment over.
"""
import argparse
import datetime as dt
import os
import shutil
import sys
import time

import yaml

# Less fragmentation: at 2560 px a batch of 2 uses most of the 32 GB. Set before CUDA starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import config as config_mod  # noqa: E402
import engine  # noqa: E402
import runlog  # noqa: E402
import splits  # noqa: E402
from config import ConfigError  # noqa: E402

LOGGED = {"bench", "train", "cv", "final", "test", "predict"}
# --smoke: a few minutes end to end, in a separate experiment.
SMOKE = ["model.weights=yolo26n.pt", "model.imgsz=640", "train.epochs=2", "train.batch=8",
         "train.close_mosaic=1", "evaluate.batch=8", "predict.batch=8"]
# May differ between invocations of one experiment (they don't change the result).
RUNTIME_KEYS = {"device", "workers", "cache", "plots", "batch"}


def log(*a):
    print(*a, flush=True)


def banner(text):
    log(f"\n{'=' * 78}\n{text}  [{dt.datetime.now():%Y-%m-%d %H:%M:%S}]\n{'=' * 78}")


# ------------------------------------------------------------- experiment

def started(cfg):
    run = cfg["run_dir"]
    return [d for d in (os.path.join(run, "cv"), os.path.join(run, "final")) if os.path.isdir(d) and os.listdir(d)]


def check_same_settings(cfg):
    """An experiment's folds must all be trained the same way: refuse to carry
    on one that was started with different settings."""
    path = os.path.join(cfg["run_dir"], "config.yaml")
    if not os.path.exists(path) or not started(cfg):
        return
    with open(path) as fh:
        old = yaml.safe_load(fh) or {}
    new = cfg["raw"]
    diffs = []
    for sec in ("split", "data", "model", "train"):
        a, b = old.get(sec) or {}, new.get(sec) or {}
        for k in sorted(set(a) | set(b)):
            if sec == "train" and k in RUNTIME_KEYS:
                continue
            if a.get(k) != b.get(k):
                diffs.append(f"{sec}.{k}: {a.get(k)!r} -> {b.get(k)!r}")
    if diffs:
        raise ConfigError(f"experiment {cfg['experiment']} was started with other settings ("
                          + "; ".join(diffs) + "). Use a new experiment name, or --fresh to start it over.")


def ensure_data(cfg):
    """This experiment's datasets: built now, or the existing ones if the
    annotations and split haven't changed. Once training has started they are
    frozen, so every fold sees the same data."""
    split = splits.read_split(cfg)
    if split and split["fingerprint"] == splits.current_fingerprint(cfg):
        log(f"Datasets: {os.path.dirname(split['final'])} (prepared {split['created']}, up to date)")
        return split
    if split and started(cfg):
        raise ConfigError(f"the annotations or split changed since experiment {cfg['experiment']} started "
                          "training. Use a new experiment name, or --fresh to start it over.")
    return splits.prepare(cfg, log)


def save_config(cfg):
    os.makedirs(cfg["run_dir"], exist_ok=True)
    with open(os.path.join(cfg["run_dir"], "config.yaml"), "w") as fh:
        fh.write(f"# {cfg['experiment']}: config as run ({dt.datetime.now():%Y-%m-%d %H:%M}) from {cfg['config_path']}\n")
        fh.write(config_mod.dump(cfg))


def need_weights(cfg):
    if not os.path.exists(cfg["weights_path"]):
        log(f"{cfg['weights_path']} not found: downloading")
        engine.pull(cfg, [os.path.basename(cfg["weights_path"])], log)


# ----------------------------------------------------------------- stages

def run_cv(cfg, split, only=None):
    if not split["folds"]:
        log("Cross-validation needs at least two cameras in split.cross_validation: skipped")
        return
    for cam in only or split["folds"]:
        if cam not in split["folds"]:
            raise ConfigError(f"{cam} is not a CV camera ({', '.join(split['folds'])})")
        out = os.path.join(cfg["run_dir"], "cv", splits.slug(cam))
        done = engine.read_json(os.path.join(out, "done.json"))
        if done:
            log(f"Fold {cam}: done ({engine.fmt_scores(done['last'])})")
            continue
        others = [c for c in split["cross_validation"] if c != cam]
        n_train = sum(split["cameras"][c]["frames"] for c in others)
        n_val = split["cameras"][cam]["frames"]
        banner(f"CV fold {cam}: train on {', '.join(others)} ({n_train} frames), validate on {cam} ({n_val} frames)")
        t0 = time.time()
        last = engine.train(cfg, split["folds"][cam], out, val=True, log=log)
        minutes = (time.time() - t0) / 60
        best = os.path.join(out, "weights", "best.pt")
        res = {"camera": cam, "train_cameras": others, "train_frames": n_train, "val_frames": n_val,
               "weights": last, "train_minutes": round(minutes, 1),
               "last": engine.evaluate(cfg, last, split["folds"][cam], "val", out, "eval_last"),
               "best": engine.evaluate(cfg, best, split["folds"][cam], "val", out, "eval_best"),
               "best_epoch": engine.best_epoch(engine.epoch_stats(out)),
               "epochs": len(engine.epoch_stats(out))}
        engine.write_json(os.path.join(out, "done.json"), res)
        log(f"Fold {cam} ({minutes:.0f} min): last.pt {engine.fmt_scores(res['last'])}")
        log(f"{'':>{len(cam) + 14}}best.pt {engine.fmt_scores(res['best'])} (epoch {res['best_epoch']})")


def run_final(cfg, split):
    out = os.path.join(cfg["run_dir"], "final")
    done = engine.read_json(os.path.join(out, "done.json"))
    if done:
        log(f"Final model: done ({done['weights']})")
        return done
    n = sum(split["cameras"][c]["frames"] for c in split["cross_validation"])
    banner(f"Final model: train on every CV camera ({', '.join(split['cross_validation'])}, {n} frames)")
    t0 = time.time()
    last = engine.train(cfg, split["final"], out, val=False, log=log)
    rows = engine.epoch_stats(out)
    done = {"weights": last, "train_frames": n, "train_minutes": round((time.time() - t0) / 60, 1),
            "epochs": len(rows),
            "train_fit": {k: rows[-1].get(f"metrics/{k}(B)") for k in ("precision", "recall", "mAP50", "mAP50-95")}
            if rows else None}
    engine.write_json(os.path.join(out, "done.json"), done)
    log(f"Final model: {last} ({done['train_minutes']:.0f} min)")
    return done


def run_test(cfg, split):
    if not split["test"]:
        log("No test cameras in split.test: nothing to test")
        return None
    final = engine.read_json(os.path.join(cfg["run_dir"], "final", "done.json"))
    if not final:
        raise ConfigError("the final model isn't trained yet: run `train` or `final` first")
    out = os.path.join(cfg["run_dir"], "test")
    banner(f"Test: final model on {', '.join(split['test'])}")
    shutil.rmtree(out, ignore_errors=True)
    s = engine.evaluate(cfg, final["weights"], split["final"], "test", out, "eval")
    log(f"Test scores: {engine.fmt_scores(s)}")
    sources = [os.path.join(cfg["paths"]["raw_data"], p) for c in split["test"] for p in split["cameras"][c]["images"]]
    pred = engine.predict(cfg, final["weights"], sources, os.path.join(out, "predict"), log)
    done = {"weights": final["weights"], "cameras": split["test"],
            "frames": sum(split["cameras"][c]["frames"] for c in split["test"]),
            "scores": s, "eval_dir": os.path.join(out, "eval"), "predict_dir": pred["out_dir"],
            "detections": pred["detections"]}
    engine.write_json(os.path.join(out, "done.json"), done)
    return done


def run_bench(cfg, split, epochs, fold):
    """Train `epochs` epochs of one fold and extrapolate to the whole `train` run."""
    import torch

    folds = split["folds"] or {}
    if not folds:
        raise ConfigError("bench times a CV fold: list at least two cameras in split.cross_validation")
    frames = lambda cams: sum(split["cameras"][c]["frames"] for c in cams)  # noqa: E731
    cv = split["cross_validation"]
    cam = fold or max(folds, key=lambda c: frames([x for x in cv if x != c]))
    out = os.path.join(cfg["run_dir"], dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    n_train, n_val = frames([c for c in cv if c != cam]), frames([cam])
    banner(f"Bench: {epochs} epochs of fold {cam} ({n_train} train / {n_val} val frames), "
           f"{os.path.basename(cfg['weights_path'])} at {cfg['model']['imgsz']} px, batch {cfg['train'].get('batch')}")
    seen = {"batch": None, "peak": 0.0, "alloc": 0.0}

    def epoch_end(trainer):  # GPU memory of each epoch after the first (which may retry after running out)
        if trainer.epoch >= 1:
            seen["peak"] = max(seen["peak"], torch.cuda.max_memory_reserved(trainer.device) / 2**30)
            seen["alloc"] = max(seen["alloc"], torch.cuda.max_memory_allocated(trainer.device) / 2**30)
        seen["batch"] = trainer.batch_size
        seen["total"] = torch.cuda.get_device_properties(trainer.device).total_memory / 2**30
        torch.cuda.reset_peak_memory_stats(trainer.device)

    t0 = time.time()
    engine.train(cfg, folds[cam], out, val=True, log=log, callbacks={"on_fit_epoch_end": epoch_end},
                 extra={"epochs": epochs, "close_mosaic": 0, "plots": False})
    wall = time.time() - t0
    peak, batch = seen["peak"], seen["batch"]
    if batch != cfg["train"].get("batch"):
        log(f"\nNote: batch {cfg['train'].get('batch')} ran out of GPU memory; Ultralytics trained with batch {batch}. "
            f"Set train.batch: {batch} in the config.")
    rows = engine.epoch_stats(out)
    ends = [r["time"] for r in rows]
    per_epoch = [b - a for a, b in zip(ends, ends[1:])] or ends  # leave out epoch 1 (warm-up)
    epoch_s = sorted(per_epoch)[len(per_epoch) // 2]
    t1 = time.time()
    engine.evaluate(cfg, os.path.join(out, "weights", "last.pt"), folds[cam], "val", out, "bench_val")
    val_s = time.time() - t1
    train_img = max(epoch_s - val_s, 0) / n_train  # s per training frame per epoch
    val_img = val_s / n_val
    setup_s = max(wall - ends[-1], 0) if ends else 0  # model load, caching, final validation

    E = int(cfg["train"].get("epochs", 100))
    log(f"\nMeasured: {epoch_s:.0f} s per epoch ({train_img:.2f} s per training frame, "
        f"{val_img:.2f} s per validation frame), {setup_s:.0f} s set-up; batch {batch}, "
        f"GPU memory peak {seen['alloc']:.1f} GiB allocated, {peak:.1f} GiB reserved "
        f"of {seen.get('total', 0):.1f}")
    # Each stage: E epochs (per_epoch) plus a fixed part (set-up, final scoring).
    stages = [(f"fold {c}", frames([x for x in cv if x != c]), frames([c])) for c in folds]
    stages.append(("final model", frames(cv), 0))
    per_epoch = sum(nt * train_img + nv * val_img for _, nt, nv in stages)
    fixed = len(stages) * setup_s + sum(2 * nv * val_img for _, _, nv in stages) + frames(cv) * val_img \
        + 2 * frames(split["test"]) * val_img  # fold scoring (last + best), final's own check, test
    total = E * per_epoch + fixed
    log(f"\nEstimate for `train` with {E} epochs:")
    for name, nt, nv in stages:
        log(f"  {name:<22}{nt:>5} train /{nv:>4} val frames  {(E * (nt * train_img + nv * val_img) + setup_s) / 3600:5.2f} h")
    log(f"  total, with scoring and the test set           {total / 3600:5.2f} h")
    fit = int(max(4 * 3600 - fixed, 0) / per_epoch)
    log(f"  epochs that fit in 4 h: about {fit}")
    engine.write_json(os.path.join(out, "bench.json"), {
        "fold": cam, "epochs": epochs, "imgsz": cfg["model"]["imgsz"], "weights": cfg["weights_path"],
        "batch": batch, "epoch_s": epoch_s, "train_s_per_frame": train_img,
        "val_s_per_frame": val_img, "setup_s": setup_s, "peak_gib": peak, "peak_alloc_gib": seen["alloc"], "estimate_h": total / 3600,
        "epochs_in_4h": fit})


# ------------------------------------------------------------------- main

def parse(argv):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=config_mod.DEFAULT_CONFIG, help="config file (default: yolo/config.yaml)")
    common.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.epochs=10 experiment=test1")
    common.add_argument("--smoke", action="store_true",
                        help="quick end-to-end check: yolo26n, 640 px, 2 epochs, experiment <name>_smoke "
                             "(train, cv and final start it over)")
    common.add_argument("--no-log", action="store_true", help="don't write yolo/logs/ (Model tab)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[common])
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")
    sub.add_parser("pull", parents=[common], help="download YOLO26 checkpoints").add_argument(
        "names", nargs="*", help="checkpoints, e.g. yolo26x.pt (default: model.weights and yolo26n.pt)")
    p = sub.add_parser("prepare", parents=[common], help="build datasets and data.yaml files")
    p.add_argument("--force", action="store_true", help="rebuild even though training has started")
    p = sub.add_parser("bench", parents=[common], help="time a few epochs, estimate `train`")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--fold", help="CV camera to time (default: the fold with the most training frames)")
    for name, text in (("train", "CV folds, final model, test"), ("cv", "CV folds only"),
                       ("final", "final model only")):
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument("--fresh", action="store_true", help="delete this experiment's results and start over")
        if name == "cv":
            p.add_argument("--fold", nargs="+", metavar="CAM", help="only these folds, e.g. LR/Camera1")
    sub.add_parser("test", parents=[common], help="score the final model on the test cameras")
    p = sub.add_parser("predict", parents=[common], help="detect clusters in images")
    p.add_argument("source", nargs="+", help="image files, folders (searched recursively) or globs")
    p.add_argument("--weights", help="model to use (default: this experiment's final model)")
    p.add_argument("--out", help="output folder (default: <run_dir>/predict/<time>)")
    sub.add_parser("summary", parents=[common], help="print CV and test scores")
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse(argv)
    try:
        cfg = config_mod.load(args.config, (SMOKE if args.smoke else []) + args.set)
        if args.smoke:
            cfg = config_mod.load(args.config, SMOKE + args.set + [f"experiment={cfg['experiment']}_smoke"])
    except (ConfigError, OSError, yaml.YAMLError) as e:
        sys.exit(f"config: {e}")

    if args.command in LOGGED and not args.no_log and not os.environ.get(runlog.CHILD_ENV):
        sys.exit(runlog.run_logged(args.command, cfg["experiment"], argv, os.path.abspath(__file__)))

    try:
        if args.command == "prepare":
            if started(cfg) and not args.force:
                split = splits.read_split(cfg)
                if split and split["fingerprint"] != splits.current_fingerprint(cfg):
                    raise ConfigError(f"experiment {cfg['experiment']} has started training on the current datasets; "
                                      "use a new experiment name (or --force to rebuild them anyway)")
            splits.prepare(cfg, log)
            return 0
        if args.command in ("summary", "test"):
            split = splits.read_split(cfg)  # the datasets the experiment was trained on
            if not split:
                raise ConfigError(f"experiment {cfg['experiment']} has no datasets yet: run `train` first")
            if args.command == "summary":
                engine.summarize(cfg, split, log)
                return 0
        engine.init(cfg)
        if args.command == "pull":
            engine.pull(cfg, args.names or None, log)
            return 0
        if args.command == "predict":
            weights = args.weights or os.path.join(cfg["run_dir"], "final", "weights", "last.pt")
            if not os.path.exists(weights):
                raise ConfigError(f"{weights} doesn't exist: train the final model or pass --weights")
            out = args.out or os.path.join(cfg["run_dir"], "predict", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
            engine.predict(cfg, weights, args.source, out, log)
            return 0

        t0 = time.time()
        if args.command == "bench":
            # Its own folder, so a bench never touches an experiment's data or results.
            cfg["run_dir"] = os.path.join(cfg["paths"]["output"], "bench", cfg["experiment"])
            need_weights(cfg)
            run_bench(cfg, splits.prepare(cfg, log), args.epochs, args.fold)
            return 0
        log(f"Experiment {cfg['experiment']}: {cfg['run_dir']}")
        if args.command != "test":
            if (args.fresh or args.smoke) and os.path.exists(cfg["run_dir"]):  # a smoke run always starts over
                log(f"Starting over: deleting {cfg['run_dir']}")
                shutil.rmtree(cfg["run_dir"])
            check_same_settings(cfg)
            need_weights(cfg)
            split = ensure_data(cfg)
            save_config(cfg)
        if args.command in ("train", "cv"):
            run_cv(cfg, split, getattr(args, "fold", None))
        if args.command in ("train", "final"):
            run_final(cfg, split)
        if args.command in ("train", "test"):
            run_test(cfg, split)
        engine.summarize(cfg, split, log)
        log(f"\n{args.command} took {(time.time() - t0) / 3600:.2f} h")
        return 0
    except ConfigError as e:
        log(f"Error: {e}")
        return 2
    except KeyboardInterrupt:
        log("\nStopped. Run the same command again to resume.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
