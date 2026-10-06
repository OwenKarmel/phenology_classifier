# Phenology Classifier

Tools for the 2025 vineyard trailcam and mobile phone photo data: a viewer that lines
the images up with phenology dates, an annotation tool for labeling grape clusters, and
the pipeline that trains a YOLO26 grape-cluster detector on those labels.

The viewer and annotation tool are tabs of one local web app in
[`visualization/`](visualization/), whose third tab, **Model**, shows the live log of a
training run. The longer, detailed guide is
[`visualization/README.md`](visualization/README.md). Training is in [`yolo/`](yolo/):
see [YOLO Model Training](#yolo-model-training).

```bash
.venv/bin/pip install -r visualization/requirements.txt   # just Pillow
.venv/bin/python visualization/server.py                  # http://127.0.0.1:8000/
```

Options: `--port`, `--host 0.0.0.0` (open it from another machine; anyone who can reach
it can also edit the annotations), `--rebuild` (rescan `raw_data` after adding images),
`--annotations DIR` (default `annotated_images/`), `--model-logs DIR` (training logs for
the Model tab, default `yolo/logs/`). `raw_data/` is only ever read.

The data comes from Google Drive: `python download_drive.py <DRIVE_FOLDER_OR_FILE_ID>`
downloads a folder into `raw_data/` (needs `credentials.json`; `--verify` re-checks it).

## Viewer Tool

Side-by-side viewer for the **Across** trailcam frames and the phone photos, with
phenology timelines (onset of bloom, bloom, fruit set) for every field.

- **Matching:** the trailcam frame (left) is paired with the phone photo (right) by
  same date and nearest capture time, and the time difference is shown. Stepping
  frames brings up the phone photos for that date.
- **Timelines:** an all-fields overview and a per-field timeline with stage bands,
  phone visits and trailcam coverage. Click one to jump to a field, camera or date.
- **Zoom:** scroll or pinch to zoom, drag to pan, `1:1` for actual pixels, `f` for full
  screen. The original file loads once the preview runs out of resolution.
- **Keys:** `←/→` frame, `Shift+←/→` same time on the next/previous day, `, .` phone
  photo, `[ ]` phone visit. The URL hash keeps the view, so links reopen it.
- **Show all data** (unticked by default): only `Across` views from Jun 11 (onset of
  prebloom) to the end of Jul 15, 2025 (last phone visit). Tick it for every view
  (`Flower`, `onpost`, …) and the whole season.
- **How the data is read:** trailcam time comes from the filename, phone time from EXIF
  (EDT). Phone photos without EXIF use their folder's date and are flagged "time
  unknown". Phenology dates come from `Pheno_2025_from_mobilephone.xlsx` (typed
  day-month; red text means approximate). Details are in the long README.

## Annotation Tool

The **Annotate** tab draws bounding boxes on trailcam frames, in the style of Label
Studio, and saves them as a YOLO dataset in `annotated_images/`. The **Annotate** link
above a trailcam frame in the Viewer opens that frame.

- **Draw** by dragging. Click a box to select it, then move it or resize it with its
  handles. Zoom, pan and full screen work as in the Viewer (`Space` or `H` to pan).
- **`A` / `D`** (or `←/→`) step to the previous / next frame and keep the zoom.
  `Shift+A/D` jump to the previous / next unlabeled frame, and a filter limits stepping
  to unlabeled, labeled or skipped frames. A status bar under the photo shows every
  frame and jumps on click.
- **Saving is automatic.** A frame with boxes joins the dataset as soon as you draw
  them. A frame with no clusters is only added when you **Submit** (`Ctrl+Enter`), as a
  negative example with an empty label file. **Skip** (`Ctrl+Space`) leaves a frame out;
  **Reset** makes it unlabeled again.
- **Shortcuts** follow Label Studio: `Ctrl+Z` undo, `Backspace` delete, `1`–`9` pick a
  label, `Ctrl+D` duplicate, `Ctrl+C/V` copy/paste boxes, `Ctrl+H` hide. `C` copies the
  previous labeled frame's boxes, since the cameras don't move. The full list is in the
  side panel.
- **Labels:** `classes.txt` currently has `grape_cluster` (0) and `AI_grape_cluster` (1).
  "+ Add label" adds a class and double-clicking a label renames it. Class ids are line
  numbers, so only remove a class from the end, and only if no box uses it.
- **Show all data** works as in the Viewer, with its own separate checkbox.

### `annotated_images/`

```
annotated_images/
  classes.txt          class names, one per line (line n = class id n)
  notes.json           the same classes, Label Studio style
  dataset.yaml         Ultralytics config: path, train.txt, val.txt, names
  train.txt, val.txt   ./images/<name>.jpg, one per line
  images/<name>.jpg    symlink to the original frame in raw_data (not a copy)
  labels/<name>.txt    "class x_center y_center width height" per box, 0–1
  annotations.json     per frame: source, field, camera, view, capture time, size,
                       status, boxes, split, last edit
```

Each camera folder and day goes entirely to train or val (about 20% val), so
near-identical frames are never split.

```bash
yolo detect train data=annotated_images/dataset.yaml model=yolo11n.pt imgsz=1280
```

Frames are 9216×5184 and clusters in `Across` are small, so train at a large `imgsz` or
on tiles. To copy the dataset to another machine, use `rsync -aL` (copies the real
images instead of the links) and fix `path:` in `dataset.yaml`.

## YOLO Model Training

[`yolo/`](yolo/) trains a YOLO26 grape-cluster detector on the boxes from the Annotate
tab. It follows sections 3–6 of `Train_YOLO_Models.ipynb` (install Ultralytics, write
`data.yaml`, train, test), with leave-one-camera-out cross-validation and a held-out
test camera. One human-readable file, [`yolo/config.yaml`](yolo/config.yaml), sets the
split, what goes into each `data.yaml`, and every training, evaluation and inference
parameter. The viewer's **Model** tab shows the live log of any run started with the
commands below.

### Commands

Run everything from the project root, preferably inside `tmux` (`tmux new -s train`),
since a full run takes about 4 hours.

```bash
# Section 3, once: virtualenv on /data with PyTorch (CUDA 12.6) + Ultralytics 8.4.173,
# linked as yolo/.venv; pulls yolo26l.pt (and yolo26n.pt, used by Ultralytics' AMP check)
bash yolo/setup_env.sh

# Checks
yolo/.venv/bin/python -m pytest yolo/tests              # unit tests, no GPU, ~5 s
yolo/.venv/bin/python yolo/pipeline.py train --smoke    # whole pipeline in ~2 min (yolo26n, 640 px, 2 epochs)

# Section 4: build the datasets and data.yaml files and print the split (train does this too)
yolo/.venv/bin/python yolo/pipeline.py prepare

# Sections 5 + 6: CV folds -> final model -> test camera -> summary (~4 h on one GPU)
yolo/.venv/bin/python yolo/pipeline.py train

# The same stages one at a time
yolo/.venv/bin/python yolo/pipeline.py cv                        # all folds
yolo/.venv/bin/python yolo/pipeline.py cv --fold LR/Camera1      # one fold
yolo/.venv/bin/python yolo/pipeline.py final                     # final model on every CV camera
yolo/.venv/bin/python yolo/pipeline.py test                      # score it on the test camera, save predictions
yolo/.venv/bin/python yolo/pipeline.py summary                   # CV and test scores

# Inference with the final model: files, folders (searched recursively) or globs
yolo/.venv/bin/python yolo/pipeline.py predict raw_data/Raw_Data/2025/CAT/Camera1/Across
yolo/.venv/bin/python yolo/pipeline.py predict "raw_data/Raw_Data/2025/*/Camera1/Across/2025070*_133001.jpg" --out /data/owen/preds
yolo/.venv/bin/python yolo/pipeline.py predict some/folder --weights /data/owen/phenology_classifier/yolo/runs/yolo26l_2560/cv/LR_Camera1/weights/last.pt

# Time 3 epochs of the largest fold and estimate the full run (redo after adding annotations)
yolo/.venv/bin/python yolo/pipeline.py bench
yolo/.venv/bin/python yolo/pipeline.py bench --set model.imgsz=3200 train.batch=1
```

Every command takes `--config FILE` (default `yolo/config.yaml`) and `--set KEY=VALUE ...`
to override config values for one run, e.g. `--set experiment=more_epochs train.epochs=150`.

- **Stopping and resuming:** `Ctrl+C` stops a run cleanly. Running the same command again
  resumes it: finished stages (those with a `done.json`) are skipped, and an interrupted
  stage continues from its last epoch's `weights/last.pt`. This also covers the machine
  being switched off. Closing the terminal does not stop a run; `kill <pid>` does (the
  pid is in the Model tab).
- **An experiment is frozen once training starts:** if the annotations, split, classes,
  model or training settings change, the pipeline refuses to mix them into the same
  experiment. Use a new `experiment:` name (the old results stay), or `--fresh` to delete
  the experiment and start over. Only `train.device`, `workers`, `cache`, `plots` and
  `batch` may change between runs of one experiment.
- **Model tab:** `bench`, `train`, `cv`, `final`, `test` and `predict` write
  `yolo/logs/<time>_<command>_<experiment>.log` (plus a `.json` status), and the Model tab
  follows the newest one live; older runs are in its drop-down. Restart the viewer once to
  load the tab: in `tmux attach -t viewer`, press `Ctrl+C`, then run
  `.venv/bin/python visualization/server.py --host 0.0.0.0 --port 8000` again.
  `--no-log` runs a command without a log.

### The split and how models are scored

- **Cameras** are written `FIELD/CameraN` (as in the Annotate tab) and cover every year.
  The config lists them explicitly: `split.test` is `CAC/Camera1`, and
  `split.cross_validation` is `CAC/Camera2`, `FRC/Camera1` and `LR/Camera1`. Nothing is
  included by default. An annotated camera in neither list is left out, and `prepare`
  names it. To add a camera, append it to one of the lists (with a new experiment name).
- **Frames:** only frames with status "done" in `annotated_images/annotations.json`, in
  the `split.views` views (`Across`). Frames submitted with no boxes are kept as negatives;
  skipped frames are left out. Current counts: test 87 frames, CV 213 frames (43 + 81 + 89).
- **Classes:** `data.class_map` maps annotation labels to the model's classes
  (`data.names`). Both `grape_cluster` and `AI_grape_cluster` currently map to a single
  class, `grape_cluster`. A label missing from the map stops `prepare`, so a new label is
  never silently dropped. Map a label to `null` to drop its boxes.
- **Leave-one-camera-out CV:** one fold per CV camera. It trains on the other CV cameras
  and validates every epoch on the held-out one. The **final model** then trains on all
  CV cameras with the same settings and is scored once on the test camera. The test
  camera is never trained on or used to choose anything.
- **Scores:** precision, recall, mAP50 and mAP50-95 from Ultralytics `val`. A fold is
  scored with `last.pt`, the same way the final model is chosen (it has no validation
  set, so its `last.pt` is the model). `best.pt` scores are shown in brackets: that
  checkpoint is picked on the held-out camera, so its score is optimistic. `summary`
  shows each fold, the mean ± sd over folds, the epoch with the best validation mAP, and
  the test score.
- **Image size:** frames are 9216×5184 and clusters are small, so frames are shrunk once
  to `model.imgsz` (long side) with area interpolation and cached. Left alone,
  Ultralytics would shrink them with `INTER_LINEAR`, which aliases at this scale.
  `predict` shrinks new frames the same way. Its labels are normalized, so they apply to
  the full-size frame too.

### Parameters chosen

The constraint was about 4 hours for the whole `train` run (3 CV folds + final model +
test) on one RTX 5000 Ada (32 GB), maximizing resolution and epochs. `bench` timings on
the largest fold (170 training frames):

| Model | `imgsz` | Batch that fits | s / epoch | Epochs in 4 h |
|---|---|---|---|---|
| **yolo26l** | **2560** | **2** (25 GB) | **39** | **92** |
| yolo26l | 3200 | 1 (2 runs out of memory) | 70 | 51 |
| yolo26x | 2560 | 1 | 62 | 58 |
| yolo26m | 3200 | 1 | 53 | 68 |

**Chosen: `yolo26l.pt`, `imgsz: 2560`, `batch: 2`, `epochs: 90`, on GPU 1, about
3.9 hours.**

- 2560 px is the largest size at which yolo26l still trains with a batch of 2 (it fills
  about 30 GB). At 2560 px a median cluster is 36–70 px and the smallest is about 19 px,
  well within what YOLO's stride-8 head detects. 3200 px only runs at batch 1 and halves
  the epochs that fit.
- With 124–213 training frames, Ultralytics takes only about 3 optimizer steps per epoch
  (nominal batch 64). Epochs are therefore the scarce resource, which favours about 90
  epochs at 2560 over about 50 at 3200, or over the larger yolo26x.
- `patience: 0` (no early stopping) gives every fold and the final model exactly the same
  schedule. `optimizer: auto` picks AdamW at lr 0.002 for runs this short. Augmentation
  is Ultralytics' default (mosaic, closed for the last 10 epochs), listed in the config.
- `cache: ram` keeps the shrunken frames in memory (about 3 GB). `amp: true` is on.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to avoid running out of GPU
  memory through fragmentation. If a batch of 2 still runs out in a fold's first epoch,
  Ultralytics drops to 1 and carries on.
- Inference (`predict`): `conf: 0.25`, `iou: 0.7` (NMS), batch 4, at the same 2560 px.
  Scoring uses `conf: 0.001` for the full precision-recall curve.

To spend the 4 hours differently, change `train.epochs` or `model.imgsz` / `train.batch`
under a new `experiment:` name, and check with `bench` first. For example, `final` alone takes about 1.05 h at 90 epochs, so
giving only the final model 4 hours would allow about 340 epochs.

### Where things are

```
yolo/                     code (git)
  config.yaml             split, data.yaml contents, every train / val / predict setting
  pipeline.py             the command line above
  config.py               loads and checks config.yaml, --set overrides
  splits.py               datasets and data.yaml files from annotated_images/ (section 4)
  engine.py               Ultralytics training, resuming, scoring, prediction (sections 5–6)
  runlog.py               logs each command to yolo/logs/ for the Model tab
  setup_env.sh            section 3: virtualenv + YOLO26 weights
  requirements.txt
  tests/                  pytest suite
  logs/                   run logs + status (git-ignored)
  .venv -> /data/owen/phenology_classifier/yolo/venv       (git-ignored)
visualization/server.py   /api/model/runs and /api/model/log (read only)
visualization/static/     model.js (the Model tab), plus small additions to
                          index.html, app.js and style.css

/data/owen/phenology_classifier/yolo/       outputs (/home is full)
  venv/  weights/  ultralytics/             env, yolo26*.pt, Ultralytics settings (analytics off)
  image_cache/<imgsz>/                      frames shrunk to imgsz, shared by experiments
  runs/<experiment>/
    config.yaml                             the config as run
    data/  cameras/<FIELD>_<Camera>/{images,labels}, cv_<FIELD>_<Camera>.yaml,
           final.yaml (test: = test camera), split.json
    cv/<FIELD>_<Camera>/                    weights/{last,best}.pt, results.csv/png, plots,
                                            eval_last/, eval_best/, done.json
    final/weights/last.pt                   THE MODEL
    test/  eval/ (scores, PR curves, sample batches), done.json,
           predict/ (images/ with boxes, labels/ with confidences, detections.csv)
    predict/<time>/                         `predict` outputs
    summary.json
  bench/<experiment>/                       `bench` runs (never touch an experiment)
```
