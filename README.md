# Phenology Classifier

Tools for the 2025 vineyard trailcam and mobile phone photo data: a viewer that lines
the images up with phenology dates, and an annotation tool for labeling grape clusters
to fine-tune a YOLO model.

Both tools are tabs of one local web app in [`visualization/`](visualization/). The
longer, detailed guide is [`visualization/README.md`](visualization/README.md).

```bash
.venv/bin/pip install -r visualization/requirements.txt   # just Pillow
.venv/bin/python visualization/server.py                  # http://127.0.0.1:8000/
```

Options: `--port`, `--host 0.0.0.0` (open it from another machine; anyone who can reach
it can also edit the annotations), `--rebuild` (rescan `raw_data` after adding images),
`--annotations DIR` (default `annotated_images/`). `raw_data/` is only ever read.

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
