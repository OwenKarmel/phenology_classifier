# Phenology Image Viewer

Side-by-side viewer for the 2025 **Across** trailcam frames and the mobile phone
photos, with phenology timelines (onset of bloom, bloom, fruit set) for every field.
The **Annotate** tab draws grape-cluster bounding boxes on trailcam frames and
saves them as a YOLO dataset in `annotated_images/`. The **Model** tab shows the
live log of a YOLO training run (see "YOLO Model Training" in the project README).

```bash
# from the project root
.venv/bin/pip install -r visualization/requirements.txt   # just Pillow
.venv/bin/python visualization/server.py                  # http://127.0.0.1:8000/
```

Options: `--port 8000`, `--host 0.0.0.0` (to open it from another machine; anyone
who can reach it can also edit the annotations), `--rebuild` (rescan `raw_data`
after adding images), `--annotations DIR` (default `annotated_images` in the
project root) and `--model-logs DIR` (training logs for the Model tab, default
`yolo/logs`). You can also rebuild the index on its own with
`python visualization/build_index.py`.

`raw_data` is only read. Image previews are generated on demand and cached in
`visualization/cache/` (safe to delete at any time).

## Using it

From the top: the photos, the phone photo strip, the field timeline, then the
all-fields timeline.

- **Show all data** (unticked by default): only `Across` views and only dates
  from Jun 11 (onset of prebloom) to the end of Jul 15, 2025 (the last phone
  photo visit) are shown: frames, phone photos, visits and the timelines'
  date range. Tick it for every view (`Across`, `Flower`, …) and the whole
  season; the camera buttons then list each camera's views. A link to a frame
  or photo outside that range turns it on.
- **Viewer**: the trailcam frame (left) is matched to the phone photo (right)
  by **same date and nearest capture time**, and the time difference is shown.
  Stepping trailcam frames brings up the phone photos for that date.
- **Field timeline**: full stage labels, phone visits, and one row per trailcam
  camera (view). Click a camera row or use the buttons to choose the camera.
- **All fields**: one timeline per field showing the stage bands, phone visits
  (dots) and trailcam coverage (grey line). Click a row to open that field at that date.
- **Zoom**: scroll or pinch to zoom, drag to pan, double-click to zoom or fit,
  `1:1` for actual pixels, and the corner-brackets button (or `f`) for full
  screen. When the preview runs out of resolution, the original file is loaded.
  Trailcam zoom stays put while you step through frames. "Full size ↗" opens
  the original file.
- Keys: `←/→` frame, `Shift+←/→` same time on the next/previous day, `, .` phone
  photo, `[ ]` phone visit.
- The URL hash keeps the current view, so a link reopens the same frame and photo.

## Annotating grape clusters

The **Annotate** tab works like Label Studio's image bounding-box editor, on
every trailcam folder (`Across`, `Flower`, `onpost`, …), not just `Across`.
Pick the field, camera and view at the top. By default only `Across` views from
Jun 11 (onset of prebloom) to the end of Jul 15, 2025 (the last phone photo
visit) are offered; tick **Show all data** for every view and date (this box is
separate from the Viewer's). Opening a frame outside that range by link turns
it on. The field and all-fields timelines sit under the photo and follow the
frame being annotated, in the same date range; click them to jump to a camera,
field or date. The **Annotate** link above a
trailcam frame in the Viewer opens that frame.

- **Draw** by dragging on the photo. Click a box to select it, then drag it or
  its handles. The drawing tool shows a crosshair to line up edges. Zoom, pan and
  full screen work as in the Viewer: scroll to zoom; hold `Space`, middle-drag, or
  switch to the pan tool (`H`) to pan. The original file loads when you zoom past
  the preview's resolution.
- **`A` / `D`** (or `←` / `→`) go to the previous / next frame, keeping the zoom.
  "A / D steps through" limits stepping to unlabeled, labeled or skipped frames;
  `Shift+A` / `Shift+D` always jump to the previous / next unlabeled frame;
  `Shift+←` / `Shift+→` show the same time on the adjacent day. The bar under the
  photo shows every frame's status; click it to jump.
- **Saving is automatic.** A frame with boxes is in the dataset as soon as you
  draw them. A frame with **no clusters** is only added when you press
  **Submit** (`Ctrl+Enter`), which saves it as a negative example (an empty
  label file). **Skip** (`Ctrl+Space`) leaves a frame out (blurry, fogged,
  camera moved); its boxes are kept but not exported. **Reset** makes a frame
  unlabeled again. Removing the last box of a frame also makes it unlabeled.
- `C` copies the boxes of the nearest earlier labeled frame of the same camera
  view. The cameras don't move, so clusters stay roughly in place between frames.
  `Ctrl+C` / `Ctrl+V` copy the selected box (or all boxes) to any frame.
- Other keys follow Label Studio: `Ctrl+Z` / `Ctrl+Shift+Z` undo / redo (per
  frame, this session), `Backspace` delete the selected box, `Ctrl+Backspace`
  delete all, `1`–`9` choose a label or relabel the selected box, `Ctrl+D`
  duplicate, `Alt+.` select the next box, `Esc` or `U` unselect, `Ctrl+H` hide
  all boxes, `Alt+H` hide the selected one, `Shift+1` fit, `Shift+2` 100%, `F`
  full screen. The full list is under "Keyboard shortcuts" in the side panel.
- **Labels:** the one label is `grape_cluster` (class 0). "+ Add label" adds a
  class and double-clicking a label renames it. Class ids are line numbers in
  `classes.txt`, so delete a class only from the end of that file, and only if
  no box uses it.

### `annotated_images/`

Laid out like Label Studio's "YOLO" export, plus what Ultralytics needs to train:

```
annotated_images/
  classes.txt          class names, one per line (line n = class id n)
  notes.json           the same classes, Label Studio style
  dataset.yaml         Ultralytics config: path, train.txt, val.txt, names
  train.txt, val.txt   ./images/<name>.jpg, one per line
  images/<name>.jpg    symlink to the original frame in raw_data (not a copy)
  labels/<name>.txt    "class x_center y_center width height" per box, 0–1
  annotations.json     per frame: source path, field, camera, view, capture
                       time, size, status, boxes (YOLO), split, last edit
```

`<name>` is the frame's path, e.g.
`Raw_Data/2025/CAT/Camera1/Flower/20250707_133001.jpg` →
`2025_CAT_Camera1_Flower_20250707_133001`. Each camera folder and day goes
entirely to train or entirely to val (about 20% val), so near-identical frames
from one day are never split. The folder is rewritten on every save; edit
`annotations.json` only while the server is stopped.

```bash
yolo detect train data=annotated_images/dataset.yaml model=yolo11n.pt imgsz=1280
```

The trailcam frames are 9216×5184 and clusters in the `Across` view are small,
so train at a large `imgsz` or on tiles. Ultralytics needs at least one image in
`val.txt`. To copy the dataset to another machine, use `rsync -aL` (this copies
the real images instead of the links), then fix `path:` in `dataset.yaml`.

## How the data is interpreted

- **Trailcam time** comes from the filename (`YYYYMMDD_HHMMSS`), which matches
  the time stamped on the image. **Phone time** comes from EXIF `DateTimeOriginal`
  (EDT, −04:00). Both are local time.
- **Phone photos without EXIF** (82 photos: CAC 6-27, CAT 6-25, LR 6-24,
  VG Big 6-25 and others) take their date from the folder name. They are matched
  to the daytime frame nearest midday and flagged as "time unknown".
  **6 photos have no date at all** (`TB_Niagara/TBNiagra4/<number>.jpeg`,
  `Railroad/TBRailRoad2/image.jpg`). They are listed under "Undated" and are
  not matched to a trailcam frame.
- **Camera tags** come from naming: `Camera N…` file names, `Camera N` folders, and
  `TB<Field>N` folders (`TBMartin1` → Martin Camera 1). The folder
  `Railroad/TBRailRoad1/TBNIAGRA1` is treated as **TB_Niagara Camera 1**.
- **Phenology dates** come from `Pheno_2025_from_mobilephone.xlsx`. The dates
  were typed day-month, but Excel stored some of them as month-day, and a few
  have the wrong year (e.g. `11-6-2027`). These are all read as day-month 2025
  (onset = Jun 11, or Jun 12 for HC). Dates in red text in the sheet are shown
  as approximate (dashed marker, "~").
- Both tabs list every folder of timestamped frames once "Show all data" is
  ticked (`Flower`, `onpost`, `TB_Niagara/Camera3/New folder`, …), with frames
  loose in a camera folder as "Unsorted"; unticked, only `Across`. Frames between 21:00 and 05:00 count as night frames ("Skip night
  frames", "Daytime only").
