# Phenology Image Viewer

Side-by-side viewer for the 2025 **Across** trailcam frames and the mobile phone
photos, with phenology timelines (onset of bloom, bloom, fruit set) for every field.

```bash
# from the project root
.venv/bin/pip install -r visualization/requirements.txt   # just Pillow
.venv/bin/python visualization/server.py                  # http://127.0.0.1:8000/
```

Options: `--port 8000`, `--host 0.0.0.0` (to open it from another machine; anyone
who can reach it can also edit the annotations), `--rebuild` (rescan `raw_data`
after adding images) and `--annotations DIR` (default `annotated_images` in the
project root). You can also rebuild the index on its own with
`python visualization/build_index.py`.

`raw_data` is only read. Image previews are generated on demand and cached in
`visualization/cache/` (safe to delete at any time).

## Using it

- **All fields**: one timeline per field showing the stage bands, phone visits
  (dots) and trailcam coverage (grey line). Click a row to open that field at that date.
- **Field timeline**: full stage labels, phone visits, and one row per Across
  camera. Click a camera row or use the buttons to choose the camera.
- **Viewer**: the trailcam frame (left) is matched to the phone photo (right)
  by **same date and nearest capture time**, and the time difference is shown.
  Stepping trailcam frames brings up the phone photos for that date.
- **Zoom**: scroll or pinch to zoom, drag to pan, double-click to zoom or fit,
  `1:1` for actual pixels, and the corner-brackets button (or `f`) for full
  screen. When the preview runs out of resolution, the original file is loaded.
  Trailcam zoom stays put while you step through frames. "Full size ↗" opens
  the original file.
- Keys: `←/→` frame, `Shift+←/→` same time on the next/previous day, `, .` phone
  photo, `[ ]` phone visit.
- The URL hash keeps the current view, so a link reopens the same frame and photo.

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
- Only `Across` folders are used. `TB_Niagara/Camera3/New folder` and the
  `Flower`/`onpost` folders are not shown. Frames between 21:00 and 05:00 count
  as night frames ("Skip night frames").
