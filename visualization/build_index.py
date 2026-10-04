"""Scan raw_data (read-only) and write data/index.json for the viewer.

Collects:
  * Across trailcam frames per field/camera (timestamp from the filename,
    which matches the time burned into the image).
  * Mobile phone photos with their EXIF capture time (falls back to the
    folder date when EXIF has been stripped) and, where the folder or file
    name says so, the trailcam they were taken at.
  * Phenology dates from Pheno_2025_from_mobilephone.xlsx.
  * Every trailcam folder of timestamped frames (Across, Flower, onpost, ...)
    for the Annotate tab.

Usage: python3 build_index.py
"""
import datetime as dt
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "raw_data"))
TRAIL_BASE = os.path.join(RAW, "Raw_Data")
TRAIL_ROOT = os.path.join(TRAIL_BASE, "2025")
PHONE_ROOT = os.path.join(RAW, "Mobile_Phone_Images_2025", "Phone_images")
PHENO_XLSX = os.path.join(RAW, "Pheno_2025_from_mobilephone.xlsx")
OUT = os.path.join(HERE, "data", "index.json")

YEAR = 2025
TS_RE = re.compile(r"^(\d{8}_\d{6})\.jpe?g$", re.I)
IMG_EXT = (".jpg", ".jpeg", ".png")

# Phone folder names that differ from the trailcam field folder names.
PHONE_FIELD_ALIASES = {"VG Big": "VG_Big"}


# ---------------------------------------------------------------- trailcams

def scan_trailcams():
    fields = {}
    for field in sorted(os.listdir(TRAIL_ROOT)):
        fdir = os.path.join(TRAIL_ROOT, field)
        if not os.path.isdir(fdir):
            continue
        cams = []
        # Most fields: <field>/CameraN/Across. Phenology_blocks: <field>/Across.
        candidates = []
        if os.path.isdir(os.path.join(fdir, "Across")):
            candidates.append(("Camera1", os.path.join(fdir, "Across")))
        for sub in sorted(os.listdir(fdir), key=natural_key):
            adir = os.path.join(fdir, sub, "Across")
            if re.match(r"camera\s*\d+$", sub, re.I) and os.path.isdir(adir):
                candidates.append((sub, adir))
        for cam, adir in candidates:
            frames = sorted(m.group(1) for f in os.listdir(adir)
                            if (m := TS_RE.match(f)))
            if frames:
                cams.append({
                    "id": cam,
                    "label": re.sub(r"(?i)camera\s*", "Camera ", cam),
                    "dir": os.path.relpath(adir, RAW),
                    "frames": frames,
                })
        fields[field] = cams
    return fields


# Views in the order the Annotate tab lists them; frames left directly in a
# camera folder (view "") go last.
VIEW_ORDER = {"across": 0, "flower": 1, "onpost": 2}


def trailcam_folder_info(rel_dir):
    """(year, field, camera, view) of a folder Raw_Data/<year>/<field>/[CameraN/][<view>].

    Fields with a single camera have no CameraN folder; they count as Camera1.
    """
    parts = rel_dir.replace(os.sep, "/").split("/")[1:]
    year, field, rest = parts[0], parts[1], parts[2:]
    cam = "Camera1"
    if rest and re.match(r"(?i)camera\s*\d+$", rest[0]):
        cam, rest = rest[0], rest[1:]
    return year, field, cam, "/".join(rest)


def scan_trailcam_views():
    """Every folder of timestamped trailcam frames, of every year and view."""
    out = []
    if not os.path.isdir(TRAIL_BASE):
        return out
    for dp, dns, fns in os.walk(TRAIL_BASE):
        dns.sort(key=natural_key)
        rel = os.path.relpath(dp, RAW)
        names = sorted(f for f in fns if TS_RE.match(f))
        parts = rel.split(os.sep)
        if not names or len(parts) < 3 or not re.match(r"^\d{4}$", parts[1]):
            continue
        year, field, cam, view = trailcam_folder_info(rel)
        out.append({
            "year": year,
            "field": field,
            "camera": cam,
            "cameraLabel": re.sub(r"(?i)camera\s*", "Camera ", cam),
            "view": view,
            "dir": rel.replace(os.sep, "/"),
            # ".jpg" is implied; any other extension is kept.
            "frames": [f[:-4] if f.endswith(".jpg") else f for f in names],
        })
    out.sort(key=lambda s: (s["year"], s["field"].lower(), natural_key(s["camera"]),
                            VIEW_ORDER.get(s["view"].lower(), 9 if s["view"] else 10),
                            s["view"].lower()))
    return out


# ------------------------------------------------------------------- phone

def exif_info(path):
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
            taken = sub.get(36867) or ex.get(306)
            return {
                "taken": taken.replace(":", "-", 2) if taken else None,
                "offset": sub.get(36881),
                "model": (ex.get(272) or "").strip() or None,
            }
    except Exception as e:  # unreadable image: still list it
        print("warn: cannot read", path, e, file=sys.stderr)
        return {"taken": None, "offset": None, "model": None}


def folder_date(parts):
    """Date from a folder named like 6-27, 7-18-25 or 6-24-25."""
    for p in reversed(parts):
        m = re.match(r"^(\d{1,2})-(\d{1,2})(?:-(\d{2,4}))?$", p)
        if m:
            return dt.date(YEAR, int(m.group(1)), int(m.group(2))).isoformat()
    return None


def phone_camera_tag(parts, fname):
    """Return (field_override, camera_id) from folder/file naming conventions."""
    field = None
    cam = None
    for p in parts:
        m = re.match(r"(?i)^camera\s*(\d+)$", p)
        if m:
            cam = int(m.group(1))
        m = re.match(r"(?i)^TB(martin|rail\s*road|niag(?:a)?ra)(\d+)$", p)
        if m:
            cam = int(m.group(2))
            if m.group(1).lower().startswith("niag"):
                field = "TB_Niagara"
            else:
                field = None  # keep the containing field
    m = re.match(r"(?i)^camera\s*(\d+)", fname)
    if m:
        cam = int(m.group(1))
    return field, (f"Camera{cam}" if cam else None)


def scan_phone():
    photos = []
    for dp, dns, fns in os.walk(PHONE_ROOT):
        dns.sort(key=natural_key)
        rel_dir = os.path.relpath(dp, PHONE_ROOT)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if not parts:
            continue
        field = PHONE_FIELD_ALIASES.get(parts[0], parts[0])
        for f in sorted(fns, key=natural_key):
            if not f.lower().endswith(IMG_EXT):
                continue
            path = os.path.join(dp, f)
            info = exif_info(path)
            override, cam = phone_camera_tag(parts[1:], f)
            taken = info.get("taken")
            if taken:
                date, time = taken.split(" ")
                date_source = "exif"
            else:
                date = folder_date(parts[1:])
                time = None
                date_source = "folder" if date else None
            photos.append({
                "field": override or field,
                "path": os.path.relpath(path, RAW),
                "name": f,
                "folder": "/".join(parts),
                "date": date,
                "time": time,
                "dateSource": date_source,
                "camera": cam,
                "model": info.get("model"),
                "offset": info.get("offset"),
            })
    return photos


# --------------------------------------------------------------- phenology

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_pheno():
    """Parse the phenology sheet with the stdlib (no openpyxl needed).

    Date cells were typed day-month (e.g. 11-6-2025 = 11 June) but Excel read
    some as month-day dates (6 Nov), and a few have a mistyped year. Those cells
    are converted back by swapping day/month and forcing the season year.
    Red text marks an approximate date in the sheet.
    """
    z = zipfile.ZipFile(PHENO_XLSX)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    styles = ET.fromstring(z.read("xl/styles.xml"))
    fonts = styles.find("m:fonts", NS).findall("m:font", NS)
    red_fonts = set()
    for i, font in enumerate(fonts):
        c = font.find("m:color", NS)
        if c is not None and (c.get("rgb") or "").upper().endswith("FF0000"):
            red_fonts.add(i)
    xfs = styles.find("m:cellXfs", NS).findall("m:xf", NS)
    red_styles = {i for i, xf in enumerate(xfs) if int(xf.get("fontId", 0)) in red_fonts}

    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.find("m:sheetData", NS).findall("m:row", NS):
        cells = {}
        for c in row.findall("m:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", NS)
            raw = v.text if v is not None else None
            if c.get("t") == "s" and raw is not None:
                val = shared[int(raw)]
            elif raw is not None:
                val = float(raw)
            else:
                val = None
            cells[col] = (val, int(c.get("s", 0)) in red_styles)
        rows.append(cells)

    def to_date(cell):
        if not cell or cell[0] in (None, "", "-"):
            return None
        val, red = cell
        if isinstance(val, float):  # Excel serial read as month-day
            d = dt.date(1899, 12, 30) + dt.timedelta(days=int(val))
            day, month = d.month, d.day
            if month > 12:  # was genuinely month-day already
                day, month = d.day, d.month
            out = dt.date(YEAR, month, day)
        else:
            m = re.match(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\s*$", str(val))
            if not m:
                return None
            out = dt.date(YEAR, int(m.group(2)), int(m.group(1)))
        return {"date": out.isoformat(), "approx": red}

    header = rows[0]
    note = None
    pheno = {}
    for cells in rows[1:]:
        name = cells.get("A", (None,))[0]
        if not name:
            continue
        pheno[str(name).strip()] = {
            "onset": to_date(cells.get("B")),
            "bloom": to_date(cells.get("C")),
            "fruitset": to_date(cells.get("D")),
        }
        for col in ("E", "F", "G"):
            if cells.get(col) and isinstance(cells[col][0], str):
                note = cells[col][0]
    labels = {k: header.get(c, ("",))[0] for k, c in
              (("onset", "B"), ("bloom", "C"), ("fruitset", "D"))}
    return pheno, labels, note


# ----------------------------------------------------------------- helpers

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main():
    trail = scan_trailcams()
    views = scan_trailcam_views()
    phone = scan_phone()
    pheno, pheno_labels, pheno_note = read_pheno()

    names = sorted(set(trail) | {p["field"] for p in phone} | set(pheno), key=str.lower)
    fields = []
    for name in names:
        fields.append({
            "id": name,
            "cameras": trail.get(name, []),
            "pheno": pheno.get(name),
            "photos": [p for p in phone if p["field"] == name],
        })

    all_frames = [f for fl in fields for c in fl["cameras"] for f in c["frames"]]
    all_dates = [p["date"] for p in phone if p["date"]]
    out = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "year": YEAR,
        "range": {
            "start": min([f"{a[:4]}-{a[4:6]}-{a[6:8]}" for a in all_frames] + all_dates),
            "end": max([f"{a[:4]}-{a[4:6]}-{a[6:8]}" for a in all_frames] + all_dates),
        },
        "phenoLabels": pheno_labels,
        "phenoNote": pheno_note,
        "fields": fields,
        "trailcams": views,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    n_frames = len(all_frames)
    n_cams = sum(len(f["cameras"]) for f in fields)
    print(f"wrote {OUT}: {len(fields)} fields, {n_cams} Across cameras, "
          f"{n_frames} frames, {len(phone)} phone photos "
          f"({sum(1 for p in phone if p['time'] is None)} without capture time), "
          f"{len(views)} trailcam folders for annotation "
          f"({sum(len(v['frames']) for v in views)} frames)")


if __name__ == "__main__":
    main()
