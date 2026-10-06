"use strict";

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const STAGES = [
  { key: "onset", short: "Onset", label: "Onset of bloom", color: "var(--stage-onset)" },
  { key: "bloom", short: "Bloom", label: "Bloom", color: "var(--stage-bloom)" },
  { key: "fruitset", short: "Fruit set", label: "Fruit set", color: "var(--stage-fruit)" },
];
const FRUIT_TAIL_DAYS = 10; // fruit set has no end date; draw a fading tail
const DAY = 1440; // minutes

// All times are naive local (EDT) wall-clock; use UTC math so the browser's
// own time zone never shifts anything.
const isoDay = (iso) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10)) / 864e5;
const dayIso = (d) => new Date(d * 864e5).toISOString().slice(0, 10);
const fmtDay = (iso, weekday) => {
  const d = new Date(isoDay(iso) * 864e5);
  return (weekday ? WEEKDAYS[d.getUTCDay()] + " " : "") + MONTHS[d.getUTCMonth()] + " " + d.getUTCDate();
};
const fmtHM = (min) => {
  const m = ((Math.round(min) % DAY) + DAY) % DAY;
  return String(Math.floor(m / 60)).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0");
};
const fmtDur = (min) => {
  min = Math.round(Math.abs(min));
  if (min < 60) return `${min} min`;
  const d = Math.floor(min / DAY), h = Math.floor((min % DAY) / 60), m = min % 60;
  if (d) return `${d} d ${h} h`;
  return m ? `${h} h ${m} min` : `${h} h`;
};
const el = (tag, attrs = {}, text) => {
  const n = tag.startsWith("svg:") ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
};
const imgUrl = (path, w) => `/img?w=${w}&p=${encodeURIComponent(path)}`;
const rawUrl = (path) => `/raw?p=${encodeURIComponent(path)}`;

// ------------------------------------------------------------------- data
let IDX = null;
let FIELDS = [];
const S = { field: null, cam: null, fi: -1, pi: -1, date: null, match: null };
const opts = { skipNight: true, taggedOnly: false, showAll: false };

// With "Show all data" unticked (both tabs): Across views only, from onset of
// prebloom to the end of the day of the last phone photo visit.
const FOCUS = { view: "Across", start: "2025-06-11", end: "2025-07-15" };
const inFocus = (date) => !!date && date >= FOCUS.start && date <= FOCUS.end;
const focusText = () => `${FOCUS.view} views, ${fmtDay(FOCUS.start)} – ${fmtDay(FOCUS.end)}`;

// One camera view (folder) of a field. Across views keep the plain camera id
// ("Camera1"), so links and phone-photo tags still match; other views get
// "Camera1/Flower". Only frames on dates passing `keep` are included.
function makeCam(s, keep, withView) {
  const view = s.view || "Unsorted";
  const c = {
    id: s.view === FOCUS.view ? s.camera : `${s.camera}/${view}`,
    cam: s.camera, view, dir: s.dir, byDate: {}, frames: [],
    label: s.cameraLabel + (withView ? ` · ${view}` : ""),
  };
  for (const n of s.frames) {
    const file = n.includes(".") ? n : n + ".jpg";
    const date = `${file.slice(0, 4)}-${file.slice(4, 6)}-${file.slice(6, 8)}`;
    if (!keep(date)) continue;
    const hour = +file.slice(9, 11);
    (c.byDate[date] ||= []).push(c.frames.length);
    c.frames.push({ ts: file.replace(/\.[^.]+$/, ""), date, min: isoDay(date) * DAY + hour * 60 + +file.slice(11, 13),
      night: hour < 5 || hour >= 21, path: `${s.dir}/${file}` });
  }
  c.days = Object.keys(c.byDate).map(isoDay);
  return c;
}

const coverage = (cams) => [...new Set(cams.flatMap((c) => c.days))].sort((a, b) => a - b);

function prepare(idx) {
  const year = String(idx.year);
  const days = [];
  for (const f of idx.fields) {
    // Index from before the Annotate tab: only the Across cameras are listed.
    const srcs = idx.trailcams ? idx.trailcams.filter((s) => s.year === year && s.field === f.id)
      : f.cameras.map((c) => ({ camera: c.id, cameraLabel: c.label, view: FOCUS.view, dir: c.dir, frames: c.frames }));
    f.allCams = srcs.map((s) => makeCam(s, () => true, true));
    f.focusCams = srcs.filter((s) => s.view === FOCUS.view).map((s) => makeCam(s, inFocus, false)).filter((c) => c.frames.length);
    f.allCoverage = coverage(f.allCams);
    f.focusCoverage = coverage(f.focusCams);
    f.photos.forEach((p, i) => {
      p.i = i;
      p.min = p.date && p.time ? isoDay(p.date) * DAY + +p.time.slice(0, 2) * 60 + +p.time.slice(3, 5) : null;
      if (p.date) days.push(isoDay(p.date));
    });
    days.push(...f.allCoverage);
  }
  if (days.length) idx.range = { start: dayIso(Math.min(...days)), end: dayIso(Math.max(...days)) };
  scopeFields(idx.fields, opts.showAll);
  return idx.fields;
}

// The Viewer's cameras for its "Show all data" setting.
const camsOf = (f, all) => (all ? f.allCams : f.focusCams);
const visitsShown = (p, all) => all || inFocus(p.date);
function scopeFields(fields, all) {
  for (const f of fields) {
    f.cameras = camsOf(f, all);
    f.coverageDays = all ? f.allCoverage : f.focusCoverage;
  }
}

const field = () => S.field;
const cam = () => S.cam;
const frame = () => (S.cam && S.fi >= 0 ? S.cam.frames[S.fi] : null);
const photo = () => (S.pi >= 0 ? S.field.photos[S.pi] : null);
const camNum = (id) => (id || "").replace(/\D/g, "");

function photoVisible(p) {
  return visitsShown(p, opts.showAll) && (!opts.taggedOnly || (S.cam && p.camera === S.cam.cam));
}
function photosOn(date) {
  const list = S.field.photos.filter((p) => (date === "undated" ? !p.date : p.date === date) && photoVisible(p));
  return list.sort((a, b) => (a.min ?? 1e12) - (b.min ?? 1e12) || a.name.localeCompare(b.name, undefined, { numeric: true }));
}
function visitDates() {
  const set = new Set(S.field.photos.filter(photoVisible).map((p) => p.date || "undated"));
  const dated = [...set].filter((d) => d !== "undated").sort();
  return set.has("undated") ? [...dated, "undated"] : dated;
}

// Pick the frame of camera `c` that best matches a phone photo: same date,
// nearest capture time. Without a recorded time, use the daytime frame
// nearest midday. Falls back to the nearest frame on another date.
function matchFrame(p, c) {
  if (!c || !c.frames.length || !p.date) return null;
  const known = p.min !== null;
  const target = known ? p.min : isoDay(p.date) * DAY + 12 * 60;
  let cands = c.byDate[p.date] || [];
  const sameDay = cands.length > 0;
  if (!sameDay) cands = c.frames.map((_, i) => i);
  if (!known || opts.skipNight) {
    const day = cands.filter((i) => !c.frames[i].night);
    if (day.length) cands = day;
  }
  let best = cands[0];
  for (const i of cands) if (Math.abs(c.frames[i].min - target) < Math.abs(c.frames[best].min - target)) best = i;
  return { fi: best, sameDay, known, delta: c.frames[best].min - target };
}

function nearestFrame(c, targetMin, preferDay) {
  if (!c || !c.frames.length) return -1;
  let best = -1;
  c.frames.forEach((f, i) => {
    if (preferDay && f.night) return;
    if (best < 0 || Math.abs(f.min - targetMin) < Math.abs(c.frames[best].min - targetMin)) best = i;
  });
  return best < 0 ? nearestFrame(c, targetMin, false) : best;
}

// --------------------------------------------------------------- actions
function selectField(id, { camId, date } = {}) {
  const f = FIELDS.find((x) => x.id === id) || FIELDS[0];
  S.field = f;
  S.cam = f.cameras.find((c) => c.id === camId) || f.cameras[0] || null;
  S.fi = -1; S.pi = -1; S.match = null;
  if (!date) {
    const visits = visitDates().filter((d) => d !== "undated");
    date = visits.find((d) => S.cam && S.cam.byDate[d]) || visits[0] || (S.cam && S.cam.frames[0]?.date) || (opts.showAll ? IDX.range.start : FOCUS.start);
  }
  selectDate(date);
}

function selectCamera(id) {
  const prev = frame();
  S.cam = S.field.cameras.find((c) => c.id === id) || S.cam;
  const p = photo();
  if (p && !photoVisible(p)) return selectDate(S.date);
  if (p && p.date) {
    S.match = matchFrame(p, S.cam);
    S.fi = S.match ? S.match.fi : -1;
  } else {
    S.fi = prev ? nearestFrame(S.cam, prev.min, opts.skipNight) : (S.cam?.frames.length ? 0 : -1);
  }
  render();
}

function selectPhoto(pi) {
  const p = S.field.photos[pi];
  if (!p) return;
  S.pi = pi;
  S.date = p.date || "undated";
  if (p.date) {
    S.match = matchFrame(p, S.cam);
    S.fi = S.match ? S.match.fi : -1;
  } else {
    S.match = null; // undated: leave the trailcam where it is
  }
  render();
}

function selectDate(date) {
  const list = photosOn(date);
  if (list.length) {
    // Prefer a photo tagged to this camera, else the first of the visit.
    const tagged = list.find((p) => S.cam && p.camera === S.cam.cam);
    return selectPhoto((tagged || list[0]).i);
  }
  S.pi = -1; S.match = null; S.date = date;
  if (date !== "undated") {
    const idxs = S.cam?.byDate[date];
    const target = isoDay(date) * DAY + 13 * 60 + 30;
    S.fi = idxs ? idxs.reduce((b, i) => (!S.cam.frames[i].night && (S.cam.frames[b].night ||
      Math.abs(S.cam.frames[i].min - target) < Math.abs(S.cam.frames[b].min - target)) ? i : b), idxs[0])
      : nearestFrame(S.cam, target, true);
  }
  render();
}

function selectFrame(fi) {
  const c = S.cam;
  if (!c || fi < 0 || fi >= c.frames.length) return;
  S.fi = fi;
  const f = c.frames[fi];
  const p = photo();
  if (!p || p.date !== f.date) {
    const list = photosOn(f.date);
    if (list.length) {
      const tagged = list.filter((q) => q.camera === c.cam);
      const pool = tagged.length ? tagged : list;
      const best = pool.reduce((b, q) => (q.min !== null && (b.min === null || Math.abs(q.min - f.min) < Math.abs(b.min - f.min)) ? q : b), pool[0]);
      S.pi = best.i;
    } else {
      S.pi = -1;
    }
    S.date = f.date;
  }
  const cur = photo();
  S.match = cur && cur.date === f.date ? { fi, sameDay: true, known: cur.min !== null, delta: cur.min !== null ? f.min - cur.min : null, manual: true } : null;
  render();
}

function stepFrame(dir) {
  const c = S.cam;
  if (!c) return;
  let i = S.fi + dir;
  while (i >= 0 && i < c.frames.length && opts.skipNight && c.frames[i].night) i += dir;
  if (i >= 0 && i < c.frames.length) selectFrame(i);
}

function stepDay(dir) {
  const c = S.cam, f = frame();
  if (!c || !f) return;
  const tod = f.min % DAY;
  let d = isoDay(f.date);
  for (let n = 0; n < 200; n++) {
    d += dir;
    const idxs = c.byDate[dayIso(d)];
    if (idxs) {
      const best = idxs.reduce((b, i) => (Math.abs((c.frames[i].min % DAY) - tod) < Math.abs((c.frames[b].min % DAY) - tod) ? i : b), idxs[0]);
      return selectFrame(best);
    }
  }
}

function stepPhoto(dir) {
  const list = photosOn(S.date);
  if (!list.length) return;
  const k = list.findIndex((p) => p.i === S.pi);
  const next = list[Math.max(0, Math.min(list.length - 1, (k < 0 ? (dir > 0 ? -1 : list.length) : k) + dir))];
  if (next && next.i !== S.pi) selectPhoto(next.i);
}

function stepVisit(dir) {
  const v = visitDates();
  if (!v.length) return;
  const k = v.indexOf(S.date);
  if (k >= 0) {
    const n = v[k + dir];
    if (n) selectDate(n);
    return;
  }
  // Current date is not a visit: jump to the nearest visit in that direction.
  const cur = S.date && S.date !== "undated" ? S.date : null;
  const dated = v.filter((d) => d !== "undated");
  const n = dir > 0 ? dated.find((d) => !cur || d > cur) : [...dated].reverse().find((d) => !cur || d < cur);
  if (n) selectDate(n);
}

// ---------------------------------------------------------------- render
function render() {
  renderFieldHeader();
  renderTrail();
  renderPhone();
  renderStrip();
  renderCharts();
  writeHash();
  preload();
}

function stageOn(f, iso) {
  if (!f.pheno || !iso || iso === "undated") return null;
  let cur = null;
  for (const s of STAGES) {
    const v = f.pheno[s.key];
    if (v && iso >= v.date) cur = { ...s, ...v };
  }
  return cur;
}

function renderFieldHeader() {
  const sel = $("fieldSelect");
  if (!sel.options.length) {
    for (const f of FIELDS) sel.append(el("option", { value: f.id }, f.id.replace(/_/g, " ")));
  }
  sel.value = S.field.id;

  const picker = $("camPicker");
  picker.replaceChildren();
  if (!S.field.cameras.length) picker.append(el("span", { class: "muted" }, opts.showAll ? "No trailcam in this field" : `No trailcam frames in ${focusText()}`));
  for (const c of S.field.cameras) {
    const b = el("button", { role: "radio", "aria-checked": String(c === S.cam), title: `${c.frames.length} frames, ${fmtDay(c.frames[0].date)} – ${fmtDay(c.frames.at(-1).date)}` }, c.label);
    b.onclick = () => selectCamera(c.id);
    picker.append(b);
  }
}

const previewWidth = (box) => {
  const w = box.clientWidth * (window.devicePixelRatio || 1);
  return w <= 900 ? 800 : w <= 1700 ? 1600 : 2400;
};

function setImage(boxId, imgId, path, empty, keepZoom = false) {
  const box = $(boxId), img = $(imgId);
  const z = ZOOM[boxId];
  if (!path) {
    box.classList.add("is-empty");
    box.classList.remove("loading");
    img.removeAttribute("src");
    img.dataset.url = "";
    z.setPath(null, false);
    const e = box.querySelector(".empty");
    e.replaceChildren(...(Array.isArray(empty) ? empty : [document.createTextNode(empty || "")]));
    return;
  }
  box.classList.remove("is-empty");
  const w = previewWidth(box);
  const url = imgUrl(path, w);
  if (z.path !== path) {
    z.setPath(path, keepZoom, w);
    box.classList.add("loading");
    img.onload = () => { box.classList.remove("loading"); z.loaded(); };
    img.onerror = () => box.classList.remove("loading");
    img.src = url;
    img.dataset.url = url;
  }
}

// ------------------------------------------------------------------ zoom
// Pan/zoom inside an image box. The <img> keeps object-fit: contain and gets
// a CSS transform; once the preview can't supply enough pixels for the zoom
// level, the original full-resolution file is swapped in.
const ZOOM = {};
const MAX_ZOOM = 24;

class Zoomer {
  constructor(box, img) {
    this.box = box; this.img = img;
    this.s = 1; this.tx = 0; this.ty = 0;
    this.path = null; this.hires = false; this.previewW = 1600;
    this.pointers = new Map();
    const bar = el("div", { class: "zoombar" });
    const btn = (label, title, fn) => { const b = el("button", { type: "button", title }, label); b.onclick = (e) => { e.stopPropagation(); fn(); }; bar.append(b); return b; };
    btn("−", "Zoom out", () => this.zoomCenter(1 / 1.6));
    this.level = el("span", { class: "zlevel", title: "Zoom relative to fit" }, "1×");
    bar.append(this.level);
    btn("+", "Zoom in", () => this.zoomCenter(1.6));
    btn("Fit", "Fit to box (double-click)", () => this.reset());
    btn("1:1", "Actual pixels of the original photo", () => this.actualPixels());
    this.fsBtn = btn("", "Full screen (f)", () => this.toggleFullscreen());
    this.fsBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9"/></svg>';
    this.hiresTag = el("span", { class: "hires", hidden: "" }, "full res");
    bar.append(this.hiresTag);
    bar.addEventListener("pointerdown", (e) => e.stopPropagation());
    bar.addEventListener("dblclick", (e) => e.stopPropagation());
    box.append(bar);

    box.addEventListener("wheel", (e) => {
      if (box.classList.contains("is-empty")) return;
      e.preventDefault();
      const p = this.local(e);
      this.zoomAt(p.x, p.y, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
    }, { passive: false });
    box.addEventListener("dblclick", (e) => {
      if (box.classList.contains("is-empty")) return;
      const p = this.local(e);
      this.s > 1.01 ? this.reset() : this.zoomAt(p.x, p.y, 3);
    });
    box.addEventListener("mousedown", (e) => { if (e.button === 1) e.preventDefault(); }); // no autoscroll
    box.addEventListener("pointerdown", (e) => {
      // Left or middle button drags pan.
      if (box.classList.contains("is-empty") || e.button > 1) return;
      box.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, this.local(e));
      this.gesture = null;
    });
    box.addEventListener("pointermove", (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      const prev = this.pointers.get(e.pointerId);
      const cur = this.local(e);
      this.pointers.set(e.pointerId, cur);
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        if (this.gesture) {
          this.tx += mid.x - this.gesture.mid.x;
          this.ty += mid.y - this.gesture.mid.y;
          this.zoomAt(mid.x, mid.y, dist / this.gesture.dist);
        }
        this.gesture = { dist, mid };
      } else if (this.s > 1) {
        box.classList.add("dragging");
        this.tx += cur.x - prev.x;
        this.ty += cur.y - prev.y;
        this.apply();
      }
    });
    const up = (e) => {
      this.pointers.delete(e.pointerId);
      this.gesture = null;
      if (!this.pointers.size) box.classList.remove("dragging");
    };
    box.addEventListener("pointerup", up);
    box.addEventListener("pointercancel", up);
    box.addEventListener("keydown", (e) => {
      if (e.key === "+" || e.key === "=") this.zoomCenter(1.6);
      else if (e.key === "-" || e.key === "_") this.zoomCenter(1 / 1.6);
      else if (e.key === "0") this.reset();
      else if (e.key === "f") this.toggleFullscreen();
      else return;
      e.preventDefault();
      e.stopPropagation();
    });
    new ResizeObserver(() => this.apply()).observe(box);
  }

  local(e) {
    const r = this.box.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  // Where the image content sits inside the box at zoom 1 (object-fit: contain).
  fit() {
    const W = this.box.clientWidth, H = this.box.clientHeight;
    // Keep the last known size so the geometry holds while a new src loads.
    if (this.img.naturalWidth) { this.nw = this.img.naturalWidth; this.nh = this.img.naturalHeight; }
    const nw = this.nw || W, nh = this.nh || H;
    const r = Math.min(W / nw, H / nh);
    return { W, H, fw: nw * r, fh: nh * r, fx: (W - nw * r) / 2, fy: (H - nh * r) / 2 };
  }

  apply() {
    const { W, H, fw, fh, fx, fy } = this.fit();
    const s = this.s;
    if (fw * s <= W) this.tx = (W - fw * s) / 2 - fx * s;
    else this.tx = Math.min(-fx * s, Math.max(W - (fx + fw) * s, this.tx));
    if (fh * s <= H) this.ty = (H - fh * s) / 2 - fy * s;
    else this.ty = Math.min(-fy * s, Math.max(H - (fy + fh) * s, this.ty));
    this.img.style.transform = s === 1 ? "" : `translate(${this.tx}px, ${this.ty}px) scale(${s})`;
    this.box.classList.toggle("zoomed", s > 1.001);
    this.level.textContent = `${s < 10 ? (Math.round(s * 10) / 10) : Math.round(s)}×`;
    this.maybeHires(fw);
    this.onApply?.(); // the Annotate tab redraws its boxes
  }

  zoomAt(x, y, factor) {
    const ns = Math.min(MAX_ZOOM, Math.max(1, this.s * factor));
    this.tx = x - (x - this.tx) * (ns / this.s);
    this.ty = y - (y - this.ty) * (ns / this.s);
    this.s = ns;
    this.apply();
  }
  zoomCenter(factor) { this.zoomAt(this.box.clientWidth / 2, this.box.clientHeight / 2, factor); }
  reset() { this.s = 1; this.tx = 0; this.ty = 0; this.apply(); }

  actualPixels() {
    const fw = this.fit().fw;
    const full = this.img.dataset.fullWidth ? +this.img.dataset.fullWidth : null;
    if (!full) {
      // Load the original first to learn its pixel width, then zoom.
      this.loadHires(() => this.actualPixels());
      return;
    }
    this.zoomCenter(full / (window.devicePixelRatio || 1) / fw / this.s);
  }

  toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else this.box.requestFullscreen?.().then(() => this.box.focus());
  }

  // New image: optionally keep the current zoom/pan (trailcam frame stepping).
  setPath(path, keep, previewW) {
    this.path = path;
    this.hires = false;
    this.hiresTag.hidden = true;
    delete this.img.dataset.fullWidth;
    if (previewW) this.previewW = previewW;
    if (!keep) { this.s = 1; this.tx = 0; this.ty = 0; this.img.style.transform = ""; this.box.classList.remove("zoomed"); this.level.textContent = "1×"; }
  }
  loaded() { this.apply(); this.onLoad?.(); }

  maybeHires(fw) {
    if (!this.path || this.hires) return;
    if (this.s * fw * (window.devicePixelRatio || 1) > this.previewW * 1.05) this.loadHires();
  }

  loadHires(then) {
    if (!this.path) return;
    if (this.hires === "done") { then?.(); return; }
    this.hires = true;
    const path = this.path, url = rawUrl(path);
    this.box.classList.add("loading");
    const pre = new Image();
    pre.onload = () => {
      if (this.path !== path) return;
      this.hires = "done";
      this.box.classList.remove("loading");
      this.img.src = url;
      this.img.dataset.fullWidth = pre.naturalWidth;
      this.hiresTag.hidden = false;
      then?.();
    };
    pre.onerror = () => { if (this.path === path) { this.hires = false; this.box.classList.remove("loading"); } };
    pre.src = url;
  }
}

function renderTrail() {
  const c = S.cam, f = frame();
  $("trailCaption").textContent = c && f ? `${S.field.id.replace(/_/g, " ")} · ${c.label} · ${fmtDay(f.date, true)} ${fmtHM(f.min)}  —  ← → frame, Shift+← → day, Esc to exit` : "";
  const full = $("trailFull");
  $("trailAnnotate").toggleAttribute("disabled", !f);
  $("trailKind").textContent = `Trailcam · ${c ? c.view : opts.showAll ? "all views" : FOCUS.view}`;
  if (!c) {
    $("trailTitle").textContent = S.field.id.replace(/_/g, " ");
    setImage("trailBox", "trailImg", null, opts.showAll ? "This field has no trailcam images."
      : `This field has no trailcam frames in ${focusText()}. Tick "Show all data" to see the rest.`);
    full.setAttribute("aria-disabled", "true");
    $("trailNotice").hidden = true;
    $("timeChips").replaceChildren();
    return;
  }
  if (!f) {
    $("trailTitle").textContent = c.label;
    setImage("trailBox", "trailImg", null, "No frame selected.");
    full.setAttribute("aria-disabled", "true");
  } else {
    const st = stageOn(S.field, f.date);
    $("trailTitle").textContent = `${c.label} · ${fmtDay(f.date, true)} · ${fmtHM(f.min)}` + (st ? ` · ${st.label}` : "");
    const keep = ZOOM.trailBox.camId === c.id && ZOOM.trailBox.fieldId === S.field.id;
    ZOOM.trailBox.camId = c.id; ZOOM.trailBox.fieldId = S.field.id;
    setImage("trailBox", "trailImg", f.path, null, keep);
    full.href = rawUrl(f.path);
    full.removeAttribute("aria-disabled");
  }

  const notice = $("trailNotice");
  const p = photo();
  if (p && p.date && S.match && !S.match.sameDay && f) {
    notice.hidden = false;
    notice.className = "notice warn";
    notice.textContent = `${c.label} has no frame on ${fmtDay(p.date)} (it ran ${fmtDay(c.frames[0].date)} – ${fmtDay(c.frames.at(-1).date)}). Showing the nearest frame, ${fmtDur(S.match.delta)} ${S.match.delta < 0 ? "earlier" : "later"}.`;
  } else if (!p && S.date === "undated") {
    notice.hidden = true;
  } else {
    notice.hidden = true;
  }

  const chips = $("timeChips");
  chips.replaceChildren();
  const idxs = f ? c.byDate[f.date] || [] : [];
  for (const i of idxs) {
    const fr = c.frames[i];
    const b = el("button", { class: (i === S.fi ? "on " : "") + (fr.night ? "night" : ""), title: fr.night ? "Night frame" : "" }, fmtHM(fr.min));
    b.onclick = () => selectFrame(i);
    chips.append(b);
  }
  const nav = $("trailPane").querySelector(":scope > .nav");
  nav.querySelector('[data-act="frame-1"]').disabled = !f || S.fi <= 0;
  nav.querySelector('[data-act="frame+1"]').disabled = !f || S.fi >= c.frames.length - 1;
  nav.querySelector('[data-act="day-1"]').disabled = !f || f.date === c.frames[0].date;
  nav.querySelector('[data-act="day+1"]').disabled = !f || f.date === c.frames.at(-1).date;
}

function renderPhone() {
  const p = photo(), f = frame(), c = S.cam;
  $("phoneCaption").textContent = p ? `${S.field.id.replace(/_/g, " ")} · ${p.name} · ${p.date ? fmtDay(p.date, true) : "undated"} ${p.min !== null ? fmtHM(p.min) : ""}  —  , . photo, [ ] visit, Esc to exit` : "";
  const info = $("matchInfo");
  const full = $("phoneFull");
  info.className = "notice";
  info.replaceChildren();
  const visits = visitDates();
  const nav = $("phonePane").querySelector(":scope > .nav");
  const list = photosOn(S.date);
  const k = list.findIndex((q) => q.i === S.pi);
  $("photoCount").textContent = list.length ? `${k + 1} / ${list.length}` : "";
  nav.querySelector('[data-act="photo-1"]').disabled = k <= 0;
  nav.querySelector('[data-act="photo+1"]').disabled = k < 0 || k >= list.length - 1;
  const vk = visits.indexOf(S.date);
  const dated = visits.filter((d) => d !== "undated");
  nav.querySelector('[data-act="visit-1"]').disabled = vk === 0 || (vk < 0 && !dated.some((d) => S.date && d < S.date));
  nav.querySelector('[data-act="visit+1"]').disabled = vk === visits.length - 1 || (vk < 0 && !dated.some((d) => !S.date || d > S.date) && !visits.includes("undated"));

  if (!p) {
    $("phoneTitle").textContent = S.date && S.date !== "undated" ? fmtDay(S.date, true) : "—";
    full.setAttribute("aria-disabled", "true");
    const msg = [];
    const d = S.date && S.date !== "undated" ? S.date : null;
    msg.push(el("div", {}, S.field.photos.length
      ? (opts.taggedOnly ? `No phone photos tagged to ${c ? c.label : "this camera"} on ${d ? fmtDay(d) : "this date"}.` : `No phone photos on ${d ? fmtDay(d) : "this date"}.`)
      : "No phone photos for this field."));
    const prev = dated.filter((x) => d && x < d).at(-1), next = dated.find((x) => !d || x > d);
    const row = el("div", { class: "nav" });
    if (prev) { const b = el("button", {}, `‹ Visit ${fmtDay(prev)}`); b.onclick = () => selectDate(prev); row.append(b); }
    if (next) { const b = el("button", {}, `Visit ${fmtDay(next)} ›`); b.onclick = () => selectDate(next); row.append(b); }
    if (row.childNodes.length) msg.push(row);
    setImage("phoneBox", "phoneImg", null, msg);
    return;
  }

  $("phoneTitle").textContent = p.date ? `${fmtDay(p.date, true)} · ${p.min !== null ? fmtHM(p.min) : "time unknown"}` : "Undated photo";
  setImage("phoneBox", "phoneImg", p.path);
  full.href = rawUrl(p.path);
  full.removeAttribute("aria-disabled");

  const meta = [p.name];
  if (p.model) meta.push(p.model);
  if (p.camera) meta.push(`tagged ${p.camera.replace(/camera/i, "Camera ")} (folder/file name)`);
  info.append(el("div", {}, meta.join(" · ")));

  const line = el("div");
  if (!p.date) {
    info.classList.add("warn");
    line.textContent = "No date in the photo's metadata or folder name, so it can't be matched to a trailcam frame.";
  } else if (!c) {
    line.textContent = "No Across trailcam to match against.";
  } else if (S.match && f) {
    if (p.min === null) {
      info.classList.add("warn");
      line.textContent = `Capture time not recorded (metadata stripped); date from folder “${p.folder.split("/").at(-1)}”. Showing the ${fmtHM(f.min)} trailcam frame${S.match.sameDay ? " nearest midday" : ""}.`;
    } else {
      const d = f.min - p.min;
      line.append("Matched trailcam frame ", el("b", {}, `${fmtDay(f.date)} ${fmtHM(f.min)}`), " — ",
        el("b", {}, Math.abs(d) < 1 ? "same minute" : `${fmtDur(d)} ${d < 0 ? "before" : "after"}`), " the photo.");
      if (!S.match.sameDay) info.classList.add("warn");
    }
  }
  if (line.childNodes.length) info.append(line);

  const tagCam = p.camera && c && p.camera !== c.cam &&
    (S.field.cameras.find((x) => x.cam === p.camera && x.view === c.view) || S.field.cameras.find((x) => x.cam === p.camera));
  if (tagCam) {
    const b = el("button", {}, `Switch to ${tagCam.label}`);
    b.onclick = () => selectCamera(tagCam.id);
    const l2 = el("div", {}, `This photo was taken at ${p.camera.replace(/camera/i, "Camera ")}. `);
    l2.append(b);
    info.append(l2);
  }
}

function renderStrip() {
  const strip = $("strip");
  strip.replaceChildren();
  const list = photosOn(S.date);
  $("stripTitle").textContent = S.date === "undated" ? `Undated phone photos (${list.length})`
    : `Phone photos · ${S.date ? fmtDay(S.date, true) : ""}` + (list.length ? ` (${list.length})` : "");
  if (!list.length) strip.append(el("span", { class: "muted" }, "None on this date — pick a visit."));
  for (const p of list) {
    const b = el("button", { class: "thumb" + (p.i === S.pi ? " on" : ""), title: p.path });
    b.append(el("img", { src: imgUrl(p.path, 320), loading: "lazy", alt: p.name }));
    const cap = el("span");
    cap.append(`${p.min !== null ? fmtHM(p.min) : "--:--"} · ${p.name}`);
    b.append(cap);
    if (p.camera) b.append(el("span", { class: "tag" }, p.camera.replace(/camera/i, "Camera ")));
    b.onclick = () => selectPhoto(p.i);
    strip.append(b);
  }
  const on = strip.querySelector(".thumb.on");
  if (on) on.scrollIntoView({ block: "nearest", inline: "nearest" });

  const chips = $("visitChips");
  chips.replaceChildren();
  for (const d of visitDates()) {
    const n = photosOn(d).length;
    const b = el("button", { class: d === S.date ? "on" : "" }, `${d === "undated" ? "Undated" : fmtDay(d)} · ${n}`);
    b.onclick = () => selectDate(d);
    chips.append(b);
  }
}

// ---------------------------------------------------------------- charts
// The charts follow the Viewer, or the Annotate tab while it is open. A chart
// context says which field, camera and moment to mark, whether all data or only
// the focus range is shown, and what a click on the chart should open.
function chartCtx() {
  const a = TAB === "annotate" ? Annot.chartCtx() : null;
  if (a) return a;
  const f = frame();
  return {
    all: opts.showAll, field: S.field, cam: S.cam,
    at: f ? f.min : S.date && S.date !== "undated" ? (isoDay(S.date) + 0.5) * DAY : null,
    date: S.date,
    pick({ field: fl, cam, date }) {
      if (fl !== S.field) return selectField(fl.id, { date });
      if (cam && cam !== S.cam) {
        if (!date) return selectCamera(cam.id);
        S.cam = cam;
      }
      if (date) selectDate(date);
    },
  };
}

function renderCharts() {
  if (!S.field) return;
  const ctx = chartCtx();
  renderFieldName(ctx);
  renderOverview(ctx);
  renderFieldTimeline(ctx);
}

function renderFieldName(ctx) {
  const ph = ctx.field.pheno;
  const parts = [];
  for (const s of STAGES) {
    const v = ph && ph[s.key];
    if (v) parts.push(`${s.label} ${v.approx ? "~" : ""}${fmtDay(v.date)}`);
  }
  $("phenoSummary").textContent = parts.length ? parts.join(" · ") : "No phenology dates recorded";
  $("fieldName").textContent = ctx.field.id.replace(/_/g, " ");
  $("overviewScope").textContent = ctx.all ? "" : `${focusText()} · tick "Show all data" for the whole season`;
}

function scaleX(width, left, right, all) {
  const r = all ? IDX.range : FOCUS;
  const d0 = isoDay(r.start) - 1, d1 = isoDay(r.end) + 2;
  const k = (width - left - right) / (d1 - d0);
  return { x: (day) => left + (day - d0) * k, day: (x) => Math.floor((x - left) / k + d0), k, d0, d1, left, right: width - right };
}

function drawAxis(svg, sc, y, y2) {
  // Wide days (the focus range): label every day or every other day.
  const step = sc.k >= 34 ? 1 : sc.k >= 17 ? 2 : 0;
  for (let d = Math.ceil(sc.d0); d < sc.d1; d++) {
    const date = new Date(d * 864e5);
    const dom = date.getUTCDate();
    const weekly = dom === 1 || dom === 8 || dom === 15 || dom === 22;
    const label = step ? dom === 1 || ((dom - 1) % step === 0 && dom < 30) : weekly && (dom === 1 || sc.k * 7 > 34);
    if (!weekly && !label) continue;
    const x = sc.x(d);
    if (weekly) svg.append(el("svg:line", { x1: x, x2: x, y1: y + 6, y2, class: "grid" }));
    if (label) {
      svg.append(el("svg:text", { x: x + 3, y: y + 2, class: "axis-label", "font-weight": dom === 1 ? 600 : 400 },
        dom === 1 ? MONTHS[date.getUTCMonth()] : String(dom)));
    }
  }
}

function runs(days) {
  const out = [];
  for (const d of days) {
    const last = out.at(-1);
    if (last && d === last[1] + 1) last[1] = d; else out.push([d, d]);
  }
  return out;
}

// Stage bands: onset->bloom, bloom->fruit set, then a short fading fruit-set tail.
function drawStages(svg, sc, f, y, h, { labels, labelRows } = {}) {
  const ph = f.pheno;
  if (!ph || !STAGES.some((s) => ph[s.key])) return false;
  const pts = STAGES.map((s) => ph[s.key] && { ...s, ...ph[s.key], day: isoDay(ph[s.key].date) });
  let row = 0;
  for (const p of pts) if (p) p.row = row++;
  const labelY = (p) => y - labelRows + 11 + p.row * 15;
  pts.forEach((p, i) => {
    if (!p) return;
    const nextP = pts.slice(i + 1).find(Boolean);
    const end = nextP ? nextP.day : p.day + FRUIT_TAIL_DAYS;
    const x0 = sc.x(p.day), x1 = sc.x(end) - (nextP ? 2 : 0);
    const fillAttrs = nextP ? { fill: p.color, "fill-opacity": 0.28 } : { fill: `url(#fade-${p.key})` };
    svg.append(el("svg:rect", { x: x0, y, width: Math.max(2, x1 - x0), height: h, rx: 3, ...fillAttrs }));
    svg.append(el("svg:line", { x1: x0 + 1, x2: x0 + 1, y1: labels === "stair" ? labelY(p) - 11 : y - 2, y2: y + h + 2, stroke: p.color, "stroke-width": 2.5, "stroke-dasharray": p.approx ? "3 2" : null }));
    if (labels === "inside" && x1 - x0 > p.short.length * 6 + 8) {
      svg.append(el("svg:text", { x: x0 + 5, y: y + h / 2 + 4, class: "stage-label" }, p.short));
    }
  });
  if (labels === "stair") {
    for (const p of pts) {
      if (p) svg.append(el("svg:text", { x: sc.x(p.day) + 6, y: labelY(p), class: "stage-label big" }, `${p.label} · ${p.approx ? "~" : ""}${fmtDay(p.date)}`));
    }
  }
  return true;
}

// Gradients, plus a clip path so nothing is drawn left of the row labels.
function svgDefs(svg, id, sc, H) {
  const defs = el("svg:defs");
  for (const s of STAGES) {
    const g = el("svg:linearGradient", { id: `fade-${s.key}`, x1: 0, x2: 1, y1: 0, y2: 0 });
    g.append(el("svg:stop", { offset: 0, "stop-color": s.color, "stop-opacity": 0.32 }));
    g.append(el("svg:stop", { offset: 1, "stop-color": s.color, "stop-opacity": 0 }));
    defs.append(g);
  }
  const clip = el("svg:clipPath", { id: `${id}-plot` });
  clip.append(el("svg:rect", { x: sc.left, y: 0, width: sc.right - sc.left, height: H }));
  defs.append(clip);
  svg.append(defs);
  return `url(#${id}-plot)`;
}

function drawCursor(svg, sc, ctx, y0, y1) {
  if (ctx.at === null) return;
  const x = sc.x(ctx.at / DAY);
  svg.append(el("svg:line", { x1: x, x2: x, y1: y0, y2: y1, class: "cursor" }));
  svg.append(el("svg:path", { d: `M${x - 5},${y0 - 6} L${x + 5},${y0 - 6} L${x},${y0} Z`, class: "cursor-head" }));
}

// Annotate tab: frames as segments coloured by annotation status, like the
// frame bar under the photo. Frames come every 6 h, so each 6 h slot is one
// segment; where several cameras share a slot (the all-fields rows), it is
// split in proportion to their statuses.
const FRAME_STATUSES = ["labeled", "empty", "skipped", "unlabeled"];
const SLOT = 360; // minutes

function drawFrameStatus(g, sc, ctx, cams, y, h) {
  const slots = new Map();
  for (const c of cams) {
    for (const f of c.frames) {
      if (ctx.dayOnly && f.night) continue;
      const k = Math.floor(f.min / SLOT);
      if (!slots.has(k)) slots.set(k, { labeled: 0, empty: 0, skipped: 0, unlabeled: 0, n: 0 });
      const n = slots.get(k);
      n[ctx.status(f.path)]++;
      n.n++;
    }
  }
  const w = (sc.k * SLOT) / DAY, width = w >= 4 ? w - 1 : w; // a 1 px gap once there is room
  for (const [k, n] of slots) {
    let x = sc.x((k * SLOT) / DAY);
    for (const st of FRAME_STATUSES) {
      if (!n[st]) continue;
      const ww = (width * n[st]) / n.n;
      g.append(el("svg:rect", { x, y, width: ww, height: h, class: `fs fs-${st}` }));
      x += ww;
    }
  }
}

function visitsOf(f, all) {
  const m = new Map();
  for (const p of f.photos) if (p.date && visitsShown(p, all)) m.set(p.date, (m.get(p.date) || 0) + 1);
  return m;
}

function renderOverview(ctx) {
  const host = $("overview");
  const W = host.clientWidth || 800;
  const narrow = W < 640;
  const left = narrow ? 92 : 132, rowH = 30, top = 26;
  const H = top + FIELDS.length * rowH + 4;
  const sc = scaleX(W, left, 10, ctx.all);
  const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": "Phenology timelines for all fields" });
  const clip = svgDefs(svg, "ov", sc, H);
  drawAxis(svg, sc, 12, H);

  FIELDS.forEach((f, r) => {
    const y = top + r * rowH;
    const sel = f === ctx.field;
    if (sel) svg.append(el("svg:rect", { x: 0, y, width: W, height: rowH, class: "row-sel", rx: 6 }));
    svg.append(el("svg:rect", { x: 0, y, width: W, height: rowH, class: "row-hit", rx: 6, "data-row": r }));
    svg.append(el("svg:text", { x: 8, y: y + rowH / 2 + 4, class: "row-label" + (sel ? " sel" : ""), "pointer-events": "none" }, f.id.replace(/_/g, " ")));
    const g = el("svg:g", { "pointer-events": "none", "clip-path": clip });
    if (ctx.status) drawFrameStatus(g, sc, ctx, camsOf(f, ctx.all), y + rowH - 7, 4);
    else {
      for (const [a, b] of runs(ctx.all ? f.allCoverage : f.focusCoverage)) {
        g.append(el("svg:rect", { x: sc.x(a), y: y + rowH - 6, width: Math.max(1, sc.x(b + 1) - sc.x(a)), height: 3, rx: 1.5, class: "cov" + (sel ? " sel" : "") }));
      }
    }
    const has = drawStages(g, sc, f, y + 4, 14, { labels: "inside" });
    const visits = visitsOf(f, ctx.all);
    for (const [d] of visits) {
      g.append(el("svg:circle", { cx: sc.x(isoDay(d) + 0.5), cy: y + rowH - 4.5, r: 4.5, class: "visit" + (sel && d === ctx.date ? " sel" : "") }));
    }
    svg.append(g);
    const anyCam = camsOf(f, ctx.all).length;
    if (!has && !anyCam && !visits.size) {
      svg.append(el("svg:text", { x: left + 6, y: y + rowH / 2 + 4, class: "none", "pointer-events": "none" },
        ctx.all ? "No trailcam, phone photos or phenology dates" : "No Across frames, phone photos or phenology dates in this range"));
    } else if (!has && !narrow) {
      svg.append(el("svg:text", { x: left + 6, y: y + 17, class: "none", "pointer-events": "none" }, "No phenology dates"));
    }
  });
  const selRow = FIELDS.indexOf(ctx.field);
  drawCursor(svg, sc, ctx, top + selRow * rowH + 2, top + selRow * rowH + rowH - 2);

  svg.addEventListener("click", (e) => {
    const { x, row } = locate(e, svg, top, rowH, FIELDS.length);
    if (row < 0) return;
    const f = FIELDS[row];
    const day = sc.day(x);
    const date = x >= left && day >= sc.d0 ? nearestDateFor(f, day, ctx.all) : undefined;
    if (f !== ctx.field || date) ctx.pick({ field: f, date });
  });
  svg.addEventListener("mousemove", (e) => {
    const { x, row } = locate(e, svg, top, rowH, FIELDS.length);
    if (row < 0 || x < left) return hideTip();
    showTip(e, tipFor(FIELDS[row], dayIso(sc.day(x)), null, ctx.all));
  });
  svg.addEventListener("mouseleave", hideTip);
  host.replaceChildren(svg);
  renderLegend(ctx);
}

// Clicking a field row: snap to a phone visit within 2 days, else the clicked day.
function nearestDateFor(f, day, all) {
  let best = null;
  for (const [d] of visitsOf(f, all)) {
    const dd = Math.abs(isoDay(d) - day);
    if (dd <= 2 && (!best || dd < best[1])) best = [d, dd];
  }
  return best ? best[0] : dayIso(day);
}

function locate(e, svg, top, rowH, n) {
  const r = svg.getBoundingClientRect();
  const vb = svg.viewBox.baseVal;
  const x = ((e.clientX - r.left) * vb.width) / r.width;
  const y = ((e.clientY - r.top) * vb.height) / r.height;
  const row = Math.floor((y - top) / rowH);
  return { x, y, row: row >= 0 && row < n ? row : -1 };
}

function tipFor(f, iso, c, all) {
  const wrap = el("div");
  wrap.append(el("div", { class: "t-head" }, `${f.id.replace(/_/g, " ")} · ${fmtDay(iso, true)}`));
  const st = stageOn(f, iso);
  const row = (color, text) => {
    const r = el("div", { class: "t-row" });
    if (color) r.append(el("i", { style: `background:${color}` }));
    r.append(text);
    wrap.append(r);
  };
  if (st) row(st.color, `${st.label} (from ${st.approx ? "~" : ""}${fmtDay(st.date)})`);
  else if (f.pheno && STAGES.some((s) => f.pheno[s.key])) row(null, "Before onset of bloom");
  const n = visitsOf(f, all).get(iso) || 0;
  if (n) row("var(--ink)", `${n} phone photo${n > 1 ? "s" : ""}`);
  const cams = camsOf(f, all);
  const withFrames = cams.filter((x) => x.byDate[iso]);
  if (c) row(null, c.byDate[iso] ? `${c.label}: ${c.byDate[iso].length} frames` : `${c.label}: no frames`);
  else if (cams.length) row(null, `${withFrames.length}/${cams.length} trailcam${all ? " views" : "s"} have frames`);
  return wrap;
}

function renderFieldTimeline(ctx) {
  const host = $("fieldTimeline");
  const f = ctx.field;
  const W = host.clientWidth || 800;
  const narrow = W < 640;
  const left = narrow ? 92 : 132;
  const sc = scaleX(W, left, 10, ctx.all);
  const stageH = 64, visitH = 26, camH = 22, top = 22;
  const cams = camsOf(f, ctx.all);
  const rows = [{ kind: "stage", h: stageH }, { kind: "visits", h: visitH }, ...cams.map((c) => ({ kind: "cam", c, h: camH }))];
  let y = top;
  for (const r of rows) { r.y = y; y += r.h; }
  const H = y + 4;
  const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": `Timeline for ${f.id}` });
  const clip = svgDefs(svg, "ft", sc, H);
  drawAxis(svg, sc, 10, H);
  const isSel = (c) => !!ctx.cam && c.id === ctx.cam.id;

  for (const r of rows) {
    const g = el("svg:g", { "pointer-events": "none", "clip-path": clip });
    if (r.kind === "cam") {
      const sel = isSel(r.c);
      if (sel) svg.append(el("svg:rect", { x: 0, y: r.y, width: W, height: r.h, class: "row-sel", rx: 5 }));
      svg.append(el("svg:rect", { x: 0, y: r.y, width: W, height: r.h, class: "row-hit", rx: 5 }));
      svg.append(el("svg:text", { x: 8, y: r.y + r.h / 2 + 4, class: "row-label" + (sel ? " sel" : ""), "pointer-events": "none" }, r.c.label));
      if (ctx.status) drawFrameStatus(g, sc, ctx, [r.c], r.y + r.h / 2 - 4, 8);
      else {
        for (const [a, b] of runs(r.c.days)) {
          g.append(el("svg:rect", { x: sc.x(a), y: r.y + r.h / 2 - 4, width: Math.max(1.5, sc.x(b + 1) - sc.x(a) - 1), height: 8, rx: 2, class: "cov" + (sel ? " sel" : "") }));
        }
      }
    } else if (r.kind === "stage") {
      svg.append(el("svg:text", { x: 8, y: r.y + r.h - 10, class: "row-label sub" }, "Phenology"));
      if (!drawStages(g, sc, f, r.y + r.h - 18, 14, { labels: "stair", labelRows: r.h - 20 })) {
        g.append(el("svg:text", { x: left + 6, y: r.y + r.h - 8, class: "none" }, "No phenology dates recorded for this field"));
      }
    } else {
      svg.append(el("svg:rect", { x: 0, y: r.y, width: W, height: r.h, class: "row-hit", rx: 5 }));
      svg.append(el("svg:text", { x: 8, y: r.y + r.h / 2 + 4, class: "row-label sub", "pointer-events": "none" }, "Phone visits"));
      for (const [d] of visitsOf(f, ctx.all)) {
        g.append(el("svg:circle", { cx: sc.x(isoDay(d) + 0.5), cy: r.y + r.h / 2, r: 5, class: "visit" + (d === ctx.date ? " sel" : "") }));
      }
    }
    svg.append(g);
  }
  if (!cams.length) {
    svg.append(el("svg:text", { x: 8, y: H - 2, class: "none" }, ctx.all ? "No trailcam in this field" : `No trailcam frames in ${focusText()}`));
  }
  drawCursor(svg, sc, ctx, rows[0].y + rows[0].h - 22, H - 2);

  const rowAt = (yy) => rows.find((r) => yy >= r.y && yy < r.y + r.h);
  const pos = (e) => {
    const b = svg.getBoundingClientRect();
    return { x: ((e.clientX - b.left) * W) / b.width, y: ((e.clientY - b.top) * H) / b.height };
  };
  svg.addEventListener("click", (e) => {
    const { x, y } = pos(e);
    const r = rowAt(y);
    if (!r || (x < left && r.kind !== "cam")) return;
    const day = sc.day(x);
    const date = x < left ? undefined : r.kind === "visits" ? nearestDateFor(f, day, ctx.all) : dayIso(day);
    ctx.pick({ field: f, cam: r.kind === "cam" ? r.c : undefined, date });
  });
  svg.addEventListener("mousemove", (e) => {
    const { x, y } = pos(e);
    const r = rowAt(y);
    if (!r || x < left) return hideTip();
    showTip(e, tipFor(f, dayIso(sc.day(x)), r.kind === "cam" ? r.c : ctx.cam, ctx.all));
  });
  svg.addEventListener("mouseleave", hideTip);
  host.replaceChildren(svg);
}

function renderLegend(ctx) {
  const lg = $("legend");
  const kind = ctx.status ? "status" : "coverage";
  if (lg.dataset.kind === kind) return;
  lg.dataset.kind = kind;
  lg.replaceChildren();
  for (const s of STAGES) {
    const span = el("span");
    span.append(el("i", { style: `background:${s.color}` }), s.label);
    lg.append(span);
  }
  const mk = (cls, text) => { const s = el("span"); s.append(el("i", { class: cls }), text); lg.append(s); };
  mk("dash", "Approximate date");
  mk("dot", "Phone visit");
  if (!ctx.status) return mk("cov", "Trailcam frames");
  const names = { labeled: "Labeled", empty: "No clusters", skipped: "Skipped", unlabeled: "Unlabeled" };
  for (const st of FRAME_STATUSES) mk(`fs-key fs-${st}`, names[st]);
}

// --------------------------------------------------------------- tooltip
function showTip(e, content) {
  const t = $("tooltip");
  t.replaceChildren(content);
  t.hidden = false;
  const pad = 14;
  const r = t.getBoundingClientRect();
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}
function hideTip() { $("tooltip").hidden = true; }

// ------------------------------------------------------------------ tabs
let TAB = "viewer";
const SUBTITLES = {
  viewer: "Across trailcams beside mobile phone photos, matched by date and time of day · 2025",
  annotate: "Draw boxes around grape clusters on trailcam frames · saved as a YOLO dataset in annotated_images/",
  model: "Live log of the grape-cluster YOLO training runs started with yolo/pipeline.py",
};

// `path`: a trailcam frame to open in the Annotate tab, or a log to open in the Model tab.
function setTab(name, path) {
  TAB = name;
  $("viewerMain").hidden = name !== "viewer";
  $("annotateMain").hidden = name !== "annotate";
  $("modelMain").hidden = name !== "model";
  document.querySelectorAll(".viewer-only").forEach((n) => { n.hidden = name !== "viewer"; });
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $("subtitle").textContent = SUBTITLES[name];
  document.querySelectorAll(".annotate-only").forEach((n) => { n.hidden = name !== "annotate"; });
  // The field and all-fields charts are shared: below the photos in the Viewer,
  // below the drawing area in the Annotate tab.
  const host = $(name === "viewer" ? "viewerCharts" : "annotateCharts");
  host.append($("fieldCard"), $("overviewCard"));
  hideTip();
  if (document.fullscreenElement) document.exitFullscreen();
  if (name !== "model") ModelTab.hide();
  if (name === "viewer") { writeHash(); renderCharts(); }
  else if (name === "annotate") Annot.show(path);
  else ModelTab.show(path);
}

// Viewer "Show all data": swap every field's cameras between all views and
// dates and the focus range, then keep the selection as close as possible.
function setViewerScope(all) {
  opts.showAll = all;
  $("showAll").checked = all;
  scopeFields(FIELDS, all);
}

function rescopeViewer() {
  const prevCam = S.cam, prevFrame = frame(), p = photo();
  setViewerScope(opts.showAll);
  const cams = S.field.cameras;
  S.cam = cams.find((c) => c.id === prevCam?.id) || cams.find((c) => c.cam === prevCam?.cam) || cams[0] || null;
  if (p && photoVisible(p)) return selectPhoto(p.i);
  if (prevFrame && S.cam) {
    const fi = nearestFrame(S.cam, prevFrame.min, opts.skipNight);
    if (fi >= 0) return selectFrame(fi);
  }
  S.fi = -1; S.pi = -1;
  const d = S.date && S.date !== "undated" ? S.date : null;
  selectDate(opts.showAll || inFocus(d) ? d || FOCUS.start : d < FOCUS.start ? FOCUS.start : FOCUS.end);
}

// ----------------------------------------------------------- url + misc
function writeHash() {
  if (TAB !== "viewer") return;
  const p = photo(), f = frame();
  const q = new URLSearchParams();
  q.set("field", S.field.id);
  if (S.cam) q.set("cam", S.cam.id);
  if (opts.showAll) q.set("all", "1");
  if (p) q.set("photo", p.path);
  else if (f) q.set("frame", f.ts);
  else if (S.date) q.set("date", S.date);
  history.replaceState(null, "", "#" + q.toString());
}

function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const fid = q.get("field");
  if (!fid || !FIELDS.some((f) => f.id === fid)) return false;
  // A link to something outside the focus range turns "Show all data" on.
  const f = FIELDS.find((x) => x.id === fid);
  const ph = f.photos.find((x) => x.path === q.get("photo"));
  const fr = q.get("frame"), cam = q.get("cam") || "";
  const outside = (ph && !inFocus(ph.date)) || cam.includes("/") ||
    (fr && !inFocus(`${fr.slice(0, 4)}-${fr.slice(4, 6)}-${fr.slice(6, 8)}`)) || (q.get("date") && !inFocus(q.get("date")));
  if (q.get("all") === "1" || outside) setViewerScope(true);
  selectField(fid, { camId: q.get("cam") });
  const p = S.field.photos.find((x) => x.path === q.get("photo"));
  if (p) { selectPhoto(p.i); return true; }
  const fi = S.cam ? S.cam.frames.findIndex((x) => x.ts === q.get("frame")) : -1;
  if (fi >= 0) { selectFrame(fi); return true; }
  if (q.get("date")) selectDate(q.get("date"));
  return true;
}

function preload() {
  const c = S.cam;
  if (!c || S.fi < 0) return;
  const w = previewWidth($("trailBox"));
  const near = [];
  for (const dir of [1, -1]) {
    let i = S.fi + dir;
    while (i >= 0 && i < c.frames.length && opts.skipNight && c.frames[i].night) i += dir;
    if (i >= 0 && i < c.frames.length) near.push(c.frames[i].path);
  }
  for (const path of near) new Image().src = imgUrl(path, w);
}

// ------------------------------------------------------------------ init
function bind() {
  $("fieldSelect").onchange = (e) => selectField(e.target.value);
  $("skipNight").onchange = (e) => { opts.skipNight = e.target.checked; render(); };
  $("showAll").onchange = (e) => { opts.showAll = e.target.checked; rescopeViewer(); };
  $("taggedOnly").onchange = (e) => {
    opts.taggedOnly = e.target.checked;
    const p = photo();
    if (p && !photoVisible(p)) selectDate(S.date); else render();
  };
  document.querySelectorAll(".pane > .nav button[data-act]").forEach((b) => {
    b.onclick = () => {
      const m = b.dataset.act.match(/^(\w+)([+-]1)$/);
      const dir = +m[2];
      ({ frame: stepFrame, day: stepDay, photo: stepPhoto, visit: stepVisit })[m[1]](dir);
    };
  });
  ZOOM.trailBox = new Zoomer($("trailBox"), $("trailImg"));
  ZOOM.phoneBox = new Zoomer($("phoneBox"), $("phoneImg"));
  document.querySelectorAll(".tabs [role=tab]").forEach((b) => { b.onclick = () => setTab(b.dataset.tab); });
  $("trailAnnotate").onclick = () => { const f = frame(); if (f) setTab("annotate", f.path); };
  document.addEventListener("keydown", (e) => {
    if (TAB !== "viewer") return;
    if (e.target.closest("select, textarea, input:not([type=checkbox])") || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === "ArrowLeft" || k === "ArrowRight") {
      const dir = k === "ArrowLeft" ? -1 : 1;
      e.shiftKey ? stepDay(dir) : stepFrame(dir);
    } else if (k === "," || k === ".") stepPhoto(k === "," ? -1 : 1);
    else if (k === "[" || k === "]") stepVisit(k === "[" ? -1 : 1);
    else return;
    e.preventDefault();
  });
  let raf = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(renderCharts);
  }).observe($("overview"));
}

async function init() {
  bind();
  try {
    const res = await fetch("/api/index");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    IDX = await res.json();
  } catch (err) {
    document.querySelector("main").prepend(el("p", { class: "notice warn" }, `Could not load the image index: ${err.message}`));
    return;
  }
  FIELDS = prepare(IDX);
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get("tab") === "annotate" || q.get("tab") === "model") {
    selectField(defaultField().id);
    setTab(q.get("tab"), q.get(q.get("tab") === "annotate" ? "src" : "run"));
  } else if (!readHash()) {
    selectField(defaultField().id);
  }
}

const defaultField = () => FIELDS.find((f) => f.cameras.length && f.photos.length) || FIELDS.find((f) => f.cameras.length) || FIELDS[0];

init();
