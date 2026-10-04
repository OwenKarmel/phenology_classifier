"use strict";

// ------------------------------------------------------------- annotate
// Annotate tab: draw YOLO bounding boxes on trailcam frames, Label Studio-style.
// Boxes are kept as normalized image coordinates {x0, y0, x1, y1} (0-1), so
// they don't depend on whether a preview or the original file is on screen.
// Every change is saved to the server, which writes annotated_images/
// (see visualization/annotations.py). Uses the helpers and Zoomer of app.js.

const Annot = (() => {
  // Box outlines are drawn over foliage: magenta first, it stands out on green.
  const CLASS_COLORS = ["#ff2bd6", "#ffd60a", "#00e5ff", "#ff7a1a", "#a98bff", "#ff4d5e", "#3dffa8", "#ffffff", "#c6ff3d"];
  const FILTERS = [
    { key: "all", label: "All", title: "Every frame" },
    { key: "unlabeled", label: "Unlabeled", title: "Frames not yet submitted or skipped" },
    { key: "labeled", label: "Labeled", title: "Frames in the dataset, with or without boxes" },
    { key: "skipped", label: "Skipped", title: "Skipped frames" },
  ];
  const STATUS = {
    unlabeled: { label: "Unlabeled", color: "var(--ann-unlabeled)" },
    labeled: { label: "Labeled", color: "var(--ann-labeled)" },
    empty: { label: "No clusters", color: "var(--ann-empty)" },
    skipped: { label: "Skipped", color: "var(--ann-skipped)" },
  };
  // Unless "Show all data" is ticked: Across views only, from onset of prebloom
  // to the end of the day of the last phone photo visit.
  const FOCUS = { view: "Across", start: "2025-06-11", end: "2025-07-15" };
  const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
  const CURSORS = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };
  const GRAB = 7; // px around a handle or edge that grabs it
  const MIN_DRAG = 4; // px: shorter drags are clicks
  const UNDO_DEPTH = 100;

  const A = {
    sources: [], src: null, fi: -1, list: [], // list: frame indexes shown (daytime filter)
    classes: ["grape_cluster"], dir: "", api: null, // api: true, or why saving is unavailable
    recs: new Map(), // frame path -> record (status "done" | "skipped", YOLO boxes)
    boxes: [], sel: null, hover: null, uid: 0,
    cls: 0, tool: "rect", hideAll: false, space: false,
    filter: "all", dayOnly: true, showAll: false,
    hist: new Map(), clip: null,
    ready: false, drag: null, press: null, cursor: null,
    dirty: new Set(), failed: new Map(), saving: 0, saveTimer: 0, chain: Promise.resolve(),
    starting: null, toastTimer: 0,
  };

  // ------------------------------------------------------------ helpers
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const classColor = (c) => CLASS_COLORS[c % CLASS_COLORS.length];
  const className = (c) => A.classes[c] ?? `class ${c}`;
  const viewLabel = (v) => v || "Unsorted";
  const fieldKey = (s) => `${s.year}/${s.field}`;
  const srcName = (s) => `${s.field.replace(/_/g, " ")} · ${s.cameraLabel} · ${viewLabel(s.view)}`;
  const plural = (n, word, many = word + "s") => `${n} ${n === 1 ? word : many}`;
  const aFrame = () => (A.src && A.fi >= 0 ? A.src.frames[A.fi] : null);
  const canEdit = () => A.api === true;
  const toYolo = (b) => {
    const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1), y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
    return [b.cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0];
  };
  const fromYolo = ([cls, cx, cy, w, h]) => ({ id: ++A.uid, cls, x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2, hidden: false });
  const area = (b) => Math.abs((b.x1 - b.x0) * (b.y1 - b.y0));
  function iou(a, b) {
    const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0), h = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
    if (w <= 0 || h <= 0) return 0;
    return (w * h) / (area(a) + area(b) - w * h);
  }

  function statusOf(path) {
    const r = A.recs.get(path);
    if (!r) return "unlabeled";
    if (r.status === "skipped") return "skipped";
    return r.boxes.length ? "labeled" : "empty";
  }
  function matches(path, filter) {
    const st = statusOf(path);
    if (filter === "unlabeled") return st === "unlabeled";
    if (filter === "labeled") return st === "labeled" || st === "empty";
    if (filter === "skipped") return st === "skipped";
    return true;
  }

  function prepareSources(list) {
    return list.map((s) => ({
      ...s,
      frames: s.frames.map((n) => {
        const file = n.includes(".") ? n : n + ".jpg";
        const date = `${file.slice(0, 4)}-${file.slice(4, 6)}-${file.slice(6, 8)}`;
        const hour = +file.slice(9, 11);
        return { date, min: isoDay(date) * DAY + hour * 60 + +file.slice(11, 13), night: hour < 5 || hour >= 21, path: `${s.dir}/${file}` };
      }),
    }));
  }

  // Which frames and camera views are on offer (daytime / "Show all data").
  const inRange = (f) => A.showAll || (f.date >= FOCUS.start && f.date <= FOCUS.end);
  const shown = (f) => inRange(f) && (!A.dayOnly || !f.night);
  const srcShown = (s) => A.showAll || (s.view === FOCUS.view && s.frames.some(inRange));
  const visible = () => A.sources.filter(srcShown);
  const focusText = () => `${FOCUS.view} views, ${fmtDay(FOCUS.start)} – ${fmtDay(FOCUS.end)}`;

  // Frame of `s` nearest in time to `min`: a shown frame if possible, else one in range.
  function nearestIdx(s, min) {
    for (const ok of [shown, inRange, () => true]) {
      let best = -1;
      s.frames.forEach((f, i) => {
        if (ok(f) && (best < 0 || Math.abs(f.min - min) < Math.abs(s.frames[best].min - min))) best = i;
      });
      if (best >= 0) return best;
    }
    return -1;
  }

  // -------------------------------------------------------------- start
  async function start() {
    bind();
    let list = IDX.trailcams;
    if (!list) {
      // Index written before the Annotate tab existed: rebuild it once.
      try { list = (await (await fetch("/api/index?rebuild")).json()).trailcams; } catch { /* below */ }
    }
    A.sources = prepareSources(list || []);
    try {
      const res = await fetch("/api/annotations");
      if (res.status === 404) throw new Error("this server was started before the Annotate tab existed. Restart it (python visualization/server.py) to save boxes");
      if (!res.ok) throw new Error(`the server answered HTTP ${res.status}`);
      const j = await res.json();
      A.classes = j.classes;
      A.dir = j.dir;
      for (const r of j.images) A.recs.set(r.source, r);
      A.api = true;
    } catch (err) {
      A.api = `Annotations can't be loaded or saved: ${err.message}.`;
    }
  }

  async function show(path) {
    await (A.starting ||= start());
    if (TAB !== "annotate") return;
    if (path && openPath(path, true)) return;
    if (path) toast("That frame isn't in the trailcam index.");
    if (A.src) { render(true); writeHash(); return; }
    const vf = frame(); // where the Viewer tab is
    if (vf && openPath(vf.path, false)) return;
    const s = visible()[0];
    if (s) openFrame(s, nearestIdx(s, s.frames[0].min));
    else render(false);
  }

  // Open a frame by path. A frame outside the default selection turns on
  // "Show all data" when it was asked for (link, reload), or else gives way to
  // the nearest frame of this camera that is in the selection.
  function openPath(path, explicit) {
    const dir = path.slice(0, path.lastIndexOf("/"));
    const s = A.sources.find((x) => x.dir === dir);
    const fi = s ? s.frames.findIndex((f) => f.path === path) : -1;
    if (fi < 0) return false;
    if (srcShown(s) && inRange(s.frames[fi])) {
      openFrame(s, fi);
    } else if (explicit) {
      setShowAll(true);
      openFrame(s, fi);
      toast(`This frame is outside ${focusText()}, so "Show all data" is on.`);
    } else {
      const t = closestSource(s);
      if (!t) return false;
      openFrame(t, nearestIdx(t, s.frames[fi].min));
    }
    return true;
  }

  // The shown source most like `s`: same camera, else same field, else any.
  function closestSource(s) {
    if (srcShown(s)) return s;
    const vis = visible();
    return vis.find((x) => fieldKey(x) === fieldKey(s) && x.camera === s.camera) || vis.find((x) => fieldKey(x) === fieldKey(s)) || vis[0] || null;
  }

  function setShowAll(on) {
    A.showAll = on;
    $("annAll").checked = on;
  }

  // After a filter checkbox changes: stay put if possible, else move to the
  // nearest frame that is still shown.
  function rescope() {
    if (!A.src) return;
    const f = aFrame(), s = closestSource(A.src);
    if (!s) {
      A.list = listOf(A.src);
      renderAxis();
      refresh();
      return toast(`No trailcam frames in ${focusText()}.`);
    }
    if (s !== A.src || !inRange(f)) return openFrame(s, nearestIdx(s, f.min));
    A.list = listOf(s);
    renderAxis();
    refresh();
  }

  function openFrame(s, fi) {
    if (!s || fi < 0 || fi >= s.frames.length) return;
    flushSave();
    const same = s === A.src;
    const f = s.frames[fi];
    A.ready = A.ready && ZOOM.annBox.path === f.path;
    A.src = s;
    A.fi = fi;
    A.list = listOf(s);
    A.boxes = (A.recs.get(f.path)?.boxes || []).map(fromYolo);
    A.sel = A.hover = A.drag = A.press = null;
    render(same);
    writeHash();
    preload();
  }

  const listOf = (s) => s.frames.map((_, i) => i).filter((i) => shown(s.frames[i]));

  function switchSource(s) {
    if (!s || s === A.src) return;
    const f = aFrame();
    openFrame(s, nearestIdx(s, f ? f.min : s.frames[0].min));
  }

  // Next frame in direction `dir` that passes the daytime and status filters.
  function step(dir, filter = A.filter) {
    const f = aFrame();
    if (!f) return;
    const fr = A.src.frames;
    for (let i = A.fi + dir; i >= 0 && i < fr.length; i += dir) {
      if (!shown(fr[i])) continue;
      if (matches(fr[i].path, filter)) return openFrame(A.src, i);
    }
    const what = filter === "all" ? "" : FILTERS.find((x) => x.key === filter).label.toLowerCase() + " ";
    toast(`No ${what}frames ${dir > 0 ? "after" : "before"} this one in ${srcName(A.src)}.`);
  }

  // Same time of day on the previous/next day that has frames.
  function stepDay(dir) {
    const f = aFrame();
    if (!f) return;
    const tod = f.min % DAY;
    let best = -1;
    const day = (i) => Math.floor(A.src.frames[i].min / DAY);
    for (let i = A.fi + dir; i >= 0 && i < A.src.frames.length; i += dir) {
      if (!inRange(A.src.frames[i])) break; // frames are in time order
      if (day(i) === day(A.fi)) continue;
      if (best >= 0 && day(i) !== day(best)) break;
      const d = Math.abs((A.src.frames[i].min % DAY) - tod);
      if (best < 0 || d < Math.abs((A.src.frames[best].min % DAY) - tod)) best = i;
    }
    if (best >= 0) openFrame(A.src, best);
    else toast(`No ${dir > 0 ? "later" : "earlier"} day in ${srcName(A.src)}.`);
  }

  function preload() {
    const w = previewWidth($("annBox"));
    const fr = A.src.frames;
    for (const dir of [1, -1]) {
      for (let i = A.fi + dir, n = 0; i >= 0 && i < fr.length && n < 2; i += dir) {
        if (!shown(fr[i]) || !matches(fr[i].path, A.filter)) continue;
        new Image().src = imgUrl(fr[i].path, w);
        n++;
      }
    }
  }

  function writeHash() {
    const f = aFrame();
    const q = new URLSearchParams({ tab: "annotate" });
    if (f) q.set("src", f.path);
    history.replaceState(null, "", "#" + q.toString());
  }

  // ------------------------------------------------------------- edits
  function snapshot() {
    const r = A.recs.get(aFrame().path);
    return { status: r ? r.status : null, boxes: A.boxes.map(({ cls, x0, y0, x1, y1 }) => ({ cls, x0, y0, x1, y1 })) };
  }
  function histOf(path) {
    if (!A.hist.has(path)) A.hist.set(path, { undo: [], redo: [] });
    return A.hist.get(path);
  }

  // Store a change to the current frame: `before` goes on the undo stack and
  // the frame is saved. Without an explicit status, a frame with boxes is
  // "done", and one whose last box was removed is unlabeled again (frames
  // without boxes only become negatives through Submit).
  function commit(before, status) {
    const path = aFrame().path;
    const h = histOf(path);
    h.undo.push(before);
    if (h.undo.length > UNDO_DEPTH) h.undo.shift();
    h.redo.length = 0;
    if (status === undefined) status = A.boxes.length ? "done" : before.status === "skipped" ? "skipped" : null;
    store(path, status);
  }

  function store(path, status) {
    const old = A.recs.get(path);
    if (!status) A.recs.delete(path);
    else A.recs.set(path, { ...(old || { source: path }), status, boxes: A.boxes.map(toYolo) });
    queueSave(path);
    refresh();
  }

  function restore(snap) {
    A.boxes = snap.boxes.map((b) => ({ ...b, id: ++A.uid, hidden: false }));
    A.sel = A.hover = null;
    store(aFrame().path, snap.status);
  }

  function edit(fn) {
    if (!aFrame()) return;
    if (!canEdit()) return toast(A.api || "Still loading the annotations…");
    fn();
  }

  const undo = () => edit(() => {
    const h = histOf(aFrame().path);
    const snap = h.undo.pop();
    if (!snap) return toast("Nothing to undo on this frame.");
    h.redo.push(snapshot());
    restore(snap);
  });
  const redo = () => edit(() => {
    const h = histOf(aFrame().path);
    const snap = h.redo.pop();
    if (!snap) return toast("Nothing to redo.");
    h.undo.push(snapshot());
    restore(snap);
  });

  const submit = () => edit(() => {
    if (A.recs.get(aFrame().path)?.status !== "done") commit(snapshot(), "done");
    flushSave();
    step(1);
  });
  const skip = () => edit(() => {
    if (A.recs.get(aFrame().path)?.status !== "skipped") commit(snapshot(), "skipped");
    flushSave();
    step(1);
  });
  const reset = () => edit(() => {
    if (!A.recs.get(aFrame().path) && !A.boxes.length) return;
    const before = snapshot();
    A.boxes = [];
    A.sel = null;
    commit(before, null);
    toast("Reset: this frame is unlabeled again (Ctrl+Z to undo).");
  });

  const deleteSel = () => edit(() => {
    if (!A.sel) return;
    const before = snapshot();
    A.boxes = A.boxes.filter((b) => b !== A.sel);
    A.sel = null;
    commit(before);
  });
  const deleteAll = () => edit(() => {
    if (!A.boxes.length) return;
    const before = snapshot();
    A.boxes = [];
    A.sel = null;
    commit(before);
  });

  const chooseClass = (i) => {
    if (i >= A.classes.length) return;
    A.cls = i;
    if (A.sel && A.sel.cls !== i && canEdit()) {
      const before = snapshot();
      A.sel.cls = i;
      commit(before);
    } else {
      refresh();
    }
  };

  // Add boxes (YOLO lists), skipping any that are already here.
  function addBoxes(list, what) {
    const before = snapshot();
    let added = 0;
    for (const y of list) {
      if (y[0] >= A.classes.length) continue;
      const nb = fromYolo(y);
      if (A.boxes.some((b) => b.cls === nb.cls && iou(b, nb) > 0.9)) continue;
      A.boxes.push(nb);
      added++;
    }
    if (!added) return toast("Those boxes are already on this frame.");
    A.sel = null;
    commit(before);
    toast(`${what}: ${plural(added, "box", "boxes")} added.`);
  }

  const copyPrev = () => edit(() => {
    const fr = A.src.frames;
    const has = (i) => (A.recs.get(fr[i].path)?.boxes.length || 0) > 0;
    let i = A.fi - 1;
    while (i >= 0 && !has(i)) i--;
    if (i < 0) for (i = A.fi + 1; i < fr.length && !has(i); i++);
    if (i < 0 || i >= fr.length) return toast(`No other frame of ${srcName(A.src)} has boxes yet.`);
    addBoxes(A.recs.get(fr[i].path).boxes, `From ${fmtDay(fr[i].date)} ${fmtHM(fr[i].min)}`);
  });
  const copy = () => {
    const list = A.sel ? [A.sel] : A.boxes;
    if (!list.length) return toast("No boxes to copy.");
    A.clip = list.map(toYolo);
    toast(`Copied ${plural(list.length, "box", "boxes")}. Ctrl+V pastes on any frame.`);
  };
  const paste = () => edit(() => {
    if (!A.clip) return toast("Nothing copied yet: Ctrl+C copies the selected box, or all boxes.");
    addBoxes(A.clip, "Pasted");
  });
  const duplicate = () => edit(() => {
    if (!A.sel) return;
    const before = snapshot();
    const b = A.sel, w = b.x1 - b.x0, h = b.y1 - b.y0;
    const x0 = Math.min(1 - w, b.x0 + 0.01), y0 = Math.min(1 - h, b.y0 + 0.01);
    const nb = { id: ++A.uid, cls: b.cls, x0, y0, x1: x0 + w, y1: y0 + h, hidden: false };
    A.boxes.push(nb);
    A.sel = nb;
    commit(before);
  });

  function cycle() {
    const vis = A.boxes.filter((b) => !b.hidden);
    if (!vis.length) return;
    A.sel = vis[(vis.indexOf(A.sel) + 1) % vis.length];
    refresh();
  }
  function unselect() {
    if (!A.sel) return;
    A.sel = null;
    refresh();
  }
  function toggleHideSel() {
    if (!A.sel) return;
    A.sel.hidden = !A.sel.hidden;
    if (A.sel.hidden) A.sel = null;
    refresh();
  }
  function toggleHideAll() {
    A.hideAll = !A.hideAll;
    A.drag = null;
    toast(A.hideAll ? "Boxes hidden (Ctrl+H shows them)." : "Boxes shown.");
    refresh();
  }
  function setTool(t) {
    A.tool = t;
    refresh();
  }

  async function postClasses(names) {
    try {
      const res = await fetch("/api/classes", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ classes: names }) });
      const j = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
      A.classes = j.classes;
      refresh();
    } catch (err) {
      toast(`Couldn't change the labels: ${err.message}`);
    }
  }
  const addClass = () => edit(() => {
    const name = prompt("Name of the new label (it gets YOLO class id " + A.classes.length + "):", "");
    if (name && name.trim()) postClasses([...A.classes, name.trim()]);
  });
  const renameClass = (i) => edit(() => {
    const name = prompt(`Rename label ${i} (the class id stays ${i}):`, A.classes[i]);
    if (name && name.trim() && name.trim() !== A.classes[i]) postClasses(A.classes.map((n, k) => (k === i ? name.trim() : n)));
  });

  // -------------------------------------------------------------- saving
  // Saves are debounced, sent in order, and flushed when leaving a frame.
  function queueSave(path) {
    A.dirty.add(path);
    clearTimeout(A.saveTimer);
    A.saveTimer = setTimeout(flushSave, 400);
  }

  function flushSave(keepalive = false) {
    clearTimeout(A.saveTimer);
    for (const path of A.dirty) {
      const r = A.recs.get(path);
      const body = JSON.stringify({ source: path, status: r ? r.status : "none", boxes: r ? r.boxes : [] });
      A.saving++;
      A.chain = A.chain.then(() => post(path, body, keepalive));
    }
    A.dirty.clear();
    renderSaveState();
  }

  async function post(path, body, keepalive) {
    try {
      const res = await fetch("/api/annotations", { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive });
      const j = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
      A.failed.delete(path);
      // Take the server's details (size, split, file names), keep newer local edits.
      const local = A.recs.get(path);
      if (j.record && local) A.recs.set(path, { ...j.record, status: local.status, boxes: local.boxes });
    } catch (err) {
      A.failed.set(path, err.message);
    } finally {
      A.saving--;
      if (A.saving || A.drag) renderSaveState();
      else renderSide(); // split, file names and image size are known now
    }
  }

  function retry() {
    for (const path of A.failed.keys()) A.dirty.add(path);
    A.failed.clear();
    flushSave();
  }

  // -------------------------------------------------------------- render
  function render(sameSrc) {
    renderSources();
    renderImage(sameSrc);
    renderSide();
    renderAxis();
    drawScrub();
    drawOverlay();
  }

  // After an edit or selection change (same image).
  function refresh() {
    renderSources();
    renderSide();
    renderCaption();
    drawScrub();
    drawOverlay();
  }

  function countsByDir() {
    const m = new Map();
    for (const [path, r] of A.recs) {
      if (r.status !== "done") continue;
      const dir = path.slice(0, path.lastIndexOf("/"));
      m.set(dir, (m.get(dir) || 0) + 1);
    }
    return m;
  }

  function renderSources() {
    const sel = $("annField");
    const vis = visible();
    const years = new Set(vis.map((s) => s.year));
    const keys = [...new Set(vis.map(fieldKey))];
    if ([...sel.options].map((o) => o.value).join() !== keys.join()) {
      sel.replaceChildren(...keys.map((k) => {
        const [year, f] = k.split("/");
        return el("option", { value: k }, f.replace(/_/g, " ") + (years.size > 1 ? ` · ${year}` : ""));
      }));
    }
    const cams = $("annCams"), views = $("annViews");
    cams.replaceChildren();
    views.replaceChildren();
    if (!A.src) return;
    sel.value = fieldKey(A.src);
    const done = countsByDir();
    const inField = vis.filter((s) => fieldKey(s) === fieldKey(A.src));
    const chip = (label, on, n, title, onclick) => {
      const b = el("button", { type: "button", role: "radio", "aria-checked": String(on), title }, label);
      if (n) b.append(el("span", { class: "count" }, String(n)));
      b.onclick = onclick;
      return b;
    };
    for (const cam of [...new Set(inField.map((s) => s.camera))]) {
      const srcs = inField.filter((s) => s.camera === cam);
      const n = srcs.reduce((t, s) => t + (done.get(s.dir) || 0), 0);
      cams.append(chip(srcs[0].cameraLabel, cam === A.src.camera, n, `${plural(n, "labeled frame")} · views: ${srcs.map((s) => viewLabel(s.view)).join(", ")}`,
        () => switchSource(srcs.find((s) => s.view === A.src.view) || srcs[0])));
    }
    for (const s of inField.filter((x) => x.camera === A.src.camera)) {
      const n = done.get(s.dir) || 0;
      views.append(chip(viewLabel(s.view), s === A.src, n, `${s.dir} · ${plural(s.frames.length, "frame")}, ${plural(n, "labeled")}`, () => switchSource(s)));
    }
    const seg = $("annFilter");
    seg.replaceChildren();
    for (const f of FILTERS) {
      const b = el("button", { type: "button", role: "radio", "aria-checked": String(f.key === A.filter), title: f.title }, f.label);
      b.onclick = () => { A.filter = f.key; refresh(); preload(); };
      seg.append(b);
    }
  }

  function renderImage(sameSrc) {
    const f = aFrame(), s = A.src;
    const full = $("annFull");
    if (!f) {
      $("annKind").textContent = "Trailcam";
      $("annTitle").textContent = "—";
      full.setAttribute("aria-disabled", "true");
      setImage("annBox", "annImg", null, !A.sources.length ? "No trailcam frames found under raw_data/Raw_Data."
        : visible().length ? "No frame selected." : `No trailcam frames in ${focusText()}. Tick "Show all data".`);
      return;
    }
    $("annKind").textContent = srcName(s);
    const fieldObj = FIELDS.find((x) => x.id === s.field && String(IDX.year) === s.year);
    const st = fieldObj && stageOn(fieldObj, f.date);
    $("annTitle").textContent = `${fmtDay(f.date, true)} · ${fmtHM(f.min)}` + (st ? ` · ${st.label}` : "");
    full.href = rawUrl(f.path);
    full.removeAttribute("aria-disabled");
    setImage("annBox", "annImg", f.path, null, sameSrc);
    renderCaption();
  }

  function renderCaption() {
    const f = aFrame();
    const st = f ? statusOf(f.path) : null;
    $("annCaption").textContent = f
      ? `${srcName(A.src)} · ${fmtDay(f.date, true)} ${fmtHM(f.min)} · ${STATUS[st].label}${A.boxes.length ? ` · ${plural(A.boxes.length, "box", "boxes")}` : ""}  —  A D frame · drag to draw · Ctrl+Enter submit · Esc exit`
      : "";
  }

  function renderSide() {
    const f = aFrame();
    const st = f ? statusOf(f.path) : "unlabeled";
    const rec = f && A.recs.get(f.path);
    const pill = $("annPill");
    pill.replaceChildren(el("i", { style: `background:${STATUS[st].color}` }),
      STATUS[st].label + (st === "labeled" ? ` · ${plural(rec.boxes.length, "box", "boxes")}` : ""));
    for (const id of ["annSubmit", "annSkip", "annReset", "annCopyPrev", "annDeleteAll", "annAddLabel"]) $(id).disabled = !f || !canEdit();
    $("annReset").disabled ||= !rec && !A.boxes.length;
    $("annDeleteAll").disabled ||= !A.boxes.length;

    const hint = $("annHint");
    hint.className = "hint";
    if (!canEdit()) {
      hint.className = "hint warn";
      hint.textContent = A.api || "Loading annotations…";
    } else if (!f) {
      hint.textContent = "";
    } else if (st === "unlabeled") {
      hint.textContent = A.boxes.length
        ? "Unlabeled."
        : "Drag on the photo to box each grape cluster. No clusters? Submit (Ctrl+Enter) saves the frame as a negative example.";
    } else if (st === "labeled") {
      hint.textContent = `In the dataset (${rec.split || "…"} split). Changes save automatically.`;
    } else if (st === "empty") {
      hint.textContent = `In the dataset as a negative example: an empty label file (${rec.split || "…"} split).`;
    } else {
      hint.textContent = "Left out of the dataset. Draw a box or Submit to include it.";
    }
    renderSaveState();
    renderLabels();
    renderRegions();
    renderProgress();
    renderLegend();
    for (const b of document.querySelectorAll("#annTools [data-tool]")) b.setAttribute("aria-pressed", String(b.dataset.tool === A.tool));
    $("annHideAll").setAttribute("aria-pressed", String(A.hideAll));
    $("annPrev").disabled = !f;
    $("annNext").disabled = !f;
  }

  function renderSaveState() {
    const n = $("annSave");
    n.replaceChildren();
    n.className = "muted save";
    const f = aFrame();
    if (A.failed.size) {
      n.className = "save err";
      n.append(`Not saved: ${[...A.failed.values()][0]} `);
      const b = el("button", { type: "button", class: "linkish" }, "Retry");
      b.onclick = retry;
      n.append(b);
    } else if (A.saving || A.dirty.size) {
      n.append("Saving…");
    } else if (f && A.recs.get(f.path)?.label && A.recs.get(f.path).status === "done") {
      n.append(el("span", { title: `${A.dir}/${A.recs.get(f.path).label}` }, "Saved ✓"));
    }
  }

  function renderLabels() {
    const box = $("annLabels");
    box.replaceChildren();
    A.classes.forEach((name, i) => {
      const b = el("button", { type: "button", class: "label-btn" + (i === A.cls ? " on" : ""), "aria-pressed": String(i === A.cls),
        title: `New boxes get this label (${i + 1}); with a box selected, relabels it. Double-click to rename. YOLO class id ${i}.` });
      b.append(el("i", { style: `background:${classColor(i)}` }), el("span", { class: "name" }, name));
      if (i < 9) b.append(el("kbd", {}, String(i + 1)));
      b.onclick = () => chooseClass(i);
      b.ondblclick = () => renameClass(i);
      box.append(b);
    });
  }

  function renderRegions() {
    const ol = $("annRegions");
    ol.replaceChildren();
    $("annRegCount").textContent = A.boxes.length ? `(${A.boxes.length})` : "";
    const f = aFrame();
    const rec = f && A.recs.get(f.path);
    if (!A.boxes.length) {
      ol.append(el("li", { class: "none muted" }, "No boxes on this frame."));
      return;
    }
    A.boxes.forEach((b, i) => {
      const li = el("li", { class: (b === A.sel ? "on " : "") + (b === A.hover ? "hov " : "") + (b.hidden ? "is-hidden" : "") });
      const w = Math.abs(b.x1 - b.x0), h = Math.abs(b.y1 - b.y0);
      const size = rec?.width ? `${Math.round(w * rec.width)} × ${Math.round(h * rec.height)} px` : `${(w * 100).toFixed(1)} × ${(h * 100).toFixed(1)} %`;
      li.append(el("i", { class: "sw", style: `background:${classColor(b.cls)}` }), el("span", { class: "rname" }, `${i + 1}. ${className(b.cls)}`), el("span", { class: "rsize muted" }, size));
      const eye = el("button", { type: "button", class: "icon", title: b.hidden ? "Show (Alt+H)" : "Hide (Alt+H)", "aria-pressed": String(b.hidden) });
      eye.innerHTML = b.hidden
        ? '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z"/><path d="M2.5 13.5l11-11"/></svg>'
        : '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z"/><circle cx="8" cy="8" r="2"/></svg>';
      eye.onclick = (e) => { e.stopPropagation(); b.hidden = !b.hidden; if (b.hidden && A.sel === b) A.sel = null; refresh(); };
      const del = el("button", { type: "button", class: "icon", title: "Delete (Backspace)" });
      del.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>';
      del.onclick = (e) => { e.stopPropagation(); A.sel = b; deleteSel(); };
      li.append(eye, del);
      li.onclick = () => { if (b.hidden) b.hidden = false; A.sel = b; refresh(); };
      li.onmouseenter = () => { A.hover = b; drawOverlay(); };
      li.onmouseleave = () => { A.hover = null; drawOverlay(); };
      ol.append(li);
    });
  }

  function renderProgress() {
    const box = $("annProgress");
    box.replaceChildren();
    let labeled = 0, empty = 0, skipped = 0, boxes = 0, train = 0, val = 0;
    for (const r of A.recs.values()) {
      if (r.status === "skipped") { skipped++; continue; }
      if (r.boxes.length) labeled++; else empty++;
      boxes += r.boxes.length;
      if (r.split === "val") val++; else if (r.split === "train") train++;
    }
    const row = (k, v) => box.append(el("div", { class: "kv" }, k), el("div", { class: "kv-v" }, v));
    row("Images", `${labeled + empty} (${labeled} with clusters, ${plural(empty, "negative")})`);
    row("Boxes", String(boxes));
    row("Split", `${train} train · ${val} val`);
    row("Skipped", String(skipped));
    if (A.dir) box.append(el("div", { class: "kv-path", title: "YOLO labels, image links, dataset.yaml" }, A.dir));
  }

  function renderLegend() {
    const lg = $("annLegend");
    lg.replaceChildren();
    if (!A.src) return;
    const n = { labeled: 0, empty: 0, skipped: 0, unlabeled: 0 };
    for (const i of A.list) n[statusOf(A.src.frames[i].path)]++;
    for (const k of ["labeled", "empty", "skipped", "unlabeled"]) {
      const s = el("span");
      s.append(el("i", { style: `background:${STATUS[k].color}` }), `${n[k]} ${STATUS[k].label.toLowerCase()}`);
      lg.append(s);
    }
    const k = A.list.indexOf(A.fi);
    lg.append(el("span", { class: "pos" }, k >= 0 ? `Frame ${k + 1} of ${A.list.length}` : `Night frame (${plural(A.list.length, "daytime frame")})`));
  }

  // ------------------------------------------------------------ scrubber
  function renderAxis() {
    const axis = $("annAxis");
    axis.replaceChildren();
    if (!A.src || !A.list.length) return;
    const n = A.list.length, W = axis.clientWidth || 600;
    let prev = null, lastX = -1e9;
    A.list.forEach((fi, k) => {
      const f = A.src.frames[fi];
      const month = f.date.slice(0, 7);
      if (month === prev) return;
      prev = month;
      const x = (k / n) * W;
      if (k > 0 && (x - lastX < 48 || W - x < 30)) return; // no overlapping labels
      lastX = x;
      axis.append(el("span", { style: `left:${(k / n) * 100}%` }, k === 0 ? fmtDay(f.date) : MONTHS[+f.date.slice(5, 7) - 1]));
    });
  }

  function drawScrub() {
    const cv = $("annScrub");
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!A.src || !A.list.length) return;
    const css = getComputedStyle(document.documentElement);
    const col = {};
    for (const k of ["unlabeled", "labeled", "empty", "skipped"]) col[k] = css.getPropertyValue(`--ann-${k}`).trim();
    const n = A.list.length, bw = W / n, gap = bw > 4 ? 1 : 0;
    A.list.forEach((fi, k) => {
      const x = Math.floor(k * bw);
      ctx.fillStyle = col[statusOf(A.src.frames[fi].path)];
      ctx.fillRect(x, 7, Math.max(1, Math.floor((k + 1) * bw) - x - gap), H - 12);
    });
    const k = A.list.indexOf(A.fi);
    if (k >= 0) {
      const x = (k + 0.5) * bw;
      ctx.fillStyle = css.getPropertyValue("--ink").trim();
      ctx.fillRect(x - 1, 3, 2, H - 4);
      ctx.beginPath();
      ctx.moveTo(x - 5, 0); ctx.lineTo(x + 5, 0); ctx.lineTo(x, 6); ctx.closePath();
      ctx.fill();
    }
  }

  const scrubIndex = (e) => {
    const cv = $("annScrub"), r = cv.getBoundingClientRect();
    const k = Math.floor(((e.clientX - r.left) / r.width) * A.list.length);
    return A.list[Math.max(0, Math.min(A.list.length - 1, k))];
  };

  // ------------------------------------------------------------- overlay
  // Screen <-> normalized image coordinates, following the Zoomer's transform.
  function geom() {
    const z = ZOOM.annBox, { fx, fy, fw, fh } = z.fit(), s = z.s, tx = z.tx, ty = z.ty;
    return {
      X: (nx) => tx + s * (fx + nx * fw), Y: (ny) => ty + s * (fy + ny * fh),
      NX: (x) => ((x - tx) / s - fx) / fw, NY: (y) => ((y - ty) / s - fy) / fh,
      sw: s * fw, sh: s * fh,
    };
  }
  const screenRect = (b, g) => {
    const x = g.X(Math.min(b.x0, b.x1)), y = g.Y(Math.min(b.y0, b.y1));
    return { x, y, width: g.X(Math.max(b.x0, b.x1)) - x, height: g.Y(Math.max(b.y0, b.y1)) - y };
  };
  const local = (e) => {
    const r = $("annBox").getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const inside = (b, p, g) => { const r = screenRect(b, g); return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height; };
  const small = (r) => r.width < 28 || r.height < 28; // corner handles only

  function handlePos(r, h) {
    return {
      x: h.includes("w") ? r.x : h.includes("e") ? r.x + r.width : r.x + r.width / 2,
      y: h.includes("n") ? r.y : h.includes("s") ? r.y + r.height : r.y + r.height / 2,
    };
  }

  // Handle (or edge) of the selected box under p.
  function handleAt(p, g) {
    const b = A.sel;
    if (!b || b.hidden || A.hideAll) return null;
    const r = screenRect(b, g);
    for (const h of ["nw", "ne", "se", "sw"]) {
      const q = handlePos(r, h);
      if (Math.abs(p.x - q.x) <= GRAB && Math.abs(p.y - q.y) <= GRAB) return h;
    }
    const inX = p.x > r.x && p.x < r.x + r.width, inY = p.y > r.y && p.y < r.y + r.height;
    if (inX && Math.abs(p.y - r.y) <= GRAB / 1.5) return "n";
    if (inX && Math.abs(p.y - r.y - r.height) <= GRAB / 1.5) return "s";
    if (inY && Math.abs(p.x - r.x) <= GRAB / 1.5) return "w";
    if (inY && Math.abs(p.x - r.x - r.width) <= GRAB / 1.5) return "e";
    return null;
  }

  // Smallest visible box under p, so nested boxes stay reachable.
  function boxAt(p, g) {
    if (A.hideAll) return null;
    let best = null;
    for (const b of A.boxes) if (!b.hidden && inside(b, p, g) && (!best || area(b) < area(best))) best = b;
    return best;
  }

  const panning = (e) => A.tool === "pan" || A.space || A.hideAll || e.button === 1 || !canEdit();

  function drawOverlay() {
    const svg = $("annSvg");
    svg.replaceChildren();
    updateCursor();
    const f = aFrame();
    if (!f || !A.ready || $("annBox").classList.contains("is-empty")) return;
    const g = geom();
    const d = A.drag;
    if (!A.hideAll) {
      const muted = statusOf(f.path) === "skipped" ? " muted" : "";
      const order = A.boxes.filter((b) => !b.hidden).sort((a, b) => area(b) - area(a));
      for (const b of order) {
        const r = screenRect(b, g), c = classColor(b.cls);
        svg.append(el("svg:rect", { ...r, class: "bx-halo" }));
        svg.append(el("svg:rect", { ...r, class: "bx" + (b === A.sel ? " sel" : "") + (b === A.hover ? " hov" : "") + muted, stroke: c, fill: c }));
      }
      // Label tags: on the selected / hovered box, or on all once there are several labels.
      for (const b of order) {
        if (A.classes.length < 2 && b !== A.sel && b !== A.hover) continue;
        const r = screenRect(b, g), h = 17;
        const x = r.x, y = r.y >= h ? r.y - h : r.y; // above the box, or inside it at the top edge
        const t = el("svg:text", { x: x + 5, y: y + 12.5, class: "bx-tag" }, `${A.boxes.indexOf(b) + 1} ${className(b.cls)}`);
        svg.append(t);
        svg.insertBefore(el("svg:rect", { x, y, width: t.getComputedTextLength() + 10, height: h, rx: 3, fill: classColor(b.cls), class: "bx-tagbg" }), t);
      }
      if (A.sel && !A.sel.hidden && !(d && d.kind === "new")) {
        const r = screenRect(A.sel, g);
        for (const h of small(r) ? ["nw", "ne", "se", "sw"] : HANDLES) {
          const q = handlePos(r, h);
          svg.append(el("svg:rect", { x: q.x - 4, y: q.y - 4, width: 8, height: 8, class: "bx-handle", stroke: classColor(A.sel.cls) }));
        }
      }
    }
    if (d && d.kind === "new" && d.moved) {
      const r = screenRect({ x0: d.n0.x, y0: d.n0.y, x1: d.n1.x, y1: d.n1.y }, g);
      svg.append(el("svg:rect", { ...r, class: "bx-halo" }));
      svg.append(el("svg:rect", { ...r, class: "bx draft", stroke: classColor(A.cls), fill: classColor(A.cls) }));
    }
    // Crosshair while drawing, to line up box edges.
    const c = A.cursor;
    if (c && A.tool === "rect" && !A.space && !A.hideAll && canEdit() && (!d || d.kind === "new") && !(A.sel && handleAt(c, g))) {
      const x = Math.min(g.X(1), Math.max(g.X(0), c.x)), y = Math.min(g.Y(1), Math.max(g.Y(0), c.y));
      for (const cls of ["xhair-halo", "xhair"]) {
        svg.append(el("svg:line", { x1: g.X(0), x2: g.X(1), y1: y, y2: y, class: cls }));
        svg.append(el("svg:line", { x1: x, x2: x, y1: g.Y(0), y2: g.Y(1), class: cls }));
      }
    }
  }

  function updateCursor() {
    const svg = $("annSvg");
    const d = A.drag, c = A.cursor;
    let cur = "crosshair";
    if (d && d.kind === "move") cur = "move";
    else if (d && d.kind === "resize") cur = CURSORS[d.h];
    else if (A.tool === "pan" || A.space || A.hideAll || !canEdit()) cur = ZOOM.annBox?.s > 1 ? "grab" : "default";
    else if (c && A.ready) {
      const g = geom();
      const h = handleAt(c, g);
      if (h) cur = CURSORS[h];
      else if (A.sel && !A.sel.hidden && inside(A.sel, c, g)) cur = "move";
    }
    svg.style.cursor = cur;
  }

  function onDown(e) {
    if (!aFrame() || !A.ready) return;
    const p = local(e);
    if (A.drag) { cancelDrag(); return; } // a second finger: give up the drag
    if (panning(e)) { A.press = p; return; } // the Zoomer pans; a click still selects
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    $("annBox").focus({ preventScroll: true });
    const g = geom();
    const before = snapshot();
    const h = handleAt(p, g);
    if (h) A.drag = { kind: "resize", h, box: A.sel, orig: { ...A.sel }, start: p, before };
    else if (A.sel && !A.sel.hidden && inside(A.sel, p, g)) A.drag = { kind: "move", box: A.sel, orig: { ...A.sel }, start: p, before };
    else {
      const n0 = { x: clamp01(g.NX(p.x)), y: clamp01(g.NY(p.y)) };
      A.drag = { kind: "new", start: p, n0, n1: n0, before };
    }
    $("annSvg").setPointerCapture(e.pointerId);
  }

  function onMove(e) {
    const p = local(e);
    A.cursor = p;
    const d = A.drag;
    if (!d) {
      if (A.ready && !A.press) {
        const hov = boxAt(p, geom());
        if (hov !== A.hover) { A.hover = hov; renderRegions(); }
      }
      drawOverlay();
      return;
    }
    if (!d.moved && Math.hypot(p.x - d.start.x, p.y - d.start.y) < MIN_DRAG) return;
    d.moved = true;
    const g = geom();
    const dx = (p.x - d.start.x) / g.sw, dy = (p.y - d.start.y) / g.sh;
    const o = d.orig, b = d.box;
    if (d.kind === "new") {
      d.n1 = { x: clamp01(g.NX(p.x)), y: clamp01(g.NY(p.y)) };
    } else if (d.kind === "move") {
      const w = o.x1 - o.x0, h = o.y1 - o.y0;
      b.x0 = Math.min(1 - w, Math.max(0, o.x0 + dx));
      b.y0 = Math.min(1 - h, Math.max(0, o.y0 + dy));
      b.x1 = b.x0 + w;
      b.y1 = b.y0 + h;
    } else {
      if (d.h.includes("w")) b.x0 = clamp01(o.x0 + dx);
      if (d.h.includes("e")) b.x1 = clamp01(o.x1 + dx);
      if (d.h.includes("n")) b.y0 = clamp01(o.y0 + dy);
      if (d.h.includes("s")) b.y1 = clamp01(o.y1 + dy);
    }
    drawOverlay();
  }

  function onUp(e) {
    const d = A.drag;
    if (!d) return;
    A.drag = null;
    const p = local(e), g = geom();
    if (!d.moved) {
      // A click: select the box under the pointer (or nothing).
      A.sel = boxAt(p, g);
      return refresh();
    }
    if (d.kind === "new") {
      const r = screenRect({ x0: d.n0.x, y0: d.n0.y, x1: d.n1.x, y1: d.n1.y }, g);
      if (r.width < MIN_DRAG || r.height < MIN_DRAG) return refresh();
      const b = { id: ++A.uid, cls: A.cls, x0: Math.min(d.n0.x, d.n1.x), y0: Math.min(d.n0.y, d.n1.y), x1: Math.max(d.n0.x, d.n1.x), y1: Math.max(d.n0.y, d.n1.y), hidden: false };
      A.boxes.push(b);
      A.sel = b;
      return commit(d.before);
    }
    const b = d.box;
    [b.x0, b.x1] = [Math.min(b.x0, b.x1), Math.max(b.x0, b.x1)];
    [b.y0, b.y1] = [Math.min(b.y0, b.y1), Math.max(b.y0, b.y1)];
    if (b.x1 - b.x0 < 1e-4 || b.y1 - b.y0 < 1e-4) Object.assign(b, d.orig); // collapsed: undo the resize
    commit(d.before);
  }

  function cancelDrag() {
    const d = A.drag;
    A.drag = null;
    if (d && d.box) Object.assign(d.box, d.orig);
    drawOverlay();
  }

  // ------------------------------------------------------------ keyboard
  function onKey(e) {
    if (TAB !== "annotate") return;
    if (e.target.closest("textarea, select, input:not([type=checkbox])")) return;
    const k = e.key, low = k.length === 1 ? k.toLowerCase() : k, ctrl = e.ctrlKey || e.metaKey;
    const act = (fn) => { e.preventDefault(); fn(); };
    if (ctrl) {
      if (k === "Enter") return act(submit);
      if (k === " ") return act(skip);
      if (low === "z") return act(e.shiftKey ? redo : undo);
      if (low === "y") return act(redo);
      if (low === "h") return act(toggleHideAll);
      if (low === "d") return act(duplicate);
      if (low === "c" && !String(getSelection())) return act(copy);
      if (low === "v") return act(paste);
      if (k === "Backspace" || k === "Delete") return act(deleteAll);
      if (k === "ArrowLeft" || k === "ArrowRight") return act(() => step(k === "ArrowLeft" ? -1 : 1));
      return; // leave other browser shortcuts alone
    }
    if (e.altKey) {
      if (k === "Enter") return act(skip);
      if (e.code === "Period") return act(cycle);
      if (e.code === "KeyH") return act(toggleHideSel);
      return;
    }
    if (k === " ") {
      e.preventDefault(); // no page scroll or button press
      if (!A.space) { A.space = true; updateCursor(); drawOverlay(); }
      return;
    }
    if (e.shiftKey && e.code === "Digit1") return act(() => ZOOM.annBox.reset());
    if (e.shiftKey && e.code === "Digit2") return act(() => ZOOM.annBox.actualPixels());
    if (!e.shiftKey && /^Digit[1-9]$/.test(e.code)) return act(() => chooseClass(+e.code.slice(5) - 1));
    switch (low) {
      case "a": return act(() => step(-1, e.shiftKey ? "unlabeled" : A.filter));
      case "d": return act(() => step(1, e.shiftKey ? "unlabeled" : A.filter));
      case "ArrowLeft": return act(() => (e.shiftKey ? stepDay(-1) : step(-1)));
      case "ArrowRight": return act(() => (e.shiftKey ? stepDay(1) : step(1)));
      case "Backspace": case "Delete": return act(deleteSel);
      case "Escape": if (A.drag) { cancelDrag(); return; } return unselect();
      case "u": return act(unselect);
      case "r": return act(() => setTool("rect"));
      case "h": return act(() => setTool("pan"));
      case "c": return act(copyPrev);
      case "f": return act(() => ZOOM.annBox.toggleFullscreen());
      case "+": case "=": return act(() => ZOOM.annBox.zoomCenter(1.6));
      case "-": case "_": return act(() => ZOOM.annBox.zoomCenter(1 / 1.6));
      case "0": return act(() => ZOOM.annBox.reset());
    }
  }

  function onKeyUp(e) {
    if (e.key === " " && A.space) {
      e.preventDefault();
      A.space = false;
      drawOverlay();
    }
  }

  // -------------------------------------------------------------- bind
  function bind() {
    const box = $("annBox"), svg = $("annSvg");
    const z = (ZOOM.annBox = new Zoomer(box, $("annImg")));
    z.onApply = drawOverlay;
    z.onLoad = () => { A.ready = true; drawOverlay(); };
    $("annImg").addEventListener("error", () => toast("This frame could not be loaded."));

    svg.addEventListener("pointerdown", onDown);
    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerup", onUp);
    svg.addEventListener("pointercancel", cancelDrag);
    svg.addEventListener("pointerleave", () => {
      if (A.drag) return;
      A.cursor = null;
      if (A.hover) { A.hover = null; renderRegions(); }
      drawOverlay();
    });
    svg.addEventListener("dblclick", (e) => { if (boxAt(local(e), geom())) e.stopPropagation(); }); // no zoom on a box
    // Pan tool: the Zoomer owns the drag; a click without movement selects.
    box.addEventListener("pointerup", (e) => {
      const p0 = A.press;
      A.press = null;
      if (!p0 || !A.ready) return;
      const p = local(e);
      if (Math.hypot(p.x - p0.x, p.y - p0.y) < MIN_DRAG) { A.sel = boxAt(p, geom()); refresh(); }
    });

    document.querySelectorAll("#annTools [data-tool]").forEach((b) => { b.onclick = (e) => { e.stopPropagation(); setTool(b.dataset.tool); }; });
    $("annHideAll").onclick = (e) => { e.stopPropagation(); toggleHideAll(); };
    $("annTools").addEventListener("pointerdown", (e) => e.stopPropagation());
    $("annTools").addEventListener("dblclick", (e) => e.stopPropagation());

    $("annField").onchange = (e) => {
      const key = e.target.value;
      const inField = visible().filter((s) => fieldKey(s) === key);
      switchSource(inField.find((s) => s.camera === A.src?.camera && s.view === A.src?.view) || inField.find((s) => s.view === A.src?.view) || inField[0]);
      e.target.blur();
    };
    $("annDay").onchange = (e) => { A.dayOnly = e.target.checked; rescope(); };
    $("annAll").onchange = (e) => { A.showAll = e.target.checked; rescope(); };
    $("annPrev").onclick = () => step(-1);
    $("annNext").onclick = () => step(1);
    $("annSubmit").onclick = submit;
    $("annSkip").onclick = skip;
    $("annReset").onclick = reset;
    $("annCopyPrev").onclick = copyPrev;
    $("annDeleteAll").onclick = deleteAll;
    $("annAddLabel").onclick = addClass;

    const cv = $("annScrub");
    cv.addEventListener("click", (e) => { if (A.src && A.list.length) openFrame(A.src, scrubIndex(e)); });
    cv.addEventListener("mousemove", (e) => {
      if (!A.src || !A.list.length) return;
      const f = A.src.frames[scrubIndex(e)];
      const st = statusOf(f.path), r = A.recs.get(f.path);
      const tip = el("div");
      tip.append(el("div", { class: "t-head" }, `${fmtDay(f.date, true)} · ${fmtHM(f.min)}`));
      const row = el("div", { class: "t-row" });
      row.append(el("i", { style: `background:${STATUS[st].color}` }), STATUS[st].label + (r && r.boxes.length ? ` · ${plural(r.boxes.length, "box", "boxes")}` : ""));
      tip.append(row);
      showTip(e, tip);
    });
    cv.addEventListener("mouseleave", hideTip);
    new ResizeObserver(() => { drawScrub(); renderAxis(); }).observe(cv);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawScrub);

    document.addEventListener("keydown", onKey);
    document.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", () => { A.space = false; });
    // Don't lose the last edit when the tab closes.
    window.addEventListener("pagehide", () => flushSave(true));
    document.addEventListener("visibilitychange", () => { if (document.hidden) flushSave(true); });
  }

  function toast(msg) {
    const t = $("annToast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(A.toastTimer);
    A.toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
  }

  return { show };
})();
