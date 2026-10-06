"use strict";

// Model tab: the live log of a training run started with yolo/pipeline.py.
// The pipeline writes yolo/logs/<stamp>_<command>_<experiment>.log plus a
// .json status; the server only reads them (/api/model/runs, /api/model/log).
// A log line rewritten in place (a progress bar) is "\r<new text>".
const ModelTab = (() => {
  const POLL_MS = 2000;   // new log text
  const RUNS_MS = 5000;   // list of runs (a new run is opened automatically)
  const STATES = {
    running: { label: "Running", color: "var(--stage-onset)" },
    finished: { label: "Finished", color: "var(--ann-labeled)" },
    failed: { label: "Failed", color: "var(--stage-bloom)" },
    stopped: { label: "Stopped", color: "var(--muted)" },
    lost: { label: "Ended (no exit status)", color: "var(--muted)" },
    unknown: { label: "Unknown", color: "var(--muted)" },
  };
  const M = {
    active: false, timer: 0, busy: false, runsAt: 0,
    runs: [], dir: "", skew: 0,  // skew: server clock - browser clock (ms)
    name: null, pinned: false,   // pinned: picked by hand, so a newer run doesn't replace it
    next: -1, cur: "", status: null, follow: true,
    doneNode: null, curNode: null,
  };

  function show(run) {
    M.active = true;
    if (run) { M.name = run; M.pinned = true; M.next = -1; }
    writeHash();
    poll(true);
  }

  function hide() {
    M.active = false;
    clearTimeout(M.timer);
  }

  async function poll(force) {
    clearTimeout(M.timer);
    if (!M.active) return;
    if (!M.busy) {
      M.busy = true;
      try {
        if (force || Date.now() - M.runsAt > RUNS_MS) await loadRuns();
        if (M.name) await loadLog();
        $("modelError").hidden = true;
      } catch (err) {
        $("modelError").textContent = `Could not read the training logs: ${err.message}`;
        $("modelError").hidden = false;
      } finally {
        M.busy = false;
      }
    }
    if (M.active) M.timer = setTimeout(poll, document.hidden ? POLL_MS * 5 : POLL_MS);
  }

  async function getJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
    if (body.now) M.skew = body.now * 1000 - Date.now();
    return body;
  }

  // ----------------------------------------------------------------- runs
  async function loadRuns() {
    const body = await getJson("/api/model/runs");
    M.runsAt = Date.now();
    M.runs = body.runs;
    M.dir = body.dir;
    if (M.name && !M.runs.some((r) => r.name === M.name)) { M.name = null; M.pinned = false; }
    const want = M.pinned ? M.name : M.runs[0]?.name || null;  // follow the newest run
    if (want !== M.name || M.next < 0) open(want);
    renderRuns();
  }

  function runLabel(r) {
    const m = r.name.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_/);
    const when = m ? `${fmtDay(`${m[1]}-${m[2]}-${m[3]}`)} ${m[4]}:${m[5]}` : r.name;
    return `${when} · ${r.command || "?"} · ${r.experiment || ""}${r.state === "running" ? " (running)" : ""}`;
  }

  function renderRuns() {
    const sel = $("modelRun");
    sel.replaceChildren(...M.runs.map((r) => el("option", { value: r.name }, runLabel(r))));
    sel.value = M.name || "";
    const none = !M.runs.length;
    $("modelEmpty").hidden = !none;
    $("modelLogCard").hidden = none;
    $("modelHead").hidden = none;
    $("modelDir").textContent = M.dir;
  }

  function open(name) {
    M.name = name;
    M.next = -1;
    M.cur = "";
    M.status = null;
    const pre = $("modelLog");
    M.doneNode = document.createTextNode("");
    M.curNode = document.createTextNode("");
    pre.replaceChildren(M.doneNode, M.curNode);
    $("modelSkipped").hidden = true;
    M.follow = true;
    $("modelFollow").checked = true;
  }

  // ------------------------------------------------------------------ log
  async function loadLog() {
    // Read until caught up (a long log arrives in chunks).
    for (let i = 0; i < 20; i++) {
      const name = M.name;
      const body = await getJson(`/api/model/log?name=${encodeURIComponent(name)}&offset=${M.next}`);
      if (name !== M.name) return;  // another run was picked meanwhile
      if (M.next >= 0 && body.offset !== M.next) open(name);  // log was replaced: start over
      if (body.skipped) {
        $("modelSkipped").textContent = `The first ${fmtBytes(body.skipped)} of this log are not shown. Full log: ${M.dir}/${name}`;
        $("modelSkipped").hidden = false;
      }
      append(body.text);
      M.next = body.next;
      M.status = body.status;
      renderStatus();
      if (body.next >= body.size || !body.text) break;
    }
  }

  function append(text) {
    if (!text) return;
    let done = "";
    for (const part of text.split(/(\r\n|\r|\n)/)) {
      if (part === "\n" || part === "\r\n") { done += M.cur + "\n"; M.cur = ""; }
      else if (part === "\r") M.cur = "";
      else M.cur += part;
    }
    if (done) M.doneNode.appendData(done);
    M.curNode.data = M.cur;
    if (M.follow) scrollToEnd();
  }

  function scrollToEnd() {
    const pre = $("modelLog");
    pre.scrollTop = pre.scrollHeight;
  }

  function renderStatus() {
    const st = M.status || {};
    const s = STATES[st.state] || STATES.unknown;
    $("modelPill").replaceChildren(el("i", { style: `background:${s.color}` }), s.label);
    const now = Date.now() + M.skew;
    const started = Date.parse(st.started);
    const ended = st.ended ? Date.parse(st.ended) : st.state === "running" ? now : (st.modified || 0) * 1000;
    const bits = [];
    if (st.argv) bits.push(`pipeline.py ${st.argv.join(" ")}`);
    if (started) bits.push(`${st.state === "running" ? "running for" : "ran"} ${fmtDur((ended - started) / 60000)}`);
    if (st.state === "running" && st.modified) {
      const quiet = (now - st.modified * 1000) / 60000;
      if (quiet >= 2) bits.push(`no output for ${fmtDur(quiet)}`);
    }
    if (st.exit_code !== null && st.exit_code !== undefined && st.state !== "finished") bits.push(`exit code ${st.exit_code}`);
    if (st.pid && st.state === "running") bits.push(`pid ${st.pid}`);
    $("modelMeta").textContent = bits.join(" · ");
  }

  const fmtBytes = (n) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`);

  function writeHash() {
    const q = new URLSearchParams({ tab: "model" });
    if (M.pinned && M.name) q.set("run", M.name);
    history.replaceState(null, "", "#" + q.toString());
  }

  // ----------------------------------------------------------------- init
  $("modelRun").onchange = (e) => {
    M.pinned = e.target.value !== M.runs[0]?.name;  // picking the newest goes back to following new runs
    open(e.target.value);
    writeHash();
    poll(false);
  };
  $("modelFollow").onchange = (e) => {
    M.follow = e.target.checked;
    if (M.follow) scrollToEnd();
  };
  $("modelLog").addEventListener("scroll", () => {
    const pre = $("modelLog");
    const atEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    if (atEnd !== M.follow) {
      M.follow = atEnd;
      $("modelFollow").checked = atEnd;
    }
  });

  return { show, hide };
})();
