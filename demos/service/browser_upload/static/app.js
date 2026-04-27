// browser_upload — vanilla JS WS client
//
// Wire protocol mirrors demos/streaming/ws_client.py:
//   client → server : start, segment, ping, abort
//   server → client : started, progress, final, pong, closed

const $ = (id) => document.getElementById(id);
const drop = $("drop");
const fileInput = $("file");
const srtArea = $("srt");
const startBtn = $("start");
const abortBtn = $("abort");
const statusEl = $("status");
const logEl = $("log");
const gridBody = document.querySelector("#grid tbody");

let ws = null;
let rowsBySeq = new Map();

function setStatus(msg, cls = "") {
  statusEl.textContent = msg;
  statusEl.className = "status " + cls;
}

function logFrame(direction, frame) {
  const cls = direction === "▶" ? "frame-out" : direction === "✗" ? "frame-err" : "frame-in";
  const line = document.createElement("div");
  line.className = cls;
  line.textContent = `${direction} ${JSON.stringify(frame)}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function clearGrid() {
  gridBody.innerHTML = "";
  rowsBySeq.clear();
}

function upsertRow(seq, start, source, translation) {
  let row = rowsBySeq.get(seq);
  if (!row) {
    row = document.createElement("tr");
    row.innerHTML = `<td>${seq}</td><td></td><td></td><td></td>`;
    gridBody.appendChild(row);
    rowsBySeq.set(seq, row);
  }
  if (start !== undefined) row.children[1].textContent = Number(start).toFixed(2);
  if (source !== undefined) row.children[2].textContent = source;
  if (translation !== undefined) row.children[3].textContent = translation;
}

// ---------- SRT parsing ----------

function parseSrt(text) {
  // Returns [{seq, start, end, text}, …]
  const blocks = text.replace(/\r\n/g, "\n").split(/\n\s*\n/);
  const out = [];
  const tsToSec = (ts) => {
    const m = ts.match(/(\d+):(\d+):(\d+)[,.](\d+)/);
    if (!m) return 0;
    return (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]) + (+m[4]) / 1000;
  };
  let seq = 0;
  for (const blk of blocks) {
    const lines = blk.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.length < 2) continue;
    const tsLine = lines.find((l) => l.includes("-->"));
    if (!tsLine) continue;
    const [from, to] = tsLine.split("-->").map((s) => s.trim());
    const txt = lines
      .filter((l) => !/^\d+$/.test(l) && !l.includes("-->"))
      .join(" ");
    if (!txt) continue;
    seq += 1;
    out.push({ seq, start: tsToSec(from), end: tsToSec(to), text: txt });
  }
  return out;
}

// ---------- Drag & drop ----------

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const file = e.dataTransfer.files[0];
  if (file) {
    const txt = await file.text();
    srtArea.value = txt;
    setStatus(`loaded ${file.name} (${txt.length} chars)`);
  }
});
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (file) {
    const txt = await file.text();
    srtArea.value = txt;
    setStatus(`loaded ${file.name} (${txt.length} chars)`);
  }
});

// ---------- Start / abort ----------

startBtn.addEventListener("click", () => {
  const segments = parseSrt(srtArea.value);
  if (segments.length === 0) {
    setStatus("no segments parsed — check your SRT input", "frame-err");
    return;
  }

  clearGrid();
  logEl.textContent = "";

  const apiKey = $("apiKey").value.trim();
  const course = $("course").value.trim() || "browser-demo";
  const video = $("video").value.trim() || "lec-01";
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/api/ws/streams?access_token=${encodeURIComponent(apiKey)}`;

  setStatus(`connecting…`);
  ws = new WebSocket(url);

  ws.addEventListener("open", () => {
    setStatus("connected, opening stream");
    const startFrame = {
      type: "start",
      pipeline: "live_translate_zh",
      course,
      video,
      src: "en",
      tgt: "zh",
    };
    ws.send(JSON.stringify(startFrame));
    logFrame("▶", startFrame);
  });

  ws.addEventListener("message", (ev) => {
    let frame;
    try { frame = JSON.parse(ev.data); } catch { return; }
    logFrame("◀", frame);

    if (frame.type === "started") {
      setStatus(`streaming • stream_id=${frame.stream_id}`);
      // Seed the grid + push all segments
      for (const seg of segments) {
        upsertRow(seg.seq, seg.start, seg.text, "…");
        const f = {
          type: "segment",
          seq: seg.seq,
          start: seg.start,
          end: seg.end,
          text: seg.text,
        };
        ws.send(JSON.stringify(f));
      }
      logFrame("▶", `${segments.length} segment frames sent`);
      abortBtn.disabled = false;
    } else if (frame.type === "progress") {
      // progress emits stage/channel_fill — we display nothing here.
    } else if (frame.type === "final") {
      // Real schema: {type:'final', record_id:'rec-N', src, tgt, start, end}
      const seq = parseInt(String(frame.record_id || "").replace("rec-", ""), 10);
      // record_id is 0-indexed; UI seq is 1-indexed
      const uiSeq = Number.isFinite(seq) ? seq + 1 : (frame.seq ?? frame.id);
      if (uiSeq != null) {
        upsertRow(uiSeq, frame.start, frame.src, frame.tgt || frame.translation || "");
      }
    } else if (frame.type === "closed") {
      setStatus("closed");
      abortBtn.disabled = true;
    }
  });

  ws.addEventListener("error", (ev) => {
    setStatus("error — see console", "frame-err");
    logFrame("✗", { error: String(ev) });
  });
  ws.addEventListener("close", (ev) => {
    setStatus(`closed (${ev.code})`);
    abortBtn.disabled = true;
    ws = null;
  });
});

abortBtn.addEventListener("click", () => {
  if (!ws) return;
  const f = { type: "abort" };
  ws.send(JSON.stringify(f));
  logFrame("▶", f);
  setStatus("aborting…");
});
