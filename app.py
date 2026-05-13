#!/usr/bin/env python3
"""
YouTube Text Remover — Flask Web Application
"""
import os, sys, json, uuid, glob, threading, subprocess, time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string
import cv2
import numpy as np
import imageio_ffmpeg

# ─── Config ───────────────────────────────────────────────────────────────────
FFMPEG     = imageio_ffmpeg.get_ffmpeg_exe()
BASE_DIR   = "/home/user/webapp"
JOBS_DIR   = os.path.join(BASE_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# ── Make ffmpeg available to yt-dlp ──────────────────────────────────────────
_FFMPEG_WRAP = "/home/user/ffmpeg_bin/ffmpeg"
os.makedirs("/home/user/ffmpeg_bin", exist_ok=True)
if not os.path.exists(_FFMPEG_WRAP):
    with open(_FFMPEG_WRAP, "w") as _f:
        _f.write(f"#!/bin/bash\nexec '{FFMPEG}' \"$@\"\n")
    os.chmod(_FFMPEG_WRAP, 0o755)
# Prepend to PATH so yt-dlp picks it up
os.environ["PATH"] = "/home/user/ffmpeg_bin:" + os.environ.get("PATH", "")

app = Flask(__name__)

# ─── In-memory job store ───────────────────────────────────────────────────────
jobs = {}   # job_id -> { status, progress, message, output_path, title, error }


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def get_best_format_string(url: str):
    """
    Returns (format_string, title, thumbnail).
    Uses yt-dlp's own format selector: best h264 video ≤720p + best audio.
    Falls back to 'best' single-file if no separate streams found.
    """
    r = subprocess.run(
        ["yt-dlp", "--print", "%(format_id)s\t%(title)s\t%(thumbnail)s",
         "--no-playlist", "-f",
         "bestvideo[vcodec^=avc1][height<=720]+bestaudio/bestvideo[height<=720]+bestaudio/best",
         url],
        capture_output=True, text=True, timeout=60
    )
    if r.returncode != 0:
        raise RuntimeError("ไม่สามารถดึงข้อมูลวีดีโอได้ กรุณาตรวจสอบ URL อีกครั้ง")
    parts = r.stdout.strip().split("\t")
    fmt_id   = parts[0] if parts else "best"
    title    = parts[1] if len(parts) > 1 else "video"
    thumb    = parts[2] if len(parts) > 2 else ""
    return fmt_id, title, thumb


def detect_text_mask(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, bright = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    _, dark   = cv2.threshold(gray, 35,  255, cv2.THRESH_BINARY_INV)
    zone = np.zeros_like(gray)
    zone[int(h*0.70):, :]              = 255
    zone[:int(h*0.12), :]              = 255
    zone[:int(h*0.25), :int(w*0.35)]   = 255
    zone[:int(h*0.25), int(w*0.65):]   = 255
    zone[int(h*0.75):, :int(w*0.35)]   = 255
    zone[int(h*0.75):, int(w*0.65):]   = 255
    combined = cv2.bitwise_or(bright, dark)
    in_zone  = cv2.bitwise_and(combined, zone)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated  = cv2.dilate(in_zone, kernel, iterations=2)
    final    = np.zeros_like(gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i, :5]
        aspect = bw / (bh + 1e-5)
        if 0.2 < aspect < 40 and 30 < area < 40000 and bh < h * 0.15:
            final[y:y+bh, x:x+bw] = 255
    return final


def process_job(job_id: str, url: str):
    job = jobs[job_id]
    temp_dir = os.path.join(JOBS_DIR, job_id, "temp")
    out_dir  = os.path.join(JOBS_DIR, job_id, "out")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)

    # Ensure ffmpeg is on PATH for this thread/subprocess
    _env = os.environ.copy()
    _env["PATH"] = "/home/user/ffmpeg_bin:" + _env.get("PATH", "")

    def run(*cmd, **kwargs):
        return subprocess.run(list(cmd), capture_output=True, text=True, env=_env, **kwargs)

    def update(status, progress, message):
        jobs[job_id].update({"status": status, "progress": progress, "message": message})

    try:
        # ── Step 1: Get video info + Download (single step via yt-dlp merge) ──
        update("running", 5, "🔍 กำลังดึงข้อมูลวีดีโอ...")

        # Get title & thumbnail first
        info_r = run("yt-dlp", "--dump-json", "--no-playlist", url, timeout=60)
        if info_r.returncode == 0:
            info = json.loads(info_r.stdout)
            title = info.get("title", "video")
            thumb = info.get("thumbnail", "")
        else:
            title, thumb = "video", ""
        jobs[job_id]["title"] = title
        jobs[job_id]["thumbnail"] = thumb

        # ── Step 2: Download video+audio in one call ───────────────────────
        update("running", 10, f"📥 กำลังดาวน์โหลด: {title[:40]}...")
        merged = os.path.join(temp_dir, "merged.mp4")
        r = run("yt-dlp", "-f", "bestvideo[vcodec^=avc1][height<=720]+bestaudio/bestvideo[height<=720]+bestaudio/best", "--merge-output-format", "mp4", "--no-playlist", "-o", merged, url, timeout=600)
        if r.returncode != 0 or not os.path.exists(merged):
            # fallback: just best
            r2 = run("yt-dlp", "-f", "best", "--merge-output-format", "mp4", "--no-playlist", "-o", merged, url, timeout=600)
            if r2.returncode != 0:
                raise RuntimeError("ดาวน์โหลดวีดีโอไม่สำเร็จ")

        # ── Step 3: Strip soft subs ────────────────────────────────────────
        update("running", 30, "✂️ กำลังลบ subtitle tracks...")
        stripped = os.path.join(temp_dir, "stripped.mp4")
        r = run(FFMPEG, "-y", "-i", merged, "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-sn", "-map_metadata", "-1", stripped)
        if r.returncode != 0 or not os.path.exists(stripped):
            stripped = merged  # fallback: use merged directly
        merged = stripped

        # ── Step 4: Inpaint hardcoded text ───────────────────────────────
        update("running", 35, "🎨 กำลังวิเคราะห์และลบ text...")
        cap   = cv2.VideoCapture(merged)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_path = os.path.join(temp_dir, "frames.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(frames_path, fourcc, fps, (w, h))

        idx, last_mask = 0, None
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % 2 == 0:
                mask = detect_text_mask(frame)
                last_mask = mask
            else:
                mask = last_mask if last_mask is not None else np.zeros((h, w), np.uint8)
            cleaned = cv2.inpaint(frame, mask, 4, cv2.INPAINT_TELEA) if mask.max() > 0 else frame
            writer.write(cleaned)
            idx += 1
            if total > 0 and idx % 30 == 0:
                pct = 35 + int((idx / total) * 50)
                update("running", pct, f"🎨 ลบ text... {idx}/{total} frames ({idx/total*100:.0f}%)")

        cap.release()
        writer.release()

        # ── Step 5: Final encode + mux audio ─────────────────────────────
        update("running", 88, "🎬 กำลัง encode วีดีโอสุดท้าย...")
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:40].strip()
        final_path = os.path.join(out_dir, f"{safe}_clean.mp4")

        r = run(FFMPEG, "-y", "-i", frames_path, "-i", merged, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", final_path)
        if r.returncode != 0:
            raise RuntimeError("Export ไฟล์ไม่สำเร็จ")

        size_mb = os.path.getsize(final_path) / 1024 / 1024
        jobs[job_id]["output_path"] = final_path
        jobs[job_id]["size_mb"] = round(size_mb, 2)
        update("done", 100, f"✅ เสร็จสมบูรณ์! ({size_mb:.1f} MB)")

    except Exception as e:
        jobs[job_id]["error"] = str(e)
        update("error", 0, f"❌ เกิดข้อผิดพลาด: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.get_json()
    url  = (data or {}).get("url", "").strip()
    if not url or "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "กรุณาใส่ URL YouTube ที่ถูกต้อง"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "รอดำเนินการ...",
                    "title": "", "thumbnail": "", "output_path": None, "error": None}
    t = threading.Thread(target=process_job, args=(job_id, url), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "ไม่พบ job"}), 404
    return jsonify(job)


@app.route("/api/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or not job.get("output_path"):
        return jsonify({"error": "ไฟล์ยังไม่พร้อม"}), 404
    path = job["output_path"]
    if not os.path.exists(path):
        return jsonify({"error": "ไม่พบไฟล์"}), 404
    return send_file(path, as_attachment=True,
                     download_name=os.path.basename(path),
                     mimetype="video/mp4")


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>YouTube Text Remover</title>
<style>
  :root {
    --bg: #0f0f13;
    --card: #1a1a23;
    --card2: #22222e;
    --border: #2e2e3d;
    --accent: #7c3aed;
    --accent2: #a855f7;
    --green: #22c55e;
    --red: #ef4444;
    --text: #e2e2f0;
    --muted: #8888aa;
    --radius: 16px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px 80px;
  }

  /* ── Header ── */
  .header {
    text-align: center;
    margin-bottom: 40px;
  }
  .logo {
    width: 72px; height: 72px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 36px;
    margin: 0 auto 20px;
    box-shadow: 0 8px 32px rgba(124,58,237,.4);
  }
  h1 { font-size: 2rem; font-weight: 800; letter-spacing: -.5px; }
  h1 span { background: linear-gradient(90deg,#a855f7,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .subtitle { color: var(--muted); margin-top: 8px; font-size: .95rem; }

  /* ── Input Card ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    width: 100%;
    max-width: 640px;
    margin-bottom: 24px;
  }

  .input-row {
    display: flex;
    gap: 12px;
  }
  .url-input {
    flex: 1;
    background: var(--card2);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    color: var(--text);
    font-size: 1rem;
    padding: 14px 18px;
    outline: none;
    transition: border-color .2s;
  }
  .url-input:focus { border-color: var(--accent); }
  .url-input::placeholder { color: var(--muted); }

  .btn-submit {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none;
    border-radius: 12px;
    color: #fff;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 700;
    padding: 14px 28px;
    transition: opacity .2s, transform .1s;
    white-space: nowrap;
  }
  .btn-submit:hover { opacity: .9; }
  .btn-submit:active { transform: scale(.97); }
  .btn-submit:disabled { opacity: .5; cursor: not-allowed; }

  .input-note {
    color: var(--muted);
    font-size: .82rem;
    margin-top: 12px;
  }

  /* ── Features ── */
  .features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 24px;
  }
  .feat {
    background: var(--card2);
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
    font-size: .82rem;
    color: var(--muted);
  }
  .feat .icon { font-size: 1.4rem; margin-bottom: 6px; }
  .feat strong { display: block; color: var(--text); font-size: .88rem; margin-bottom: 2px; }

  /* ── Progress Card ── */
  #job-card {
    display: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    width: 100%;
    max-width: 640px;
    margin-bottom: 24px;
  }
  #job-card.visible { display: block; }

  .job-header {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 24px;
  }
  #thumb {
    width: 96px; height: 60px;
    border-radius: 8px;
    object-fit: cover;
    background: var(--card2);
    flex-shrink: 0;
  }
  .job-meta h3 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 4px;
    line-height: 1.3;
  }
  .job-meta .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: .75rem;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 600;
  }
  .badge.queued  { background: #27272e; color: var(--muted); }
  .badge.running { background: rgba(124,58,237,.2); color: var(--accent2); }
  .badge.done    { background: rgba(34,197,94,.15); color: var(--green); }
  .badge.error   { background: rgba(239,68,68,.15); color: var(--red); }

  .spinner {
    width: 10px; height: 10px;
    border: 2px solid transparent;
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin .7s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .progress-wrap {
    background: var(--card2);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 10px;
  }
  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 999px;
    transition: width .4s ease;
  }
  .progress-text {
    display: flex;
    justify-content: space-between;
    font-size: .82rem;
    color: var(--muted);
    margin-bottom: 18px;
  }
  .step-msg {
    font-size: .9rem;
    color: var(--text);
    min-height: 1.3em;
  }

  /* ── Steps indicator ── */
  .steps {
    display: flex;
    gap: 0;
    margin: 20px 0;
    position: relative;
  }
  .steps::before {
    content: '';
    position: absolute;
    top: 16px;
    left: 16px; right: 16px;
    height: 2px;
    background: var(--border);
    z-index: 0;
  }
  .step {
    flex: 1;
    text-align: center;
    position: relative;
    z-index: 1;
  }
  .step-dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: var(--card2);
    border: 2px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 6px;
    font-size: .8rem;
    transition: all .3s;
  }
  .step.active .step-dot  { background: var(--accent); border-color: var(--accent); }
  .step.done .step-dot    { background: var(--green);  border-color: var(--green); }
  .step-label { font-size: .72rem; color: var(--muted); }
  .step.active .step-label { color: var(--accent2); }
  .step.done .step-label   { color: var(--green); }

  /* ── Download ── */
  .btn-download {
    display: block;
    width: 100%;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    border: none;
    border-radius: 12px;
    color: #fff;
    cursor: pointer;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 16px;
    margin-top: 20px;
    text-align: center;
    text-decoration: none;
    transition: opacity .2s;
  }
  .btn-download:hover { opacity: .9; }

  /* ── History ── */
  #history {
    width: 100%;
    max-width: 640px;
  }
  #history h2 {
    font-size: .9rem;
    color: var(--muted);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .hist-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  .hist-thumb {
    width: 64px; height: 40px;
    border-radius: 6px;
    object-fit: cover;
    background: var(--card2);
    flex-shrink: 0;
  }
  .hist-info { flex: 1; min-width: 0; }
  .hist-info .title { font-size: .88rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .hist-info .size  { font-size: .75rem; color: var(--muted); }
  .hist-dl {
    background: rgba(34,197,94,.15);
    border: 1px solid rgba(34,197,94,.3);
    border-radius: 8px;
    color: var(--green);
    font-size: .8rem;
    font-weight: 600;
    padding: 6px 14px;
    text-decoration: none;
    white-space: nowrap;
  }
  .hist-dl:hover { background: rgba(34,197,94,.25); }

  /* ── Error ── */
  .err-box {
    background: rgba(239,68,68,.1);
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 10px;
    color: var(--red);
    font-size: .88rem;
    padding: 14px 16px;
    margin-top: 16px;
  }

  @media (max-width: 480px) {
    .input-row { flex-direction: column; }
    .features  { grid-template-columns: 1fr 1fr; }
    h1 { font-size: 1.5rem; }
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="logo">🎬</div>
  <h1>YouTube <span>Text Remover</span></h1>
  <p class="subtitle">ลบ subtitle, watermark และ hardcoded text ออกจากวีดีโอโดยอัตโนมัติ</p>
</div>

<!-- Input Card -->
<div class="card">
  <div class="input-row">
    <input id="url-input" class="url-input" type="text"
           placeholder="https://www.youtube.com/watch?v=..."
           autocomplete="off" spellcheck="false"/>
    <button id="btn-start" class="btn-submit" onclick="startJob()">▶ เริ่ม</button>
  </div>
  <p class="input-note">⚠️ รองรับเฉพาะ YouTube • รองรับวีดีโอความยาว ≤ 30 นาที เพื่อประสิทธิภาพที่ดีที่สุด</p>

  <div class="features">
    <div class="feat"><div class="icon">📝</div><strong>Soft Subtitles</strong>ลบ caption tracks ทั้งหมด</div>
    <div class="feat"><div class="icon">🎨</div><strong>Hardcoded Text</strong>AI inpainting ลบ text/watermark</div>
    <div class="feat"><div class="icon">📥</div><strong>ดาวน์โหลดได้ทันที</strong>MP4 คุณภาพสูง พร้อมเสียง</div>
  </div>
</div>

<!-- Job Progress Card -->
<div id="job-card">
  <div class="job-header">
    <img id="thumb" src="" alt="thumbnail"/>
    <div class="job-meta">
      <h3 id="job-title">กำลังประมวลผล...</h3>
      <span id="badge" class="badge running"><span class="spinner"></span>&nbsp;กำลังทำงาน</span>
    </div>
  </div>

  <!-- Steps -->
  <div class="steps">
    <div class="step" id="s1"><div class="step-dot">📥</div><div class="step-label">ดาวน์โหลด</div></div>
    <div class="step" id="s2"><div class="step-dot">✂️</div><div class="step-label">ลบ Subs</div></div>
    <div class="step" id="s3"><div class="step-dot">🎨</div><div class="step-label">Inpaint</div></div>
    <div class="step" id="s4"><div class="step-dot">🎬</div><div class="step-label">Export</div></div>
  </div>

  <div class="progress-wrap"><div class="progress-bar" id="pbar" style="width:0%"></div></div>
  <div class="progress-text">
    <span id="step-msg">รอดำเนินการ...</span>
    <span id="pct">0%</span>
  </div>

  <div id="error-box" class="err-box" style="display:none"></div>
  <a id="btn-dl" class="btn-download" style="display:none" href="#">⬇ ดาวน์โหลดวีดีโอ</a>
</div>

<!-- History -->
<div id="history"></div>

<script>
let currentJobId = null;
let pollTimer = null;
const history = [];

function startJob() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;

  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-dl').style.display = 'none';
  document.getElementById('error-box').style.display = 'none';
  document.getElementById('pbar').style.width = '0%';
  document.getElementById('pct').textContent = '0%';
  document.getElementById('step-msg').textContent = 'กำลังส่งงาน...';
  document.getElementById('job-title').textContent = 'กำลังประมวลผล...';
  document.getElementById('thumb').src = '';
  setBadge('running', 'กำลังทำงาน');
  ['s1','s2','s3','s4'].forEach(s => {
    document.getElementById(s).className = 'step';
  });
  document.getElementById('job-card').className = 'card visible';

  fetch('/api/submit', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({url})
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { showError(data.error); return; }
    currentJobId = data.job_id;
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollStatus(data.job_id), 1500);
  })
  .catch(e => showError('เชื่อมต่อ server ไม่สำเร็จ'));
}

function pollStatus(jobId) {
  fetch('/api/status/' + jobId)
  .then(r => r.json())
  .then(job => {
    const pct = job.progress || 0;
    document.getElementById('pbar').style.width = pct + '%';
    document.getElementById('pct').textContent = pct + '%';
    document.getElementById('step-msg').textContent = job.message || '';
    if (job.title) document.getElementById('job-title').textContent = job.title;
    if (job.thumbnail) document.getElementById('thumb').src = job.thumbnail;

    updateSteps(pct);

    if (job.status === 'done') {
      clearInterval(pollTimer);
      setBadge('done', '✅ เสร็จแล้ว');
      const dlBtn = document.getElementById('btn-dl');
      dlBtn.href = '/api/download/' + jobId;
      dlBtn.textContent = `⬇ ดาวน์โหลดวีดีโอ (${job.size_mb} MB)`;
      dlBtn.style.display = 'block';
      document.getElementById('btn-start').disabled = false;
      addHistory(jobId, job);
    } else if (job.status === 'error') {
      clearInterval(pollTimer);
      setBadge('error', '❌ ผิดพลาด');
      showError(job.error || job.message);
      document.getElementById('btn-start').disabled = false;
    }
  });
}

function updateSteps(pct) {
  const map = [{id:'s1',min:5},{id:'s2',min:30},{id:'s3',min:35},{id:'s4',min:88}];
  map.forEach((s, i) => {
    const el = document.getElementById(s.id);
    if (pct >= 100) el.className = 'step done';
    else if (pct >= s.min && (i === map.length-1 || pct < map[i+1].min)) el.className = 'step active';
    else if (pct >= s.min) el.className = 'step done';
    else el.className = 'step';
  });
}

function setBadge(cls, text) {
  const b = document.getElementById('badge');
  b.className = 'badge ' + cls;
  b.innerHTML = (cls === 'running' ? '<span class="spinner"></span>&nbsp;' : '') + text;
}

function showError(msg) {
  const box = document.getElementById('error-box');
  box.textContent = msg;
  box.style.display = 'block';
}

function addHistory(jobId, job) {
  const hist = document.getElementById('history');
  if (hist.children.length === 0) {
    const h = document.createElement('h2');
    h.textContent = 'วีดีโอที่ประมวลผลแล้ว';
    hist.appendChild(h);
  }
  const item = document.createElement('div');
  item.className = 'hist-item';
  item.innerHTML = `
    <img class="hist-thumb" src="${job.thumbnail||''}" alt=""/>
    <div class="hist-info">
      <div class="title">${job.title||'วีดีโอ'}</div>
      <div class="size">${job.size_mb} MB • MP4</div>
    </div>
    <a class="hist-dl" href="/api/download/${jobId}">⬇ ดาวน์โหลด</a>
  `;
  hist.appendChild(item);
}

// Allow Enter key
document.getElementById('url-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') startJob();
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
