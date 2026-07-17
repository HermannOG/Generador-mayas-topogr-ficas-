"""
Topo Trail Generator – FastAPI backend (routes + streaming plumbing only;
the generation pipeline lives in src/pipeline.py).

API contract aligned with topotrail.com:

  POST /api/upload            multipart: file* (GPX, repeatable) OR fileHash
                              (JSON list of hashes of already-uploaded files),
                              plus settings (JSON, camelCase schema).
                              Streams NDJSON progress lines:
                                {"type":"info","message": "..."}
                                {"type":"path","path": <jobId>, "fileHash":[...]}
                                {"type":"error","message": "..."}
                              The server generates ALL assets up front.

  POST /api/preview           fast terrain-only generation (image.png only).

  GET  /api/settings-schema   the default settings object (JSON).

  GET  /api/public/{job}/…    generated assets for the browser viewer:
                              zone_*.stl, trail{i}.stl, image.png, meta.json
                              (plus the printable STLs).

  POST /api/download/{job}    body {"name": str} → ZIP with printable STLs.
"""

import asyncio
import hashlib
import io
import json
import os
import uuid as uuid_module
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.gpx_handler import is_swim, parse_gpx_with_type
from src.pipeline import GENERATED_DIR, GPX_STORE, cleanup_old_jobs, generate_job
from src.settings import (  # noqa: F401  (re-exported: public schema surface)
    DEFAULT_SETTINGS,
    LEGACY_KEYS,
    SettingsError,
    normalize_settings,
)

# Concurrency guard: at most N generation jobs run at once (env-overridable)
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("TOPOTRAIL_MAX_JOBS", "2")))
_job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

app = FastAPI(title="Topo Trail Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/settings-schema")
async def settings_schema():
    """The default settings object — the front-end builds its form from it."""
    return DEFAULT_SETTINGS


# ── Upload / generate endpoints ───────────────────────────────────────────────
@app.post("/api/upload")
async def upload(
    file: Optional[List[UploadFile]] = File(None),
    fileHash: Optional[str] = Form(None),
    settings: str = Form("{}"),
):
    """Full generation: 2D preview + all 3D meshes and printable STLs."""
    return await _generate_endpoint(file, fileHash, settings, terrain_only=False)


@app.post("/api/preview")
async def preview(
    file: Optional[List[UploadFile]] = File(None),
    fileHash: Optional[str] = Form(None),
    settings: str = Form("{}"),
):
    """Fast terrain-only generation: just the 2D map preview (image.png)."""
    return await _generate_endpoint(file, fileHash, settings, terrain_only=True)


async def _generate_endpoint(file, fileHash, settings, terrain_only):
    try:
        cfg = normalize_settings(json.loads(settings))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid settings JSON") from exc
    except SettingsError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Resolve GPX contents: new uploads and/or previously stored hashes
    GPX_STORE.mkdir(parents=True, exist_ok=True)
    contents: list[bytes] = []
    hashes:   list[str]   = []

    if file:
        for up in file:
            data = await up.read()
            h = hashlib.sha256(data).hexdigest()
            (GPX_STORE / f"{h}.gpx").write_bytes(data)
            contents.append(data)
            hashes.append(h)

    if not contents and fileHash:
        try:
            requested = json.loads(fileHash)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "Invalid fileHash JSON") from exc
        for h in requested:
            p = GPX_STORE / f"{Path(str(h)).name}.gpx"
            if not p.exists():
                raise HTTPException(410, f"Cached GPX no longer available: {h}")
            contents.append(p.read_bytes())
            hashes.append(str(h))

    if not contents:
        raise HTTPException(400, "No GPX file or fileHash provided")

    async def stream():
        def line(obj):
            return json.dumps(obj) + "\n"

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(msg):
            # Strings become info lines; dicts pass through verbatim
            loop.call_soon_threadsafe(queue.put_nowait, msg)

        def as_line(msg):
            if isinstance(msg, dict):
                return line(msg)
            return line({"type": "info", "message": msg})

        yield line({"type": "info", "message": "Parsing GPX files"})
        try:
            trails = []
            for data in contents:
                pts, activity = parse_gpx_with_type(data)
                if not pts:
                    continue
                flat = is_swim(activity)
                if flat:
                    # Swims are flat water: GPS elevation is noise, use median
                    med = float(np.median([p[2] for p in pts]))
                    pts = [(p[0], p[1], med) for p in pts]
                trails.append({"points": pts, "flat": flat})
            if not trails:
                yield line({"type": "error", "message": "No points found in the GPX files"})
                return

            cleanup_old_jobs()
            job_id = uuid_module.uuid4().hex[:12]
            job_dir = GENERATED_DIR / job_id

            async def run_guarded():
                async with _job_semaphore:
                    await asyncio.to_thread(generate_job, job_dir, trails, cfg,
                                            progress, terrain_only)

            task = asyncio.create_task(run_guarded())
            while not task.done():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield as_line(msg)
                except asyncio.TimeoutError:
                    pass
            task.result()  # re-raise generation errors

            while not queue.empty():
                yield as_line(queue.get_nowait())
            yield line({"type": "path", "path": job_id, "fileHash": hashes})

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield line({"type": "error", "message": f"Error generating map: {exc}"})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── Download endpoint ─────────────────────────────────────────────────────────
@app.post("/api/download/{job_id}")
async def download(job_id: str, body: dict):
    job_dir = GENERATED_DIR / Path(job_id).name
    meta_path = job_dir / "meta.json"
    if not job_dir.is_dir() or not meta_path.exists():
        raise HTTPException(404, "Unknown or expired job")

    meta = json.loads(meta_path.read_text())
    name = str(body.get("name") or "map3d").strip() or "map3d"

    # singleColor → separate terrain + trail STLs for filament swaps;
    # otherwise a single combined STL.
    if meta["settings"].get("singleColor"):
        files = [("terrain.stl", f"{name}_terrain.stl")] + [
            (f"trail{i}.stl", f"{name}_trail{i}.stl")
            for i in range(meta.get("trailAmount", 1))
        ]
    else:
        files = [("map.stl", f"{name}.stl")]

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files:
            p = job_dir / src
            if p.exists():
                zf.write(p, arcname)
    zip_buf.seek(0)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )


# Generated assets (zone_*.stl, trail{i}.stl, image.png, meta.json, STLs)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/public", StaticFiles(directory=GENERATED_DIR), name="public")

# Static files – mounted last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")
