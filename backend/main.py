"""
Brain Morph – FastAPI backend
Run:  uvicorn backend.main:app --reload
      (from the project root with DL_CV_312 activated)
"""
from __future__ import annotations

import asyncio
import base64
import fnmatch
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── src path setup ────────────────────────────────────────────────────────────
_root = Path(__file__).parent.parent
for _p in (
    str(_root / "src" / "registration"),
    str(_root / "src" / "utils"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from volume import Volume                              # noqa: E402
from pipeline import RegistrationPipeline, Stage, _make_regular_grid  # noqa: E402
from optimizers import SAOptimizer, GradientOptimizer, HybridOptimizer  # noqa: E402
from mesh_transformer_3d import MeshTransformer3D    # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Job state
# ─────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    STOPPED  = "stopped"
    ERROR    = "error"


_RESULT_DIR = Path(tempfile.gettempdir()) / "brain_morph_results"
_RESULT_DIR.mkdir(exist_ok=True)

_MAX_JOBS   = 20          # максимум джобов в словаре
_JOB_TTL    = 5 * 60      # секунд до автоудаления завершённого джоба


class Job:
    def __init__(self, job_id: str, request: Any):
        self.job_id       = job_id
        self.request      = request
        self.status       = JobStatus.PENDING
        self.events: queue.Queue = queue.Queue()
        self.stop_event   = threading.Event()
        self.result_path: Path | None = None  # путь к .npy на диске, не в RAM
        self.error:       str | None = None
        self.thread:      threading.Thread | None = None
        self.finished_at: float | None = None  # time.monotonic() момента завершения

    def delete_result(self) -> None:
        if self.result_path and self.result_path.exists():
            self.result_path.unlink(missing_ok=True)
            self.result_path = None


class _JobStopped(Exception):
    pass


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _evict_old_jobs() -> None:
    """Удалить завершённые джобы старше TTL или превышающие лимит."""
    terminal = {JobStatus.DONE, JobStatus.STOPPED, JobStatus.ERROR}
    now = time.monotonic()
    with _jobs_lock:
        # Удалить по TTL
        expired = [
            jid for jid, j in _jobs.items()
            if j.status in terminal and j.finished_at is not None
               and (now - j.finished_at) > _JOB_TTL
        ]
        for jid in expired:
            _jobs[jid].delete_result()
            del _jobs[jid]

        # Удалить самые старые если превышен лимит
        if len(_jobs) >= _MAX_JOBS:
            finished = sorted(
                [(jid, j) for jid, j in _jobs.items() if j.status in terminal],
                key=lambda x: x[1].finished_at or 0,
            )
            for jid, j in finished[:len(_jobs) - _MAX_JOBS + 1]:
                j.delete_result()
                del _jobs[jid]

# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class StageConfig(BaseModel):
    grid:  str   = "5,5,5"   # "ny,nx,nz"
    steps: int   = 2000
    lam:   float = 1e-3
    scale: int   = 1


class RegisterRequest(BaseModel):
    moving_path:      str
    fixed_path:       str
    stages:           list[StageConfig]
    optimizer:        str   = "SA"        # SA | Gradient | Hybrid
    similarity:       str   = "corr"      # corr | ncc | mse
    lam:              float = 1e-3
    mask_path:        str | None = None
    channel_weights:  list[float] | None = None
    callback_freq:    int   = 50
    temp_start:       float = 1e-3
    temp_end:         float = 3.3e-5
    coeff_start:      float = 0.2
    output_path:      str | None = None   # where to save warped .nii.gz


class PairConfig(BaseModel):
    moving_path: str
    fixed_path:  str
    output_path: str | None = None


class BatchRegisterRequest(BaseModel):
    pairs:           list[PairConfig]
    stages:          list[StageConfig]
    optimizer:       str   = "SA"
    similarity:      str   = "corr"
    lam:             float = 1e-3
    mask_path:       str | None = None
    channel_weights: list[float] | None = None
    callback_freq:   int   = 50
    temp_start:      float = 1e-3
    temp_end:        float = 3.3e-5
    coeff_start:     float = 0.2
    output_dir:      str   = "output"


# ─────────────────────────────────────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────────────────────────────────────

def _warped_slice_b64(warped: torch.Tensor) -> str:
    """Axial mid-slice of warped volume → base-64 PNG string."""
    vol = warped if warped.dim() == 3 else warped[0]   # (D, H, W)
    mid = vol.shape[0] // 2
    sl = vol[mid].float().cpu().numpy()
    mn, mx = float(sl.min()), float(sl.max())
    if mx > mn:
        sl = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        sl = np.zeros_like(sl, dtype=np.uint8)
    _, buf = cv2.imencode(".png", sl)
    return base64.b64encode(buf.tobytes()).decode()


def _load_volume(path: str, scale: int = 1) -> torch.Tensor:
    """Load NIfTI or TIFF series; return (D, H, W) float32 tensor."""
    p = Path(path)
    if p.suffix in (".nii", ".gz") or ".nii" in p.name:
        vol = Volume.load_nii(str(p), scale=scale)
    elif "*" in path or p.suffix in (".tif", ".tiff"):
        vol = Volume.load_tiff_series(str(p), scale=scale)
    else:
        raise ValueError(f"Unsupported file: {path}")
    t = torch.as_tensor(np.array(vol), dtype=torch.float32)
    if t.dim() == 4:  # (C, D, H, W) → (D, H, W)
        t = t.mean(0)
    return t


def _make_callback(job: Job, stage_idx: int, n_stages: int, n_steps: int):
    def cb(step: int, cost: float, warped: torch.Tensor) -> None:
        if job.stop_event.is_set():
            raise _JobStopped()
        event = {
            "stage":        stage_idx + 1,
            "n_stages":     n_stages,
            "step":         step,
            "n_steps":      n_steps,
            "cost":         round(float(cost), 6),
            "warped_slice": _warped_slice_b64(warped),
        }
        job.events.put(event)
    return cb


def _build_optimizer(req: RegisterRequest | BatchRegisterRequest,
                     stage_idx: int, n_stages: int, n_steps: int,
                     job: Job):
    cb = _make_callback(job, stage_idx, n_stages, n_steps)
    name = req.optimizer

    if name == "SA":
        return SAOptimizer(
            temp_start    = req.temp_start,
            temp_end      = req.temp_end,
            coeff_start   = req.coeff_start,
            similarity    = req.similarity,
            callback      = cb,
            callback_freq = req.callback_freq,
        )
    elif name == "Gradient":
        return GradientOptimizer(similarity=req.similarity)
    elif name == "Hybrid":
        return HybridOptimizer(
            sa_optimizer=SAOptimizer(
                temp_start    = req.temp_start,
                temp_end      = req.temp_end,
                coeff_start   = req.coeff_start,
                similarity    = req.similarity,
                callback      = cb,
                callback_freq = req.callback_freq,
            ),
            gd_optimizer=GradientOptimizer(similarity=req.similarity),
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _run_job(job: Job, moving_path: str, fixed_path: str,
             req: RegisterRequest | BatchRegisterRequest,
             output_path: str | None = None):
    try:
        job.status = JobStatus.RUNNING

        # Load images
        im_mov = _load_volume(moving_path)
        im_fix = _load_volume(fixed_path)

        mask = None
        if getattr(req, "mask_path", None):
            mask = _load_volume(req.mask_path) > 0.5

        cw = None
        if req.channel_weights:
            cw = torch.tensor(req.channel_weights, dtype=torch.float32)

        n_stages = len(req.stages)
        stages = []
        for i, sc in enumerate(req.stages):
            ny, nx, nz = [int(x.strip()) for x in sc.grid.split(",")]
            opt = _build_optimizer(req, i, n_stages, sc.steps, job)
            stages.append(Stage(
                grid_shape  = (ny, nx, nz),
                optimizer   = opt,
                n_steps     = sc.steps,
                lam         = sc.lam,
                image_scale = sc.scale,
            ))

        pipeline = RegistrationPipeline(stages)

        try:
            grid = pipeline.run(im_mov, im_fix, mask=mask, channel_weights=cw)
        except _JobStopped:
            job.status      = JobStatus.STOPPED
            job.finished_at = time.monotonic()
            job.events.put({"status": "stopped", "final": True})
            _evict_old_jobs()
            return

        # Apply final transform to get warped volume
        last_shape = stages[-1].grid_shape
        transformer = MeshTransformer3D(
            _make_regular_grid(*last_shape),
            tuple(im_mov.shape[-3:]),
        )
        warped = transformer.transform(im_mov, grid)
        if warped.dim() == 4:
            warped = warped.squeeze(0)

        warped_np = warped.cpu().numpy()

        # Сохранить результат во временный файл (не держать в RAM)
        result_file = _RESULT_DIR / f"{job.job_id}.npy"
        np.save(str(result_file), warped_np)
        job.result_path = result_file

        # Опционально — сохранить в указанный путь
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                import nibabel as nib
                img = nib.Nifti1Image(warped_np, affine=np.eye(4))
                nib.save(img, str(out))
            except Exception:
                np.save(str(out.with_suffix(".npy")), warped_np)

        job.status      = JobStatus.DONE
        job.finished_at = time.monotonic()
        job.events.put({
            "status":        "done",
            "final":         True,
            "warped_slice":  _warped_slice_b64(warped),
        })
        _evict_old_jobs()

    except Exception as exc:
        job.status      = JobStatus.ERROR
        job.finished_at = time.monotonic()
        job.error       = traceback.format_exc()
        job.events.put({"status": "error", "error": str(exc), "final": True})
        _evict_old_jobs()


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Brain Morph API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── /scan ─────────────────────────────────────────────────────────────────────

@app.get("/browse")
def browse(path: str = "/", filter: str = "*"):
    """Return directory listing for a given path, files filtered by glob mask."""
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(404, f"Path not found: {path}")
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {path}")

    entries = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                entries.append({"name": entry.name, "path": str(entry), "type": "dir"})
            elif entry.is_file() and (filter == "*" or fnmatch.fnmatch(entry.name, filter)):
                entries.append({
                    "name": entry.name, "path": str(entry),
                    "type": "file", "size": entry.stat().st_size,
                })
    except PermissionError:
        pass

    return {
        "path":    str(p),
        "parent":  str(p.parent),
        "entries": entries,
    }


@app.get("/scan")
def scan(folder: str, mask: str = "*"):
    """Return files in *folder* matching *mask* (shell glob)."""
    p = Path(folder)
    if not p.exists():
        raise HTTPException(404, f"Folder not found: {folder}")
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {folder}")

    files = []
    for entry in sorted(p.iterdir()):
        if entry.is_file() and fnmatch.fnmatch(entry.name, mask):
            files.append({
                "name": entry.name,
                "path": str(entry),
                "size": entry.stat().st_size,
            })
    return {"folder": str(p), "mask": mask, "files": files}


# ── /register ─────────────────────────────────────────────────────────────────

@app.post("/register")
def register(req: RegisterRequest):
    job_id = str(uuid.uuid4())
    job = Job(job_id, req)
    _jobs[job_id] = job

    t = threading.Thread(
        target=_run_job,
        args=(job, req.moving_path, req.fixed_path, req, req.output_path),
        daemon=True,
    )
    job.thread = t
    t.start()

    return {"job_id": job_id}


# ── /register/batch ───────────────────────────────────────────────────────────

@app.post("/register/batch")
def register_batch(req: BatchRegisterRequest):
    job_ids = []
    for pair in req.pairs:
        job_id = str(uuid.uuid4())
        job = Job(job_id, req)
        _jobs[job_id] = job

        out = pair.output_path
        if not out and req.output_dir:
            # Auto output path: output_dir/moving_name_to_fixed_name/warped.nii.gz
            mov_stem = Path(pair.moving_path).name.split(".")[0]
            fix_stem = Path(pair.fixed_path).name.split(".")[0]
            out = str(Path(req.output_dir) / f"{mov_stem}_to_{fix_stem}" / "warped.nii.gz")

        t = threading.Thread(
            target=_run_job,
            args=(job, pair.moving_path, pair.fixed_path, req, out),
            daemon=True,
        )
        job.thread = t
        t.start()
        job_ids.append(job_id)

    return {"batch_size": len(job_ids), "job_ids": job_ids}


# ── /progress/{job_id} (SSE) ──────────────────────────────────────────────────

@app.get("/progress/{job_id}")
async def progress(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    loop = asyncio.get_event_loop()

    async def _generator():
        while True:
            # Try to get an event (non-blocking)
            try:
                event = await loop.run_in_executor(
                    None, lambda: job.events.get(timeout=0.2)
                )
                yield {"data": json.dumps(event)}
                if event.get("final"):
                    break
            except queue.Empty:
                # Heartbeat to keep connection alive
                if job.status in (JobStatus.DONE, JobStatus.STOPPED, JobStatus.ERROR):
                    yield {"data": json.dumps({"status": job.status, "final": True})}
                    break
                yield {"data": json.dumps({"heartbeat": True})}

    return EventSourceResponse(_generator())


# ── /stop/{job_id} ────────────────────────────────────────────────────────────

@app.post("/stop/{job_id}")
def stop_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    job.stop_event.set()
    return {"status": "stopping", "job_id": job_id}


# ── /status/{job_id} ──────────────────────────────────────────────────────────

@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return {
        "job_id": job_id,
        "status": job.status,
        "error":  job.error,
    }


# ── /result/{job_id} ──────────────────────────────────────────────────────────

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(400, f"Job not done yet (status={job.status})")
    if not job.result_path or not job.result_path.exists():
        raise HTTPException(500, "Result file not available (may have been cleaned up)")
    content = job.result_path.read_bytes()
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="warped_{job_id}.npy"'},
    )


# ── /jobs ─────────────────────────────────────────────────────────────────────

@app.get("/jobs")
def list_jobs():
    return [
        {
            "job_id":      jid,
            "status":      j.status,
            "error":       j.error,
            "result_mb":   round(j.result_path.stat().st_size / 1024 / 1024, 1)
                           if j.result_path and j.result_path.exists() else None,
            "age_sec":     round(time.monotonic() - j.finished_at)
                           if j.finished_at else None,
        }
        for jid, j in _jobs.items()
    ]


# ── /memory ───────────────────────────────────────────────────────────────────

@app.get("/memory")
def memory_info():
    """Disk usage of temp result files + job counts by status."""
    total_bytes = sum(
        j.result_path.stat().st_size
        for j in _jobs.values()
        if j.result_path and j.result_path.exists()
    )
    by_status: dict[str, int] = {}
    for j in _jobs.values():
        by_status[j.status] = by_status.get(j.status, 0) + 1
    return {
        "jobs_total":     len(_jobs),
        "jobs_by_status": by_status,
        "result_disk_mb": round(total_bytes / 1024 / 1024, 1),
        "ttl_sec":        _JOB_TTL,
        "max_jobs":       _MAX_JOBS,
        "result_dir":     str(_RESULT_DIR),
    }


# ── /jobs/clear ───────────────────────────────────────────────────────────────

@app.delete("/jobs/clear")
def clear_jobs():
    """Remove all finished jobs from memory and delete their temp result files."""
    terminal = {JobStatus.DONE, JobStatus.STOPPED, JobStatus.ERROR}
    with _jobs_lock:
        to_remove = [jid for jid, j in _jobs.items() if j.status in terminal]
        for jid in to_remove:
            _jobs[jid].delete_result()
            del _jobs[jid]
    return {"cleared": len(to_remove), "remaining": len(_jobs)}
