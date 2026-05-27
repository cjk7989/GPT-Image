import os
import json
import base64
import traceback
import gc
import zipfile
import io
import uuid
import threading
from datetime import datetime
from pathlib import Path
import hashlib
import time
import shutil
import concurrent.futures
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import AzureOpenAI, APIStatusError, APITimeoutError
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

app = FastAPI()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent
IS_AZURE = bool(os.getenv("WEBSITE_SITE_NAME"))

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-image-2")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
    timeout=300,
)


def _get_user(request: Request) -> Optional[str]:
    """Return user id or None (anonymous on Azure)."""
    if IS_AZURE:
        name = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME")
        if not name:
            return None
        return hashlib.sha256(name.encode()).hexdigest()[:12]
    return "local"


def _user_dir(user: str) -> Path:
    d = OUTPUT_DIR / user
    d.mkdir(exist_ok=True)
    return d


def _history_file(user: str) -> Path:
    return _user_dir(user) / "history.json"


_history_locks: dict[str, threading.Lock] = {}
_history_locks_lock = threading.Lock()


def _get_history_lock(user: str) -> threading.Lock:
    with _history_locks_lock:
        if user not in _history_locks:
            _history_locks[user] = threading.Lock()
        return _history_locks[user]


def _load_history(user: str) -> list:
    hf = _history_file(user)
    if hf.exists():
        return json.loads(hf.read_text(encoding="utf-8"))
    return []


def _save_history(user: str, history: list):
    _history_file(user).write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Favorites helpers ──

def _fav_file(user: str, kind: str) -> Path:
    """kind = 'prompt' or 'image'"""
    return _user_dir(user) / f"{kind}_favorites.json"


def _default_collection():
    return {"id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "name": "默认", "items": []}


def _load_fav(user: str, kind: str) -> list:
    f = _fav_file(user, kind)
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        if data:
            return data
    return [_default_collection()]


def _save_fav(user: str, kind: str, data: list):
    _fav_file(user, kind).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: str = "auto"


MAX_RETRIES = 3


def _is_retryable(e: APIStatusError) -> bool:
    """Return True for transient errors worth retrying."""
    return e.status_code in (400, 404, 429, 500, 502, 503, 504)


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


SSE_KEEPALIVE = ": keepalive\n\n"
KEEPALIVE_INTERVAL = 10  # seconds
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ── Task management ──
# Tasks run independently of SSE connections. Clients can reconnect at any time.

_tasks: dict[str, dict] = {}  # task_id -> {status, events[], result, error, done}
_tasks_lock = threading.Lock()
_TASK_TTL = 600  # seconds to keep completed tasks


def _task_emit(task_id: str, event: str, data: dict):
    """Append an SSE event to a task's event log."""
    with _tasks_lock:
        t = _tasks.get(task_id)
        if t:
            t["events"].append((event, data))


def _task_finish(task_id: str):
    """Mark task as done so watchers know to stop."""
    with _tasks_lock:
        t = _tasks.get(task_id)
        if t:
            t["done"] = True
            t["finished_at"] = time.time()


def _cleanup_old_tasks():
    """Remove tasks that finished more than _TASK_TTL seconds ago."""
    now = time.time()
    with _tasks_lock:
        expired = [tid for tid, t in _tasks.items()
                    if t.get("done") and now - t.get("finished_at", now) > _TASK_TTL]
        for tid in expired:
            del _tasks[tid]


def _create_task() -> str:
    _cleanup_old_tasks()
    task_id = uuid.uuid4().hex[:12]
    with _tasks_lock:
        _tasks[task_id] = {"events": [], "done": False, "finished_at": None}
    return task_id


def _run_generate_task(task_id: str, user: str, req_prompt: str, req_n: int, req_size: str, req_quality: str):
    """Run image generation in background thread, emitting events to task log."""
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            _task_emit(task_id, "status", {"message": f"正在生成图像（第 {attempt} 次尝试）..."})
        else:
            _task_emit(task_id, "status", {"message": "正在生成图像..."})
        try:
            result = client.images.generate(
                model=deployment,
                prompt=req_prompt,
                n=req_n,
                size=req_size,
                quality=req_quality,
                output_format="jpeg",
            )
            break
        except APIStatusError as e:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            print(f"[API Error] generate attempt {attempt}/{MAX_RETRIES} status={e.status_code} body={json.dumps(body, ensure_ascii=False, indent=2)}")
            if _is_retryable(e) and attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"[Retry] retrying in {wait}s...")
                _task_emit(task_id, "error", {"attempt": attempt, "status": e.status_code, "message": str(e.message), "wait": wait})
                time.sleep(wait)
                continue
            _task_emit(task_id, "fail", {"message": str(e.message), "raw": body})
            _task_finish(task_id)
            return
        except APITimeoutError as e:
            print(f"[Timeout] generate attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"[Retry] retrying in {wait}s...")
                _task_emit(task_id, "error", {"attempt": attempt, "status": 0, "message": "请求超时", "wait": wait})
                time.sleep(wait)
                continue
            _task_emit(task_id, "fail", {"message": "请求超时，请稍后重试"})
            _task_finish(task_id)
            return
        except Exception as e:
            traceback.print_exc()
            _task_emit(task_id, "fail", {"message": str(e)})
            _task_finish(task_id)
            return

    # Process result
    try:
        images_out = []
        saved_files = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        adir = _user_dir(user) if user else _user_dir("_anon")
        prefix = f"{user}/" if user else "_anon/"

        for idx, item in enumerate(result.data):
            b64 = item.b64_json
            if b64:
                filename = f"{ts}_{idx+1}.jpg"
                (adir / filename).write_bytes(base64.b64decode(b64))
                saved_files.append(f"{prefix}{filename}")
                images_out.append(f"/output/{prefix}{filename}")
            elif item.url:
                images_out.append(item.url)

        del result
        gc.collect()

        if user and saved_files:
            record = {
                "id": ts,
                "time": datetime.now().isoformat(),
                "prompt": req_prompt,
                "size": req_size,
                "quality": req_quality,
                "n": req_n,
                "files": saved_files,
            }
            with _get_history_lock(user):
                history = _load_history(user)
                history.insert(0, record)
                _save_history(user, history)

        _task_emit(task_id, "done", {"images": images_out})
    except Exception as e:
        traceback.print_exc()
        _task_emit(task_id, "fail", {"message": str(e)})
    _task_finish(task_id)


def _run_edit_task(task_id: str, user: str, req_prompt: str, req_n: int, req_size: str, req_quality: str,
                   image_files: list, input_filenames: list):
    """Run image editing in background thread, emitting events to task log."""
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            _task_emit(task_id, "status", {"message": f"正在编辑图像（第 {attempt} 次尝试）..."})
        else:
            _task_emit(task_id, "status", {"message": "正在编辑图像..."})
        try:
            result = client.images.edit(
                model=deployment,
                prompt=req_prompt,
                image=image_files,
                n=req_n,
                size=req_size,
                quality=req_quality,
            )
            break
        except APIStatusError as e:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            print(f"[API Error] edit attempt {attempt}/{MAX_RETRIES} status={e.status_code} body={json.dumps(body, ensure_ascii=False, indent=2)}")
            if _is_retryable(e) and attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"[Retry] retrying in {wait}s...")
                _task_emit(task_id, "error", {"attempt": attempt, "status": e.status_code, "message": str(e.message), "wait": wait})
                time.sleep(wait)
                continue
            _task_emit(task_id, "fail", {"message": str(e.message), "raw": body})
            _task_finish(task_id)
            return
        except APITimeoutError as e:
            print(f"[Timeout] edit attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"[Retry] retrying in {wait}s...")
                _task_emit(task_id, "error", {"attempt": attempt, "status": 0, "message": "请求超时", "wait": wait})
                time.sleep(wait)
                continue
            _task_emit(task_id, "fail", {"message": "请求超时，请稍后重试"})
            _task_finish(task_id)
            return
        except Exception as e:
            traceback.print_exc()
            _task_emit(task_id, "fail", {"message": str(e)})
            _task_finish(task_id)
            return

    try:
        del image_files
        result_images = []
        saved_files = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        adir = _user_dir(user) if user else _user_dir("_anon")
        prefix = f"{user}/" if user else "_anon/"

        for idx, item in enumerate(result.data):
            b64 = item.b64_json
            if b64:
                filename = f"{ts}_edit_{idx+1}.jpg"
                (adir / filename).write_bytes(base64.b64decode(b64))
                saved_files.append(f"{prefix}{filename}")
                result_images.append(f"/output/{prefix}{filename}")
            elif item.url:
                result_images.append(item.url)

        del result
        gc.collect()

        if user and saved_files:
            record = {
                "id": ts,
                "time": datetime.now().isoformat(),
                "prompt": req_prompt,
                "size": req_size,
                "quality": req_quality,
                "n": req_n,
                "type": "edit",
                "input_images": input_filenames,
                "files": saved_files,
            }
            with _get_history_lock(user):
                history = _load_history(user)
                history.insert(0, record)
                _save_history(user, history)

        _task_emit(task_id, "done", {"images": result_images})
    except Exception as e:
        traceback.print_exc()
        _task_emit(task_id, "fail", {"message": str(e)})
    _task_finish(task_id)


@app.post("/api/generate")
async def generate_image(req: ImageRequest, request: Request):
    user = _get_user(request)
    task_id = _create_task()
    _executor.submit(_run_generate_task, task_id, user, req.prompt, req.n, req.size, req.quality)
    return {"task_id": task_id}


@app.post("/api/edit")
async def edit_image(
    request: Request,
    prompt: str = Form(...),
    size: str = Form("1024x1024"),
    quality: str = Form("auto"),
    n: int = Form(1),
    images: list[UploadFile] = File(...),
):
    user = _get_user(request)
    # Read uploaded image files eagerly
    image_files = []
    for img in images:
        content = await img.read()
        content_type = img.content_type or "application/octet-stream"
        image_files.append((img.filename, content, content_type))
    input_filenames = [img.filename for img in images]

    task_id = _create_task()
    _executor.submit(_run_edit_task, task_id, user, prompt, n, size, quality, image_files, input_filenames)
    return {"task_id": task_id}


@app.get("/api/task/{task_id}")
async def task_stream(task_id: str, request: Request, last_event_id: int = 0):
    """SSE stream for task progress. Supports reconnection via last_event_id query param."""
    with _tasks_lock:
        if task_id not in _tasks:
            raise HTTPException(404, "Task not found")

    def _stream():
        cursor = last_event_id
        while True:
            with _tasks_lock:
                t = _tasks.get(task_id)
                if not t:
                    return
                events = t["events"]
                done = t["done"]

            # Replay any events after cursor
            while cursor < len(events):
                event_name, data = events[cursor]
                cursor += 1
                yield f"id: {cursor}\n{_sse_event(event_name, data)}"

            if done:
                return

            # No new events yet — send keepalive and wait
            yield SSE_KEEPALIVE
            time.sleep(KEEPALIVE_INTERVAL)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/user")
async def get_user_info(request: Request):
    if IS_AZURE:
        name = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME")
        return {"logged_in": bool(name), "name": name}
    return {"logged_in": True, "name": "local"}


@app.get("/api/history")
async def get_history(request: Request):
    user = _get_user(request)
    if not user:
        return []
    return _load_history(user)


def _fav_image_paths(user: str) -> set:
    """Return set of file paths referenced by image favorites."""
    paths = set()
    for col in _load_fav(user, "image"):
        for it in col.get("items", []):
            if it.get("path"):
                paths.add(it["path"])
    return paths


def _history_image_paths(user: str) -> set:
    """Return set of file paths referenced by history."""
    paths = set()
    for r in _load_history(user):
        paths.update(r.get("files", []))
    return paths


def _cleanup_orphan_files(user: str, file_paths: list[str]):
    """Delete files that are no longer referenced by history or favorites."""
    kept_by_fav = _fav_image_paths(user)
    kept_by_hist = _history_image_paths(user)
    protected = kept_by_fav | kept_by_hist
    for f in file_paths:
        if f not in protected:
            fp = OUTPUT_DIR / f
            if fp.exists():
                fp.unlink()


@app.post("/api/history/batch-remove-files")
async def batch_remove_files_from_history(request: Request):
    """Remove specific files from history records, delete orphan files from disk."""
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=403, detail="Login required")
    body = await request.json()
    files_to_remove = set(body.get("files", []))
    if not files_to_remove:
        raise HTTPException(status_code=400, detail="No files specified")
    removed = 0
    with _get_history_lock(user):
        history = _load_history(user)
        new_history = []
        for record in history:
            orig_len = len(record.get("files", []))
            record["files"] = [f for f in record.get("files", []) if f not in files_to_remove]
            removed += orig_len - len(record["files"])
            if record["files"]:
                new_history.append(record)
        _save_history(user, new_history)
    _cleanup_orphan_files(user, list(files_to_remove))
    return {"ok": True, "removed": removed}


@app.delete("/api/history/{record_id}")
async def delete_history(record_id: str, request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=403, detail="Login required")
    with _get_history_lock(user):
        history = _load_history(user)
        record = next((r for r in history if r["id"] == record_id), None)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        files_to_check = record.get("files", [])
        history = [r for r in history if r["id"] != record_id]
        _save_history(user, history)
    _cleanup_orphan_files(user, files_to_check)
    return {"ok": True}


@app.delete("/api/history")
async def clear_history(request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=403, detail="Login required")
    with _get_history_lock(user):
        history = _load_history(user)
        all_files = []
        for record in history:
            all_files.extend(record.get("files", []))
        _save_history(user, [])
    _cleanup_orphan_files(user, all_files)
    return {"ok": True, "deleted": len(history)}


# ── Favorites API (prompt & image) ──

def _require_user(request: Request) -> str:
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=403, detail="Login required")
    return user


MAX_EXPORT_BYTES = 200 * 1024 * 1024  # 200 MB


@app.get("/api/favorites/image/{col_id}/download")
async def download_image_collection(col_id: str, request: Request):
    """Download all images in a collection as a zip (folder inside zip)."""
    user = _require_user(request)
    image_data = _load_fav(user, "image")
    col = next((c for c in image_data if c["id"] == col_id), None)
    if not col:
        raise HTTPException(404, "收藏夹不存在")
    items = col.get("items", [])
    if not items:
        raise HTTPException(400, "收藏夹为空")
    folder_name = col["name"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen = set()
        for it in items:
            p = OUTPUT_DIR / it["path"]
            if p.exists():
                fname = Path(it["path"]).name
                if fname in seen:
                    base, ext = Path(fname).stem, Path(fname).suffix
                    fname = f"{base}_{it['id'][:6]}{ext}"
                seen.add(fname)
                zf.write(p, f"{folder_name}/{fname}")
    buf.seek(0)
    from urllib.parse import quote
    encoded_name = quote(folder_name + '.zip')
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
    )


@app.get("/api/favorites/export")
async def export_favorites(request: Request):
    user = _require_user(request)
    prompt_data = _load_fav(user, "prompt")
    image_data = _load_fav(user, "image")

    # Calculate total image size
    total_size = 0
    image_paths = []
    for col in image_data:
        for it in col.get("items", []):
            p = OUTPUT_DIR / it["path"]
            if p.exists():
                total_size += p.stat().st_size
                image_paths.append((it["path"], p))
    if total_size > MAX_EXPORT_BYTES:
        raise HTTPException(400, f"收藏图片总大小超过 {MAX_EXPORT_BYTES // 1024 // 1024}MB，请减少收藏后重试")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prompt_favorites.json", json.dumps(prompt_data, ensure_ascii=False, indent=2))
        zf.writestr("image_favorites.json", json.dumps(image_data, ensure_ascii=False, indent=2))
        seen = set()
        for rel, full in image_paths:
            fname = Path(rel).name
            if fname not in seen:
                zf.write(full, f"images/{fname}")
                seen.add(fname)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="favorites_{ts}.zip"'},
    )


@app.post("/api/favorites/import")
async def import_favorites(request: Request, file: UploadFile = File(...)):
    user = _require_user(request)
    content = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(400, "无效的 ZIP 文件")

    stats = {"prompt_added": 0, "prompt_skipped": 0, "image_added": 0, "image_skipped": 0, "collections_created": 0}

    # Import prompts
    if "prompt_favorites.json" in zf.namelist():
        imported = json.loads(zf.read("prompt_favorites.json"))
        existing = _load_fav(user, "prompt")
        existing_map = {c["name"]: c for c in existing}
        for col in imported:
            if col["name"] in existing_map:
                target = existing_map[col["name"]]
            else:
                target = {"id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "name": col["name"], "items": []}
                existing.append(target)
                existing_map[target["name"]] = target
                stats["collections_created"] += 1
            existing_set = {(i["prompt"], i.get("size", ""), i.get("quality", "")) for i in target["items"]}
            for item in col.get("items", []):
                key = (item["prompt"], item.get("size", ""), item.get("quality", ""))
                if key in existing_set:
                    stats["prompt_skipped"] += 1
                else:
                    item["id"] = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    target["items"].append(item)
                    existing_set.add(key)
                    stats["prompt_added"] += 1
        _save_fav(user, "prompt", existing)

    # Import images
    if "image_favorites.json" in zf.namelist():
        imported = json.loads(zf.read("image_favorites.json"))
        existing = _load_fav(user, "image")
        existing_map = {c["name"]: c for c in existing}
        udir = _user_dir(user)
        for col in imported:
            if col["name"] in existing_map:
                target = existing_map[col["name"]]
            else:
                target = {"id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "name": col["name"], "items": []}
                existing.append(target)
                existing_map[target["name"]] = target
                stats["collections_created"] += 1
            existing_files = {Path(i["path"]).name for i in target["items"]}
            for item in col.get("items", []):
                fname = Path(item["path"]).name
                if fname in existing_files:
                    stats["image_skipped"] += 1
                else:
                    zip_path = f"images/{fname}"
                    if zip_path in zf.namelist():
                        dest = udir / fname
                        dest.write_bytes(zf.read(zip_path))
                        item["path"] = f"{user}/{fname}"
                    item["id"] = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    target["items"].append(item)
                    existing_files.add(fname)
                    stats["image_added"] += 1
        _save_fav(user, "image", existing)

    zf.close()
    return stats


@app.delete("/api/favorites/clear/{kind}")
async def clear_favorites(kind: str, request: Request):
    if kind not in ("prompt", "image"):
        raise HTTPException(400, "Invalid kind")
    user = _require_user(request)
    data = _load_fav(user, kind)
    total_items = sum(len(c.get("items", [])) for c in data)
    # Collect image paths before clearing
    files_to_check = []
    if kind == "image":
        for col in data:
            for it in col.get("items", []):
                if it.get("path"):
                    files_to_check.append(it["path"])
    _save_fav(user, kind, [])
    if files_to_check:
        _cleanup_orphan_files(user, files_to_check)
    return {"ok": True, "deleted_collections": len(data), "deleted_items": total_items}


@app.post("/api/favorites/upload/{col_id}")
async def upload_to_fav_collection(
    col_id: str,
    request: Request,
    images: list[UploadFile] = File(...),
):
    user = _require_user(request)
    data = _load_fav(user, "image")
    col = next((c for c in data if c["id"] == col_id), None)
    if not col:
        raise HTTPException(404, "Collection not found")
    user_dir = _user_dir(user)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    added = 0
    for idx, img in enumerate(images):
        content = await img.read()
        ext = Path(img.filename).suffix.lower() if img.filename else ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            ext = ".jpg"
        filename = f"fav_{ts}_{idx}_{img.filename or 'img'}"
        # Sanitize filename
        filename = filename.replace("/", "_").replace("\\", "_").replace("..", "_")
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
            filename += ext
        (user_dir / filename).write_bytes(content)
        item = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f") + str(idx),
            "path": f"{user}/{filename}",
            "prompt": "",
            "time": datetime.now().isoformat(),
        }
        col["items"].insert(0, item)
        added += 1
    _save_fav(user, "image", data)
    return {"ok": True, "added": added}


@app.get("/api/favorites/{kind}")
async def get_favorites(kind: str, request: Request):
    if kind not in ("prompt", "image"):
        raise HTTPException(400, "Invalid kind")
    user = _require_user(request)
    return _load_fav(user, kind)


@app.post("/api/favorites/{kind}/collection")
async def create_collection(kind: str, request: Request):
    if kind not in ("prompt", "image"):
        raise HTTPException(400, "Invalid kind")
    user = _require_user(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Name required")
    data = _load_fav(user, kind)
    if any(c["name"] == name for c in data):
        raise HTTPException(409, "Collection name already exists")
    col = {"id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "name": name, "items": []}
    data.append(col)
    _save_fav(user, kind, data)
    return col


@app.put("/api/favorites/{kind}/collection/{col_id}")
async def rename_collection(kind: str, col_id: str, request: Request):
    if kind not in ("prompt", "image"):
        raise HTTPException(400, "Invalid kind")
    user = _require_user(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Name required")
    data = _load_fav(user, kind)
    col = next((c for c in data if c["id"] == col_id), None)
    if not col:
        raise HTTPException(404, "Collection not found")
    if any(c["name"] == name and c["id"] != col_id for c in data):
        raise HTTPException(409, "Collection name already exists")
    col["name"] = name
    _save_fav(user, kind, data)
    return col


@app.delete("/api/favorites/{kind}/collection/{col_id}")
async def delete_collection(kind: str, col_id: str, request: Request):
    if kind not in ("prompt", "image"):
        raise HTTPException(400, "Invalid kind")
    user = _require_user(request)
    data = _load_fav(user, kind)
    # Collect image paths from the collection being deleted
    files_to_check = []
    if kind == "image":
        col = next((c for c in data if c["id"] == col_id), None)
        if col:
            files_to_check = [it["path"] for it in col.get("items", []) if it.get("path")]
    data = [c for c in data if c["id"] != col_id]
    if not data:
        data = [_default_collection()]
    _save_fav(user, kind, data)
    if files_to_check:
        _cleanup_orphan_files(user, files_to_check)
    return {"ok": True}


@app.post("/api/favorites/{kind}/item")
async def add_favorite_item(kind: str, request: Request):
    if kind not in ("prompt", "image"):
        raise HTTPException(400, "Invalid kind")
    user = _require_user(request)
    body = await request.json()
    col_ids = body.get("collection_ids", [])
    data = _load_fav(user, kind)
    item_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    for col in data:
        if col["id"] in col_ids:
            if kind == "prompt":
                item = {
                    "id": item_id,
                    "prompt": body.get("prompt", ""),
                    "size": body.get("size", ""),
                    "quality": body.get("quality", ""),
                    "time": datetime.now().isoformat(),
                }
            else:
                item = {
                    "id": item_id,
                    "path": body.get("path", ""),
                    "prompt": body.get("prompt", ""),
                    "time": datetime.now().isoformat(),
                }
            col["items"].insert(0, item)
    _save_fav(user, kind, data)
    return {"ok": True}


@app.delete("/api/favorites/{kind}/item/{col_id}/{item_id}")
async def delete_favorite_item(kind: str, col_id: str, item_id: str, request: Request):
    if kind not in ("prompt", "image"):
        raise HTTPException(400, "Invalid kind")
    user = _require_user(request)
    data = _load_fav(user, kind)
    col = next((c for c in data if c["id"] == col_id), None)
    if not col:
        raise HTTPException(404, "Collection not found")
    # Collect image path before removing
    files_to_check = []
    if kind == "image":
        item = next((i for i in col["items"] if i["id"] == item_id), None)
        if item and item.get("path"):
            files_to_check.append(item["path"])
    col["items"] = [i for i in col["items"] if i["id"] != item_id]
    _save_fav(user, kind, data)
    if files_to_check:
        _cleanup_orphan_files(user, files_to_check)
    return {"ok": True}


# ── Batch processing ──

BATCH_INPUT_DIR = Path(os.getenv("BATCH_INPUT_DIR", "batch_input"))
BATCH_INPUT_DIR.mkdir(exist_ok=True)
BATCH_TEMP_DIR = OUTPUT_DIR / "_batch_temp"
BATCH_TEMP_DIR.mkdir(exist_ok=True)


def _user_batch_temp(user: str) -> Path:
    d = BATCH_TEMP_DIR / user
    d.mkdir(exist_ok=True)
    return d

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
_batch_jobs: dict[str, dict] = {}  # user -> batch job state
_batch_lock = threading.Lock()


def _mime_from_ext(fp: Path) -> str:
    ext = fp.suffix.lower()
    return {"png": "image/png", "webp": "image/webp", "gif": "image/gif"}.get(ext.lstrip("."), "image/jpeg")


@app.get("/api/batch/scan")
async def batch_scan(request: Request):
    """Scan a directory for images."""
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    path = request.query_params.get("path", "").strip()
    scan_dir = Path(path) if path else BATCH_INPUT_DIR
    if not scan_dir.exists():
        return {"files": [], "dir": str(scan_dir.resolve()), "error": "目录不存在"}
    if not scan_dir.is_dir():
        return {"files": [], "dir": str(scan_dir.resolve()), "error": "路径不是目录"}
    files = []
    for f in sorted(scan_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
            files.append({"name": f.name, "size": f.stat().st_size, "path": str(f.resolve())})
    return {"files": files, "dir": str(scan_dir.resolve())}


@app.post("/api/batch/upload-fixed")
async def batch_upload_fixed(request: Request, image: UploadFile = File(...)):
    """Upload a fixed image for batch processing, returns server-side path."""
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    content = await image.read()
    safe_name = (image.filename or "image.jpg").replace("..", "_").replace("/", "_").replace("\\", "_")
    user_temp = _user_batch_temp(user)
    dest = user_temp / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest.write_bytes(content)
    return {"path": str(dest.resolve()), "name": safe_name}


@app.post("/api/batch/upload-enum")
async def batch_upload_enum(request: Request):
    """Upload multiple images for an enumerate slot, returns list of file info."""
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    form = await request.form()
    images = [v for k, v in form.multi_items() if k in ("images", "images[]")]
    if not images:
        raise HTTPException(400, "No images uploaded")
    user_temp = _user_batch_temp(user)
    # Clear previous temp files to avoid accumulation
    shutil.rmtree(user_temp, ignore_errors=True)
    user_temp.mkdir(exist_ok=True)
    results = []
    for image in images:
        content = await image.read()
        safe_name = (getattr(image, 'filename', None) or "image.jpg").replace("..", "_").replace("/", "_").replace("\\", "_")
        dest = user_temp / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        dest.write_bytes(content)
        results.append({"name": safe_name, "size": len(content), "path": str(dest.resolve())})
    return {"files": results}


@app.post("/api/batch/start")
async def batch_start(request: Request):
    """Start a batch processing job with multi-slot support."""
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    size = body.get("size", "1024x1024")
    quality = body.get("quality", "auto")
    n_count = body.get("n", 1)
    slots = body.get("slots", [])

    if not prompt:
        raise HTTPException(400, "Prompt required")
    if not slots:
        raise HTTPException(400, "No image slots configured")

    # Validate slots and compute iteration count
    enum_counts = []
    for i, slot in enumerate(slots):
        if slot["type"] == "enumerate":
            if not slot.get("files"):
                raise HTTPException(400, f"槽位 {i+1} 枚举目录为空")
            enum_counts.append(len(slot["files"]))
        elif slot["type"] == "fixed":
            if not slot.get("path"):
                raise HTTPException(400, f"槽位 {i+1} 未设置固定图片")
            fixed_path = Path(slot["path"]).resolve()
            if not fixed_path.exists():
                raise HTTPException(400, f"槽位 {i+1} 图片不存在: {slot['path']}")
            user_temp = _user_batch_temp(user).resolve()
            if str(fixed_path).startswith(str(user_temp)):
                pass  # uploaded by this user, OK
            elif not str(fixed_path).startswith(str(BATCH_TEMP_DIR.resolve())):
                pass  # external path, OK
            else:
                raise HTTPException(403, f"槽位 {i+1} 无权访问该图片")
        else:
            raise HTTPException(400, f"Unknown slot type: {slot['type']}")

    if not enum_counts:
        raise HTTPException(400, "至少需要一个枚举槽位")
    iteration_count = min(enum_counts)

    with _batch_lock:
        existing = _batch_jobs.get(user)
        if existing and not existing.get("done"):
            raise HTTPException(409, "已有批量任务正在运行")
        job = {
            "id": uuid.uuid4().hex[:12],
            "user": user,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n_count,
            "slots": slots,
            "total": iteration_count,
            "completed": 0,
            "failed": 0,
            "current": None,
            "results": [],
            "errors": [],
            "done": False,
            "paused": False,
            "stop_requested": False,
            "started_at": time.time(),
        }
        _batch_jobs[user] = job
    _executor.submit(_run_batch_job, user)
    return {"job_id": job["id"], "total": iteration_count}


def _run_batch_job(user: str):
    """Run batch processing in background thread with multi-slot support."""
    with _batch_lock:
        job = _batch_jobs.get(user)
    if not job:
        return
    slots = job["slots"]

    # Pre-load fixed images into memory
    fixed_cache: dict[int, tuple] = {}
    for i, slot in enumerate(slots):
        if slot["type"] == "fixed":
            fp = Path(slot["path"])
            fixed_cache[i] = (fp.name, fp.read_bytes(), _mime_from_ext(fp))

    # Collect enumerate file lists
    enum_files: dict[int, list] = {}
    for i, slot in enumerate(slots):
        if slot["type"] == "enumerate":
            enum_files[i] = slot["files"]

    for iter_idx in range(job["total"]):
        # Check for pause / stop
        while job.get("paused"):
            if job.get("stop_requested"):
                break
            time.sleep(1)
        if job.get("stop_requested"):
            break

        # Build ordered image list for this iteration
        image_files = []
        enum_names = []
        for i, slot in enumerate(slots):
            if slot["type"] == "fixed":
                image_files.append(fixed_cache[i])
            else:
                files = enum_files[i]
                fp = Path(files[iter_idx % len(files)])
                image_files.append((fp.name, fp.read_bytes(), _mime_from_ext(fp)))
                enum_names.append(fp.name)

        display_name = " + ".join(enum_names) if enum_names else f"iter_{iter_idx+1}"
        job["current"] = {"index": iter_idx, "name": display_name, "started_at": time.time(), "attempt": 0}

        try:
            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                job["current"]["attempt"] = attempt
                print(f"[Batch] user={user} iter={iter_idx+1}/{job['total']} attempt={attempt} file={display_name}")
                try:
                    result = client.images.edit(
                        model=deployment,
                        prompt=job["prompt"],
                        image=image_files,
                        n=job["n"],
                        size=job["size"],
                        quality=job["quality"],
                    )
                    break
                except APIStatusError as e:
                    print(f"[Batch] API error: status={e.status_code} msg={e.message}")
                    if _is_retryable(e) and attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    raise
                except APITimeoutError:
                    print(f"[Batch] Timeout on attempt {attempt}")
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    raise

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            adir = _user_dir(user)
            prefix = f"{user}/"
            saved_files = []
            result_images = []
            for ridx, item in enumerate(result.data):
                b64 = item.b64_json
                if b64:
                    filename = f"{ts}_batch_{iter_idx+1}_{ridx+1}.jpg"
                    (adir / filename).write_bytes(base64.b64decode(b64))
                    saved_files.append(f"{prefix}{filename}")
                    result_images.append(f"/output/{prefix}{filename}")

            del result
            gc.collect()

            if saved_files:
                record = {
                    "id": ts + f"_b{iter_idx}",
                    "time": datetime.now().isoformat(),
                    "prompt": job["prompt"],
                    "size": job["size"],
                    "quality": job["quality"],
                    "n": job["n"],
                    "type": "edit",
                    "input_images": enum_names,
                    "files": saved_files,
                    "batch": True,
                }
                with _get_history_lock(user):
                    history = _load_history(user)
                    history.insert(0, record)
                    _save_history(user, history)

            job["results"].append({"file": display_name, "images": result_images})
            job["completed"] += 1

        except Exception as e:
            traceback.print_exc()
            job["errors"].append({"file": display_name, "error": str(e), "iter_idx": iter_idx})
            job["failed"] += 1

    job["current"] = None
    if not job["done"]:
        job["done"] = True
        job["finished_at"] = time.time()

    # Cleanup user batch temp files only if no failures (user might retry)
    if not job["errors"]:
        try:
            user_temp = _user_batch_temp(user)
            shutil.rmtree(user_temp, ignore_errors=True)
        except Exception:
            pass


@app.get("/api/batch/status")
async def batch_status(request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    with _batch_lock:
        job = _batch_jobs.get(user)
    if not job:
        return {"active": False}
    current = job["current"]
    if current and current.get("started_at"):
        current = {**current, "elapsed": round(time.time() - current["started_at"])}
    return {
        "active": True,
        "job_id": job["id"],
        "total": job["total"],
        "completed": job["completed"],
        "failed": job["failed"],
        "current": current,
        "done": job["done"],
        "paused": job["paused"],
        "results": job["results"][-5:],
        "errors": job["errors"],
    }


@app.post("/api/batch/pause")
async def batch_pause(request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    with _batch_lock:
        job = _batch_jobs.get(user)
    if not job or job["done"]:
        raise HTTPException(404, "No active batch job")
    job["paused"] = not job["paused"]
    return {"paused": job["paused"]}


@app.post("/api/batch/stop")
async def batch_stop(request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    with _batch_lock:
        job = _batch_jobs.get(user)
    if not job or job["done"]:
        raise HTTPException(404, "No active batch job")
    job["stop_requested"] = True
    job["paused"] = False
    job["done"] = True
    job["finished_at"] = time.time()
    return {"ok": True}


@app.post("/api/batch/retry")
async def batch_retry(request: Request):
    """Retry all failed iterations from the last batch job."""
    user = _get_user(request)
    if not user:
        raise HTTPException(403, "Login required")
    with _batch_lock:
        job = _batch_jobs.get(user)
    if not job:
        raise HTTPException(404, "No batch job found")
    if not job.get("done"):
        raise HTTPException(409, "批量任务尚未完成")
    if not job.get("errors"):
        raise HTTPException(400, "没有失败的项目")

    # Build retry job reusing same config, only re-running failed iterations
    failed_indices = [e["iter_idx"] for e in job["errors"] if "iter_idx" in e]
    if not failed_indices:
        raise HTTPException(400, "失败记录缺少索引信息，无法重试")

    with _batch_lock:
        retry_job = {
            "id": uuid.uuid4().hex[:12],
            "user": user,
            "prompt": job["prompt"],
            "size": job["size"],
            "quality": job["quality"],
            "n": job["n"],
            "slots": job["slots"],
            "total": job["completed"] + len(failed_indices),
            "completed": job["completed"],  # carry over previous successes
            "failed": 0,
            "current": None,
            "results": job["results"][:],  # keep previous results
            "errors": [],
            "done": False,
            "paused": False,
            "stop_requested": False,
            "started_at": time.time(),
            "_retry_indices": failed_indices,
        }
        _batch_jobs[user] = retry_job
    _executor.submit(_run_batch_retry, user)
    return {"job_id": retry_job["id"], "retry_count": len(failed_indices)}


def _run_batch_retry(user: str):
    """Run batch retry for failed iterations only."""
    with _batch_lock:
        job = _batch_jobs.get(user)
    if not job:
        return
    slots = job["slots"]
    retry_indices = job["_retry_indices"]

    # Pre-load fixed images into memory
    fixed_cache: dict[int, tuple] = {}
    for i, slot in enumerate(slots):
        if slot["type"] == "fixed":
            fp = Path(slot["path"])
            if fp.exists():
                fixed_cache[i] = (fp.name, fp.read_bytes(), _mime_from_ext(fp))

    # Collect enumerate file lists
    enum_files: dict[int, list] = {}
    for i, slot in enumerate(slots):
        if slot["type"] == "enumerate":
            enum_files[i] = slot["files"]

    for retry_idx, iter_idx in enumerate(retry_indices):
        while job.get("paused"):
            if job.get("stop_requested"):
                break
            time.sleep(1)
        if job.get("stop_requested"):
            break

        # Build image list for this iteration
        image_files = []
        enum_names = []
        for i, slot in enumerate(slots):
            if slot["type"] == "fixed":
                image_files.append(fixed_cache[i])
            else:
                files = enum_files[i]
                fp = Path(files[iter_idx % len(files)])
                image_files.append((fp.name, fp.read_bytes(), _mime_from_ext(fp)))
                enum_names.append(fp.name)

        display_name = " + ".join(enum_names) if enum_names else f"iter_{iter_idx+1}"
        job["current"] = {"index": retry_idx, "name": f"[重试] {display_name}", "started_at": time.time(), "attempt": 0}

        try:
            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                job["current"]["attempt"] = attempt
                print(f"[Batch-Retry] user={user} retry={retry_idx+1}/{len(retry_indices)} iter={iter_idx+1} attempt={attempt}")
                try:
                    result = client.images.edit(
                        model=deployment,
                        prompt=job["prompt"],
                        image=image_files,
                        n=job["n"],
                        size=job["size"],
                        quality=job["quality"],
                    )
                    break
                except APIStatusError as e:
                    print(f"[Batch-Retry] API error: status={e.status_code} msg={e.message}")
                    if _is_retryable(e) and attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    raise
                except APITimeoutError:
                    print(f"[Batch-Retry] Timeout on attempt {attempt}")
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue
                    raise

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            adir = _user_dir(user)
            prefix = f"{user}/"
            saved_files = []
            result_images = []
            for ridx, item in enumerate(result.data):
                b64 = item.b64_json
                if b64:
                    filename = f"{ts}_batch_{iter_idx+1}_{ridx+1}.jpg"
                    (adir / filename).write_bytes(base64.b64decode(b64))
                    saved_files.append(f"{prefix}{filename}")
                    result_images.append(f"/output/{prefix}{filename}")

            del result
            gc.collect()

            if saved_files:
                record = {
                    "id": ts + f"_b{iter_idx}",
                    "time": datetime.now().isoformat(),
                    "prompt": job["prompt"],
                    "size": job["size"],
                    "quality": job["quality"],
                    "n": job["n"],
                    "type": "edit",
                    "input_images": enum_names,
                    "files": saved_files,
                    "batch": True,
                }
                with _get_history_lock(user):
                    history = _load_history(user)
                    history.insert(0, record)
                    _save_history(user, history)

            job["results"].append({"file": display_name, "images": result_images})
            job["completed"] += 1

        except Exception as e:
            traceback.print_exc()
            job["errors"].append({"file": display_name, "error": str(e), "iter_idx": iter_idx})
            job["failed"] += 1

    job["current"] = None
    if not job["done"]:
        job["done"] = True
        job["finished_at"] = time.time()

    # Cleanup temp files if no more failures
    if not job["errors"]:
        try:
            user_temp = _user_batch_temp(user)
            shutil.rmtree(user_temp, ignore_errors=True)
        except Exception:
            pass


# Serve output images
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Build version for cache busting
_version_file = BASE_DIR / "version.txt"
BUILD_VERSION = _version_file.read_text().strip() if _version_file.exists() else datetime.now().strftime("%Y%m%d%H%M%S")

# Load index.html at startup — inject version
_index_html = (BASE_DIR / "index.html").read_text(encoding="utf-8").replace("__BUILD_VERSION__", BUILD_VERSION)

# Load batch.html at startup
_batch_html = (BASE_DIR / "batch.html").read_text(encoding="utf-8") if (BASE_DIR / "batch.html").exists() else ""

# Load sw.js at startup — inject version
_sw_js = (BASE_DIR / "sw.js").read_text(encoding="utf-8").replace("__BUILD_VERSION__", BUILD_VERSION)


@app.get("/")
async def index():
    return HTMLResponse(_index_html, headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/batch")
async def batch_page():
    return HTMLResponse(_batch_html, headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/favicon.ico")
async def favicon_ico():
    return FileResponse(BASE_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/favicon.png")
async def favicon_png():
    return FileResponse(BASE_DIR / "favicon.png", media_type="image/png")


@app.get("/manifest.json")
async def manifest():
    return FileResponse(BASE_DIR / "manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
async def service_worker():
    from fastapi.responses import Response
    return Response(_sw_js, media_type="application/javascript", headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/icon-192.png")
async def icon_192():
    return FileResponse(BASE_DIR / "icon-192.png", media_type="image/png")


@app.get("/icon-512.png")
async def icon_512():
    return FileResponse(BASE_DIR / "icon-512.png", media_type="image/png")
