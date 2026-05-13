import os
import json
import base64
import traceback
import gc
import zipfile
import io
from datetime import datetime
from pathlib import Path
import hashlib
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import AzureOpenAI, APIStatusError
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


@app.post("/api/generate")
async def generate_image(req: ImageRequest, request: Request):
    user = _get_user(request)
    try:
        result = client.images.generate(
            model=deployment,
            prompt=req.prompt,
            n=req.n,
            size=req.size,
            quality=req.quality,
            output_format="jpeg",
        )
        images = []
        saved_files = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Always save to file to avoid OOM; anonymous uses _anon dir
        adir = _user_dir(user) if user else _user_dir("_anon")
        prefix = f"{user}/" if user else "_anon/"

        for idx, item in enumerate(result.data):
            b64 = item.b64_json
            if b64:
                filename = f"{ts}_{idx+1}.jpg"
                (adir / filename).write_bytes(base64.b64decode(b64))
                saved_files.append(f"{prefix}{filename}")
                images.append(f"/output/{prefix}{filename}")
            elif item.url:
                images.append(item.url)

        # Clear API response from memory
        del result
        gc.collect()

        # Record history
        if user and saved_files:
            record = {
                "id": ts,
                "time": datetime.now().isoformat(),
                "prompt": req.prompt,
                "size": req.size,
                "quality": req.quality,
                "n": req.n,
                "files": saved_files,
            }
            history = _load_history(user)
            history.insert(0, record)
            _save_history(user, history)

        return {"images": images}
    except APIStatusError as e:
        # Return full API error details for debugging content filters
        try:
            body = e.response.json()
        except Exception:
            body = e.response.text
        print(f"[API Error] status={e.status_code} body={json.dumps(body, ensure_ascii=False, indent=2)}")
        raise HTTPException(status_code=e.status_code, detail={
            "message": str(e.message),
            "raw": body,
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        # Read uploaded image files with mime types
        image_files = []
        for img in images:
            content = await img.read()
            content_type = img.content_type or "application/octet-stream"
            image_files.append((img.filename, content, content_type))

        result = client.images.edit(
            model=deployment,
            prompt=prompt,
            image=image_files,
            n=n,
            size=size,
            quality=quality,
        )
        # Free upload data
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

        # Clear API response from memory
        del result
        gc.collect()

        # Record history
        if user and saved_files:
            record = {
                "id": ts,
                "time": datetime.now().isoformat(),
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "n": n,
                "type": "edit",
                "input_images": [img.filename for img in images],
                "files": saved_files,
            }
            history = _load_history(user)
            history.insert(0, record)
            _save_history(user, history)

        return {"images": result_images}
    except APIStatusError as e:
        try:
            body = e.response.json()
        except Exception:
            body = e.response.text
        print(f"[API Error] status={e.status_code} body={json.dumps(body, ensure_ascii=False, indent=2)}")
        raise HTTPException(status_code=e.status_code, detail={
            "message": str(e.message),
            "raw": body,
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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


@app.delete("/api/history/{record_id}")
async def delete_history(record_id: str, request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=403, detail="Login required")
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


# Serve output images
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Build version for cache busting
_version_file = BASE_DIR / "version.txt"
BUILD_VERSION = _version_file.read_text().strip() if _version_file.exists() else datetime.now().strftime("%Y%m%d%H%M%S")

# Load index.html at startup — inject version
_index_html = (BASE_DIR / "index.html").read_text(encoding="utf-8").replace("__BUILD_VERSION__", BUILD_VERSION)

# Load sw.js at startup — inject version
_sw_js = (BASE_DIR / "sw.js").read_text(encoding="utf-8").replace("__BUILD_VERSION__", BUILD_VERSION)


@app.get("/")
async def index():
    return HTMLResponse(_index_html, headers={"Cache-Control": "no-cache, must-revalidate"})


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
