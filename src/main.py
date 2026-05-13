import os
import json
import base64
import traceback
import gc
from datetime import datetime
from pathlib import Path
import hashlib
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
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
        udir = _user_dir(user) if user else None
        prefix = f"{user}/" if user else ""

        for idx, item in enumerate(result.data):
            b64 = item.b64_json
            if b64:
                # Save to file first, return file URL instead of base64
                if udir:
                    filename = f"{ts}_{idx+1}.jpg"
                    (udir / filename).write_bytes(base64.b64decode(b64))
                    saved_files.append(f"{prefix}{filename}")
                    images.append(f"/output/{prefix}{filename}")
                else:
                    images.append(f"data:image/jpeg;base64,{b64}")
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
        udir = _user_dir(user) if user else None
        prefix = f"{user}/" if user else ""

        for idx, item in enumerate(result.data):
            b64 = item.b64_json
            if b64:
                if udir:
                    filename = f"{ts}_edit_{idx+1}.jpg"
                    (udir / filename).write_bytes(base64.b64decode(b64))
                    saved_files.append(f"{prefix}{filename}")
                    result_images.append(f"/output/{prefix}{filename}")
                else:
                    result_images.append(f"data:image/png;base64,{b64}")
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


@app.delete("/api/history/{record_id}")
async def delete_history(record_id: str, request: Request):
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=403, detail="Login required")
    history = _load_history(user)
    record = next((r for r in history if r["id"] == record_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    for f in record.get("files", []):
        fp = OUTPUT_DIR / f
        if fp.exists():
            fp.unlink()
    history = [r for r in history if r["id"] != record_id]
    _save_history(user, history)
    return {"ok": True}


# Serve output images
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Load index.html at startup
_index_html = (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/")
async def index():
    return HTMLResponse(_index_html)


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
    return FileResponse(BASE_DIR / "sw.js", media_type="application/javascript")


@app.get("/icon-192.png")
async def icon_192():
    return FileResponse(BASE_DIR / "icon-192.png", media_type="image/png")


@app.get("/icon-512.png")
async def icon_512():
    return FileResponse(BASE_DIR / "icon-512.png", media_type="image/png")
