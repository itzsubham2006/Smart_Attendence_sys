import os
import time
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
import aiofiles

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from app.services.recognition_service import recognize_faces
from app.services.export_service import export_all
from app.scripts.generate_embeddings import generate_embeddings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
EMBEDDINGS_FOLDER = os.path.join(PROJECT_ROOT, "embeddings")
EXPORTS_FOLDER = os.path.join(PROJECT_ROOT, "exports")
FAISS_INDEX = os.path.join(EMBEDDINGS_FOLDER, "faiss_index.bin")
LABELS_FILE = os.path.join(EMBEDDINGS_FOLDER, "labels.pkl")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "heic", "heif", "avif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
os.makedirs(EXPORTS_FOLDER, exist_ok=True)

app = FastAPI(title="Smart Attendance System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(PROJECT_ROOT, "app", "templates"))


@app.on_event("startup")
async def startup():
    ensure_embeddings()
    print("Model pre-loaded. Ready for requests.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_embeddings():
    if not (os.path.exists(FAISS_INDEX) and os.path.exists(LABELS_FILE)):
        print("Embeddings not found. Generating now...")
        start = time.time()
        generate_embeddings()
        print(f"Embeddings generated in {time.time() - start:.1f}s")
    else:
        from app.services.recognition_service import _load_index_and_labels
        try:
            _load_index_and_labels()
            print("Embeddings loaded. FAISS index ready.")
        except Exception as e:
            print(f"Error loading embeddings: {e}. Regenerating...")
            generate_embeddings()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
        return templates.TemplateResponse(request, "index.html", {
            "results": None,
            "message": None,
            "msg_type": None,
            "image": None,
            "present_count": 0,
        })


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    if file.filename is None or file.filename == "":
        return templates.TemplateResponse(request, "index.html", {
            "results": None, "message": "No file uploaded", "msg_type": "error", "image": None, "present_count": 0,
        })

    if not allowed_file(file.filename):
        return templates.TemplateResponse(request, "index.html", {
            "results": None,
            "message": "Invalid file type. Please upload jpg/jpeg/png/webp",
            "msg_type": "error",
            "image": None,
            "present_count": 0,
        })

    safe_name = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

  
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext in ("heic", "heif"):
        try:
            from PIL import Image
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = Image.open(file_path)
            jpg_path = file_path.rsplit(".", 1)[0] + ".jpg"
            img.save(jpg_path, "JPEG", quality=92)
            os.remove(file_path)
            file_path = jpg_path
        except Exception as e:
            return templates.TemplateResponse(request, "index.html", {
                "results": None, "message": f"Could not process HEIC image: {str(e)}",
                "msg_type": "error", "image": None, "present_count": 0,
            })

    try:
        start = time.time()
        results, attendance, output_image = recognize_faces(file_path)
        elapsed = time.time() - start

        if attendance:
            export_all(attendance)
            message = f"Attendance marked. Present: {len(attendance)} | Processed in {elapsed:.1f}s"
            msg_type = "success"
        else:
            message = f"No known faces recognized. ({elapsed:.1f}s)"
            msg_type = "info"

        return templates.TemplateResponse(request, "index.html", {
            "results": results,
            "present_count": len(attendance),
            "message": message,
            "msg_type": msg_type,
            "image": output_image,
            "processing_time": round(elapsed, 1),
        })

    except Exception as e:
        return templates.TemplateResponse(request, "index.html", {
            "results": None,
            "message": f"Error: {str(e)}",
            "msg_type": "error",
            "image": None,
            "present_count": 0,
        })


@app.get("/download/csv")
async def download_csv():
    path = os.path.join(EXPORTS_FOLDER, "attendance.csv")
    if os.path.exists(path):
        return FileResponse(path, filename="attendance.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="No CSV file found")


@app.get("/download/excel")
async def download_excel():
    path = os.path.join(EXPORTS_FOLDER, "attendance.xlsx")
    if os.path.exists(path):
        return FileResponse(path, filename="attendance.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    raise HTTPException(status_code=404, detail="No Excel file found")


@app.get("/download/pdf")
async def download_pdf():
    path = os.path.join(EXPORTS_FOLDER, "attendance.pdf")
    if os.path.exists(path):
        return FileResponse(path, filename="attendance.pdf", media_type="application/pdf")
    raise HTTPException(status_code=404, detail="No PDF file found")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
