from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from app.services.recognition_service import recognize_faces
from app.services.export_service import export_all
from app.scripts.generate_embeddings import generate_embeddings

app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
EMBEDDINGS_FOLDER = os.path.join(PROJECT_ROOT, "embeddings")
EXPORTS_FOLDER = os.path.join(PROJECT_ROOT, "exports")
FAISS_INDEX = os.path.join(EMBEDDINGS_FOLDER, "faiss_index.bin")
LABELS_FILE = os.path.join(EMBEDDINGS_FOLDER, "labels.pkl")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
os.makedirs(EXPORTS_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_embeddings():
    if not (os.path.exists(FAISS_INDEX) and os.path.exists(LABELS_FILE)):
        print("Embeddings not found. Generating now...")
        generate_embeddings()
    else:
        print("Embeddings found. Skipping regeneration.")


@app.route("/")
def home():
    return render_template("index.html", results=None, message=None, image=None)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if file is None or file.filename == "":
        return render_template("index.html", results=None, message="No file uploaded", image=None)

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            results=None,
            message="Invalid file type. Please upload jpg/jpeg/png/webp",
            image=None
        )

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        ensure_embeddings()

        # Returns: (per_face_results, recognized_unique_students, output_image_path)
        results, attendance, output_image = recognize_faces(file_path)

        if attendance:
            export_all(attendance)  # your existing exporter: CSV + Excel + PDF
            message = f"Attendance marked successfully. Present: {len(attendance)}"
        else:
            message = "No known faces recognized."

        return render_template(
            "index.html",
            results=results,
            present_count=len(attendance),
            message=message,
            image=output_image
        )

    except Exception as e:
        return render_template(
            "index.html",
            results=None,
            message=f"Error during processing: {str(e)}",
            image=None
        )


@app.route("/download/csv")
def download_csv():
    file_path = os.path.join(EXPORTS_FOLDER, "attendance.csv")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "No CSV file found", 404


@app.route("/download/excel")
def download_excel():
    file_path = os.path.join(EXPORTS_FOLDER, "attendance.xlsx")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "No Excel file found", 404


@app.route("/download/pdf")
def download_pdf():
    file_path = os.path.join(EXPORTS_FOLDER, "attendance.pdf")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "No PDF file found", 404


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)