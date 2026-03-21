from flask import Flask, request, render_template, send_file
import os

from app.services.recognition_service import recognize_faces
from app.services.export_service import export_all

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html", results=None)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    if not file:
        return render_template("index.html", results=None, message="No file uploaded")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    attendance = recognize_faces(path)

   
    if attendance:
        export_all(attendance)
        message = "Attendance marked successfully"
    else:
        message = " No known faces recognized"

    return render_template("index.html", results=attendance, message=message)


@app.route("/download/csv")
def csv():
    file_path = "exports/attendance.csv"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return " No CSV file found"


@app.route("/download/excel")
def excel():
    file_path = "exports/attendance.xlsx"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "No Excel file found"


@app.route("/download/pdf")
def pdf():
    file_path = "exports/attendance.pdf"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "No PDF file found"


if __name__ == "__main__":
    app.run(debug=True)