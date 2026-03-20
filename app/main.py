from flask import Flask, request, render_template, send_file
import os

from app.services.recognition_service import recognize_faces
from app.services.export_service import export_all
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    attendance = recognize_faces(path)
    export_all(attendance)

    return render_template("index.html")

@app.route("/download/csv")
def csv():
    return send_file("exports/attendance.csv", as_attachment=True)

@app.route("/download/excel")
def excel():
    return send_file("exports/attendance.xlsx", as_attachment=True)

@app.route("/download/pdf")
def pdf():
    return send_file("exports/attendance.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)