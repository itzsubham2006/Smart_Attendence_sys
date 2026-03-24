import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table
import os
def export_all(attendance):
    
    df = pd.DataFrame(attendance)
    df.to_csv("exports/attendance.csv", index=False)
    df.to_excel("exports/attendance.xlsx", index=False)
    pdf = SimpleDocTemplate("exports/attendance.pdf")
    data = [df.columns.tolist()] + df.values.tolist()
    
    if not attendance:
        return
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    exports_dir = os.path.join(project_root, "exports")
    os.makedirs(exports_dir, exist_ok=True)
    # Support both list[str] and list[dict] attendance formats
    if isinstance(attendance, list) and attendance and isinstance(attendance[0], str):
        df = pd.DataFrame({"Student": attendance, "Status": "Present"})
    else:
        df = pd.DataFrame(attendance)
    csv_path = os.path.join(exports_dir, "attendance.csv")
    xlsx_path = os.path.join(exports_dir, "attendance.xlsx")
    pdf_path = os.path.join(exports_dir, "attendance.pdf")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    pdf = SimpleDocTemplate(pdf_path)
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    pdf.build([table])

    print("Exported all formats")