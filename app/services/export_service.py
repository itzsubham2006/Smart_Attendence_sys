import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")


def export_all(attendance_records, class_name="Class"):
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    if attendance_records:
        if isinstance(attendance_records[0], str):
            df = pd.DataFrame({"Name": attendance_records, "Status": ["Present"] * len(attendance_records)})
        else:
            df = pd.DataFrame(attendance_records)
    else:
        df = pd.DataFrame({"Name": [], "Status": []})

    csv_path = os.path.join(EXPORTS_DIR, "attendance.csv")
    xlsx_path = os.path.join(EXPORTS_DIR, "attendance.xlsx")
    pdf_path = os.path.join(EXPORTS_DIR, "attendance.pdf")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    doc = SimpleDocTemplate(
        pdf_path, pagesize=landscape(A4),
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=20, alignment=TA_CENTER,
    )
    header_style = ParagraphStyle(
        "Header", parent=styles["Normal"],
        fontSize=10, textColor=colors.white, alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "Cell", parent=styles["Normal"],
        fontSize=10, alignment=TA_CENTER,
    )

    elements = []
    elements.append(Paragraph(f"Attendance Report - {class_name}", title_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Paragraph(f"Total Present: {len(attendance_records)}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    table_data = [[Paragraph(str(col), header_style) for col in df.columns]]
    for _, row in df.iterrows():
        table_data.append([Paragraph(str(val), cell_style) for val in row])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))

    elements.append(table)
    doc.build(elements)

    return {"csv": csv_path, "xlsx": xlsx_path, "pdf": pdf_path}
