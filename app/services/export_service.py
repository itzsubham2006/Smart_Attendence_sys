import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table

def export_all(attendance):
    

    df = pd.DataFrame(attendance)

    df.to_csv("exports/attendance.csv", index=False)
    df.to_excel("exports/attendance.xlsx", index=False)

    pdf = SimpleDocTemplate("exports/attendance.pdf")
    data = [df.columns.tolist()] + df.values.tolist()
    
    if not attendance:
        print(" No attendance data to export")
        return
    
    table = Table(data)
    pdf.build([table])

    print("Exported all formats")