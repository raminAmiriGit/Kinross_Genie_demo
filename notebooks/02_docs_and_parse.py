# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Inspection reports (unstructured) → parsed table
# MAGIC Generates synthetic maintenance **inspection report PDFs** into a Volume, then
# MAGIC extracts text with `ai_parse_document` and lands a structured
# MAGIC `maintenance_reports_parsed` table. The **ignored vibration warning** on
# MAGIC PARA-SAG-2 lives here — this is the unstructured source Genie-1 can cite that a
# MAGIC plain Genie space cannot read.

# COMMAND ----------
# MAGIC %pip install fpdf2 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
VOL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/raw_docs/inspection_reports"

from datetime import datetime, timedelta
import os
TODAY = datetime.utcnow().date()

# COMMAND ----------
# MAGIC %md ## Define the reports (one is the ignored critical finding)
# COMMAND ----------
reports = [
    dict(report_id="INSP-PARA-SAG2-CRIT", site_id="PARA", site_name="Paracatu",
         equipment_id="PARA-SAG-2", equipment_name="SAG Mill #2",
         report_date=TODAY - timedelta(days=12), inspector="R. Okonkwo",
         finding="Elevated spindle bearing vibration", severity="High",
         reading="Vibration 7.8 mm/s (alarm threshold 6.0 mm/s); bearing temperature trending up",
         recommendation="Inspect drive-end spindle bearing at next opportunity; signature consistent with bearing wear.",
         disposition="Acknowledged - No Action (deferred to next planned shutdown)"),
    dict(report_id="INSP-PARA-SAG1-01", site_id="PARA", site_name="Paracatu",
         equipment_id="PARA-SAG-1", equipment_name="SAG Mill #1",
         report_date=TODAY - timedelta(days=20), inspector="R. Okonkwo",
         finding="Routine vibration survey", severity="Low",
         reading="Vibration 3.6 mm/s, within limits", recommendation="No action required.",
         disposition="Closed"),
    dict(report_id="INSP-TASI-SAG2-01", site_id="TASI", site_name="Tasiast",
         equipment_id="TASI-SAG-2", equipment_name="SAG Mill #2",
         report_date=TODAY - timedelta(days=18), inspector="M. Diallo",
         finding="Liner wear check", severity="Medium",
         reading="Liner wear at 62% of service life", recommendation="Schedule reline within 6 weeks.",
         disposition="Work order raised"),
    dict(report_id="INSP-FKNX-BALL1-01", site_id="FKNX", site_name="Fort Knox",
         equipment_id="FKNX-BALL-1", equipment_name="Ball Mill #1",
         report_date=TODAY - timedelta(days=25), inspector="J. Whitefeather",
         finding="Lubrication system inspection", severity="Low",
         reading="Oil pressure and temperature nominal", recommendation="No action required.",
         disposition="Closed"),
]

# COMMAND ----------
# MAGIC %md ## Render PDFs into the Volume
# COMMAND ----------
from fpdf import FPDF
os.makedirs(VOL_DIR, exist_ok=True)

def render_pdf(r):
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    W = pdf.epw  # effective page width — use explicit width everywhere (avoids w=0 edge cases)
    def line(txt, size=11, style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(W, 6 if size <= 11 else 9, txt, new_x="LMARGIN", new_y="NEXT")
    line("Condition Monitoring - Inspection Report", size=15, style="B")
    for label, key in [("Report ID", "report_id"), ("Site", "site_name"),
                       ("Equipment", "equipment_name"), ("Inspector", "inspector")]:
        line(f"{label}: {r[key]}")
    line(f"Date: {r['report_date'].isoformat()}")
    pdf.ln(3)
    line("Finding", size=12, style="B")
    line(f"{r['finding']} (Severity: {r['severity']}).")
    line(f"Reading: {r['reading']}.")
    pdf.ln(2)
    line("Recommendation", size=12, style="B")
    line(r["recommendation"])
    pdf.ln(2)
    line("Disposition", size=12, style="B")
    line(r["disposition"])
    path = f"{VOL_DIR}/{r['report_id']}.pdf"
    pdf.output(path)
    return path

# PDF rendering is a demo flourish; never let it block the parsed table downstream.
pdf_ok = 0
for r in reports:
    try:
        render_pdf(r)
        pdf_ok += 1
    except Exception as e:
        # Fall back to a plain-text document so the Volume still holds the source.
        with open(f"{VOL_DIR}/{r['report_id']}.txt", "w") as fh:
            fh.write("\n".join(f"{k}: {v}" for k, v in r.items()))
        print(f"  PDF render failed for {r['report_id']} ({str(e)[:80]}); wrote .txt instead")
print(f"documents written to {VOL_DIR} (pdf={pdf_ok}/{len(reports)})")

# COMMAND ----------
# MAGIC %md ## Demonstrate ai_parse_document over the Volume (best-effort)
# COMMAND ----------
parsed_ok = False
try:
    raw = spark.sql(f"""
        SELECT path, ai_parse_document(content) AS parsed
        FROM READ_FILES('{VOL_DIR}/*.pdf', format => 'binaryFile')
    """)
    raw.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.maintenance_reports_raw")
    parsed_ok = True
    print("ai_parse_document succeeded ->", f"{FQ}.maintenance_reports_raw")
except Exception as e:
    print("ai_parse_document unavailable, continuing with authored text. Reason:", str(e)[:200])

# COMMAND ----------
# MAGIC %md ## Structured parsed table (deterministic — always populated)
# COMMAND ----------
import pandas as pd
rows = []
for r in reports:
    raw_text = (f"Condition Monitoring Inspection Report. Report ID {r['report_id']}. "
                f"Site {r['site_name']}. Equipment {r['equipment_name']} ({r['equipment_id']}). "
                f"Inspector {r['inspector']}. Date {r['report_date'].isoformat()}. "
                f"Finding: {r['finding']} (Severity {r['severity']}). Reading: {r['reading']}. "
                f"Recommendation: {r['recommendation']} Disposition: {r['disposition']}.")
    rows.append((r["report_id"], r["site_id"], r["equipment_id"], r["report_date"],
                 r["finding"], r["severity"], r["recommendation"], r["disposition"], raw_text))

parsed = pd.DataFrame(rows, columns=[
    "report_id", "site_id", "equipment_id", "report_date", "finding",
    "severity", "recommendation", "disposition", "raw_text"])
(spark.createDataFrame(parsed)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{FQ}.maintenance_reports_parsed"))
spark.sql(f"COMMENT ON TABLE {FQ}.maintenance_reports_parsed IS "
          f"'Parsed maintenance inspection reports. raw_text is the full document text for retrieval.'")
print(f"wrote {FQ}.maintenance_reports_parsed: {len(rows)} rows (ai_parse_document={parsed_ok})")

# COMMAND ----------
display(spark.sql(f"SELECT report_id, equipment_id, severity, recommendation FROM {FQ}.maintenance_reports_parsed"))
