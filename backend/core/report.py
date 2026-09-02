"""
report.py
---------
Generates a 1-page PDF incident report when an attack is detected.
"""

from fpdf import FPDF


def generate_incident_report(entry: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(180, 30, 30)
    pdf.cell(0, 12, "VoiceSentinel - Incident Report", ln=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(4)

    rows = [
        ("Timestamp (UTC)", entry["timestamp"]),
        ("User / Claimed Identity", entry["username"]),
        ("Decision", entry["decision"]),
        ("Composite Risk Score", f"{entry['risk_score']:.1f} / 100"),
        ("Spoof Confidence", f"{entry['spoof_confidence']*100:.1f}%"),
        (
            "Speaker Similarity",
            f"{entry['speaker_similarity']*100:.1f}%" if entry["speaker_similarity"] is not None else "N/A",
        ),
        ("Detector Mode", entry["detector_mode"]),
        ("Chain-of-custody hash", entry["this_hash"]),
        ("Previous hash", entry["prev_hash"]),
    ]

    for label, value in rows:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(60, 9, label, border=0)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 9, str(value))

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 6,
        "This report was generated automatically by VoiceSentinel's risk engine. "
        "The chain-of-custody hash can be independently recomputed from the audit "
        "log to confirm this entry has not been altered since it was recorded.",
    )

    return bytes(pdf.output())
