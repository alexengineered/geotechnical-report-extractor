# Geotechnical Report Summary Extractor

Streamlit app that extracts **key geotechnical design parameters** from **text-based PDF reports** and generates concise **PDF and CSV summaries** for rapid engineering review.

Built for engineers who want answers quickly without reading an entire report.

---

## What it does

Extracts high-value information commonly needed for preliminary design and peer review:

• Project metadata including project name, location, client, report date, and file or job number
• Allowable bearing capacity from tables or narrative text
• Settlement-related tables when present
• Groundwater depth and “not encountered” statements
• Inventory of borings with detected depths and groundwater notes
• Identifies pages likely containing graphical boring logs for manual review
• Quality control flags for found versus not found parameters

All extracted values retain source page traceability.

---

## Input and output

**Input**
• One geotechnical PDF report

**Outputs**
• CSV file with extracted values, source page, and extraction method
• Formatted PDF summary suitable for quick review or sharing

---

## Important limitations

• **Text-based PDFs only. No OCR.**
• Scanned or image-only reports are intentionally rejected
• Table extraction is conservative to avoid false positives
• Graphical boring logs are flagged but not interpreted
• Scope is intentionally limited and not a full report parser

---

## Tracked parameters

• Allowable bearing capacity
• Friction angle
• Unit weight
• Groundwater depth

Basic plausibility checks are applied and flagged when values appear out of range.

---

## Install

Requires **Python 3.9 or newer**.

```bash
pip install streamlit PyMuPDF pdfplumber pandas reportlab
```

If you encounter a `fitz` import error:

```bash
pip uninstall fitz
pip install PyMuPDF
```

---

## Run

```bash
streamlit run geotech.py
```

Open the local Streamlit URL printed in the terminal.

---

## Usage

1. Upload a PDF
2. Review extracted data in the app tabs
3. Export CSV and PDF summaries

Optional: enable or disable table extraction from the sidebar.

---

## Troubleshooting

| Issue                | Solution                                                                          |
| -------------------- | --------------------------------------------------------------------------------- |
| Scanned PDF detected | Re-export as a text-based PDF or run OCR externally                               |
| Missing values       | Not all reports include every parameter; some data exists only in figures or logs |

---

## Disclaimer

This tool assists with document review only.
All extracted values must be verified against the stamped report before use in design.

---


## License

Free to use and adapt for internal business, personal, or educational use.
Not permitted for resale or inclusion in paid products.

Licensed under Creative Commons Attribution–NonCommercial 4.0 (CC BY-NC 4.0).


---

## Feedback 

Found a bug or have a suggestion? Send feedback to contact@alexengineered.com 

--- 

## Author

AlexEngineered

---

*Built for civil engineers who value their time.*
