"""
Geotechnical Report Summary Extractor
"""

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import tempfile
import os
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from collections import defaultdict

# Verify PyMuPDF
if not hasattr(fitz, "open"):
    st.error("Wrong 'fitz' package installed!")
    st.code("pip uninstall fitz && pip install PyMuPDF")
    st.stop()

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
def load_custom_css():
    """Load custom CSS for dark theme matching target design."""

    st.markdown("""
    <style>
        /*  #51776C sage green, #3E5C5C dark green, #243A47 dark blue*/
    /* Main app background - dark green */
    
    .stApp {
        background-color: #3E5C5C;
    }
    
    .stApp a{
        color: rgba(255, 255, 255, .8);
        text-decoration: none; 
    }
    
        .stApp a:hover{
        color: #b6dbd0;
    }
    
    .stMainBlockContainer {
    padding: 2rem 5rem 10rem;
    }
    
    .stDownloadButton button {
    background-color: #273f4d !important;
    color: white !important;
    width: 156px !important;
    padding: .5rem 0;
    }
    
    .stDownloadButton button:hover {
        background-color: #4A6670 !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar - dark blue */
    [data-testid="stSidebar"] {
        background-color: #243A47;
    }

    [data-testid="stFileUploader"] {
        background: #51776C;
        border-radius: 8px;
        padding: 1rem 2.5rem 2.5rem;
    }

    [data-testid="stFileUploader"] section {
        border: none !important;
    }

    [data-testid="stFileUploader"] section > div {
        color: #333333 !important;
    }

    [data-testid="stFileUploader"] small {
        color: #666666 !important;
    }

    /* Tabs and content - dark blue */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #273f4d;
        padding: 0.5rem;
        border-radius: 10px 10px 0 0;
        margin-bottom: .5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #c0d0c8;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #87CEEB !important;
        min-height: 5px;
        border-radius: 1px;
    }
    
    /* Dark text in tab panels */
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] span,
    .stTabs [data-baseweb="tab-panel"] label,
    .stTabs [data-baseweb="tab-panel"] div {
        color: #ffffff !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
    }

    [data-testid="stElementToolbarButtonContainer"] {
        background: transparent !important;
    }

    /* Markdown text specifically */
    .stMarkdown p, .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Target specific dark text elements */
    [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    [data-testid="stMarkdownContainer"] p.sage {
        color: #b6dbd0 !important;
    }
[data-testid="stMarkdownContainer"] p.footer {
        color: rgba(255, 255, 255, .6) !important;
        font-size: .9rem;
    }

    .stHeading {
        color: #ffffff !important;
        }
    
    .stMarkdown hr {
        display: none;
    }
    
    .stFileUploaderFileName {
    color: #fff;
    }
        
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: #273f4d;
        padding: 1.5rem;
        border-radius: 0 0 10px 10px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #51776C;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #5d8577;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    [data-testid="stMetricLabel"] {
        color: #c0d0c8;
    }

    /* Messages */
    .stSuccess {
        background: rgba(90, 143, 112, 0.3);
        border-left: 4px solid #5a8f70;
        border-radius: 8px;
    }

    .stWarning {
        background: rgba(230, 180, 34, 0.2);
        border-left: 4px solid #e6b422;
        border-radius: 8px;
    }

    .stAlert {
    background-color: rgb(255, 255, 255, .2);
    }
    
    .stError {
        background: rgba(200, 80, 70, 0.2);
        border-left: 4px solid #c85046;
        border-radius: 8px;
    }

    .stInfo {
        background: rgba(100, 140, 130, 0.2);
        border-left: 4px solid #648c82;
        border-radius: 8px;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedValue:
    """Represents a single extracted parameter value."""
    parameter: str
    value: Optional[float] = None
    value_text: Optional[str] = None
    units: str = ""
    units_normalized: str = ""
    location: Optional[str] = None
    source_pages: List[int] = field(default_factory=list)
    confidence: str = "high"
    not_found_reason: Optional[str] = None
    raw_text: str = ""
    extraction_method: str = "text"
    case_type: Optional[str] = None
    validation_status: str = "ok"
    validation_message: Optional[str] = None
    conditions_text: Optional[str] = None
    source_table_name: Optional[str] = None
    source_row_text: Optional[str] = None


@dataclass
class ExtractedTable:
    """Represents a full extracted table."""
    table_name: str
    table_type: str
    headers: List[str]
    rows: List[Dict[str, Any]]
    notes: List[str]
    source_page: int
    raw_text: str


@dataclass
class ReportMetadata:
    """Report-level metadata."""
    project_name: Optional[str] = None
    project_location: Optional[str] = None
    report_date: Optional[str] = None
    prepared_for: Optional[str] = None
    reference_number: Optional[str] = None
    all_identifiers: Dict[str, str] = field(default_factory=dict)
    source_pages: List[int] = field(default_factory=list)


@dataclass
class BoringInfo:
    """Information about a single boring."""
    boring_id: str
    depth: Optional[float] = None
    depth_units: str = "ft"
    groundwater_depth: Optional[float] = None
    groundwater_note: Optional[str] = None
    source_pages: List[int] = field(default_factory=list)
    needs_vision_review: bool = False


@dataclass
class BoringLogPage:
    """Tracks pages that are graphical boring logs."""
    page_num: int
    boring_id: Optional[str] = None
    text_length: int = 0
    has_log_keywords: bool = False
    needs_vision: bool = False
    reason: str = ""


@dataclass
class ValidationResult:
    """Result of range validation."""
    status: str
    message: Optional[str] = None


# =============================================================================
# VALIDATION RANGES
# =============================================================================

VALIDATION_RANGES = {
    "friction_angle": {"min": 10, "max": 55, "warn_low": 15, "warn_high": 50,
                       "fail_message": "Friction angle outside plausible range",
                       "warn_message": "Friction angle at edge of typical range"},
    "allowable_bearing_capacity": {"min": 100, "max": 100000, "warn_low": 500, "warn_high": 50000,
                                   "fail_message": "Bearing capacity outside typical range",
                                   "warn_message": "Bearing capacity unusually low or high"},
    "groundwater_depth": {"min": 0, "max": 500, "warn_low": 0, "warn_high": 200,
                          "fail_message": "Groundwater depth unrealistic",
                          "warn_message": "Groundwater depth unusually deep"},
    "unit_weight_total": {"min": 80, "max": 170, "warn_low": 90, "warn_high": 150,
                          "fail_message": "Unit weight outside plausible range",
                          "warn_message": "Unit weight at edge of typical range"},
}


def validate_value(parameter: str, value: float, units: str = "") -> ValidationResult:
    """Validate an extracted value against expected ranges."""
    if parameter not in VALIDATION_RANGES:
        return ValidationResult(status="ok")
    ranges = VALIDATION_RANGES[parameter]
    converted_value = value
    if parameter == "allowable_bearing_capacity" and "ksf" in units.lower():
        converted_value = value * 1000
    if converted_value < ranges["min"] or converted_value > ranges["max"]:
        return ValidationResult(status="fail", message=ranges["fail_message"])
    if converted_value < ranges["warn_low"] or converted_value > ranges["warn_high"]:
        return ValidationResult(status="warning", message=ranges["warn_message"])
    return ValidationResult(status="ok")

def normalize_for_export(value):
    """Normalize dash characters for CSV/Excel/PDF export."""
    if value is None:
        return value
    text = str(value) if not isinstance(value, str) else value
    return text.replace('\u2014', '-').replace('\u2013', '-')

# =============================================================================
# TEXT EXTRACTION
# =============================================================================
def extract_section_text(pages_text: Dict[int, str], section_titles: List[str]) -> Dict[int, str]:
    """Extract text from specific sections by title."""
    section_text = {}
    for page_num, text in pages_text.items():
        for title in section_titles:
            pattern = rf'{title}.*?(?=\n[A-Z0-9]+\.|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text[page_num] = match.group(0)
    return section_text


def extract_bearing_capacity_from_text(text: str, page_num: int) -> List[ExtractedValue]:
    """Extract bearing capacity from prose, bullets, and inline mentions."""
    results = []

    patterns = [
        r'(?:allowable\s+)?(?:bearing\s+capacity|bearing\s+pressure)[:\s]*(\d{1,3}(?:,\d{3})*)\s*(lbs?/ft[²2]|psf|ksf)',
        r'(?:nominal|ultimate)\s+bearing\s+resistance[:\s]*(\d{1,3}(?:,\d{3})*)\s*(psf|ksf)',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = float(match.group(1).replace(',', ''))
            units_raw = match.group(2).lower()
            units = 'psf' if ('lb' in units_raw or 'ft' in units_raw) else units_raw

            context_start = max(0, match.start() - 150)
            context_end = min(len(text), match.end() + 150)
            context = text[context_start:context_end]

            width_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:feet|ft|foot|inch|in)', context)
            width_text = width_match.group(0) if width_match else "unspecified"

            is_maximum = bool(re.search(r'\b(maximum|governing|shall not exceed)\b', context, re.IGNORECASE))

            validation = validate_value("allowable_bearing_capacity", value, units)

            results.append(ExtractedValue(
                parameter="allowable_bearing_capacity",
                value=value,
                units=units,
                units_normalized=f"{int(value):,} {units}" + (
                    " (maximum)" if is_maximum else f" (footing: {width_text})"
                ),
                source_pages=[page_num],
                confidence="high" if validation.status == "ok" else "medium",
                extraction_method="text",
                case_type="maximum" if is_maximum else "general",
                source_row_text=match.group(0),
                validation_status=validation.status,
                validation_message=validation.message
            ))

    if 'bearing' in text.lower():
        pattern2 = r'(\d{1,3}(?:,\d{3})*)\s*(psf|ksf)'
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            value = float(match.group(1).replace(',', ''))
            units = match.group(2).lower()

            if 100 <= value <= 100000:
                validation = validate_value("allowable_bearing_capacity", value, units)

                results.append(ExtractedValue(
                    parameter="allowable_bearing_capacity",
                    value=value,
                    units=units,
                    units_normalized=f"{int(value):,} {units}",
                    source_pages=[page_num],
                    confidence="medium",
                    extraction_method="text",
                    case_type="general",
                    source_row_text=match.group(0),
                    validation_status=validation.status,
                    validation_message=validation.message
                ))

    return results

@st.cache_data(show_spinner=False)
def extract_text_by_page(pdf_bytes: bytes) -> Dict[int, str]:
    """Extract text from PDF."""
    pages_text = {}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages_text[page_num + 1] = text
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return pages_text

@st.cache_data(show_spinner=False)
def extract_tables_from_pdf(pdf_bytes: bytes, pages_text: Dict[int, str] = None) -> Dict[int, List[List[str]]]:
    """Extract tables using pdfplumber - only on pages with TABLE keyword."""
    tables_by_page = {}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Skip pages without TABLE keyword
                if pages_text and page_num in pages_text:
                    if 'table' not in pages_text[page_num].lower():
                        continue
                tables = page.extract_tables()
                if tables:
                    tables_by_page[page_num] = tables
    except Exception as e:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return tables_by_page


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

@st.cache_data(show_spinner=False)
def extract_metadata(pdf_bytes: bytes, pages_text: Dict[int, str]) -> ReportMetadata:
    """Extract report metadata including file number from footers."""
    metadata = ReportMetadata()

    first_pages_text = "\n".join([pages_text.get(p, "") for p in sorted(pages_text.keys())[:5]])

    for pattern in [
        r'GEOTECHNICAL\s+(?:ENGINEERING\s+)?(?:REPORT|INVESTIGATION)\s*\n\s*(.+?)(?:\n(?:for|NWC|[A-Z]{2,}[a-z]))',
        r'(?:geotechnical\s+(?:engineering\s+)?(?:report|investigation))\s*[:\-]\s*(.+?)(?:\n|$)',
        r'(?:project|site)\s*[:\-]\s*(.+?)(?:\n|$)',
    ]:
        match = re.search(pattern, first_pages_text, re.IGNORECASE | re.MULTILINE)
        if match and not metadata.project_name:
            name = re.sub(r'\s+', ' ', match.group(1).strip())
            if 10 < len(name) < 200:
                metadata.project_name = name
                break

    for pattern in [
        r'(?:location|site|project\s+location|address)\s*[:\-]\s*(.+?)(?:\n|$)',
        r'([A-Za-z][A-Za-z\s]+,\s*(?:[A-Z]{2}|Washington|Oregon|California|Idaho|Montana|Nevada|Arizona|Utah|Colorado|Texas|Florida|New York))',
        r'(\d{1,6}\s+[A-Za-z0-9\s\.]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\.?,?\s*[A-Za-z\s]+,\s*[A-Z]{2})'
    ]:
        match = re.search(pattern, first_pages_text, re.IGNORECASE)
        if match and not metadata.project_location:
            loc = re.sub(r'\s+', ' ', match.group(1)).strip()
            if (5 < len(loc) < 200 and not re.search(r'(date|report|project|prepared)', loc, re.IGNORECASE)):
                metadata.project_location = loc
                break

    for pattern in [
        r'(?:date|dated)\s*[:\-]?\s*(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\w+\s+\d{1,2},?\s+\d{4})',
        r'(?:date|dated)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
        r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})',
    ]:
        match = re.search(pattern, first_pages_text, re.IGNORECASE)
        if match and not metadata.report_date:
            metadata.report_date = match.group(1).strip()
            break

    for pattern in [
        r'(?:prepared\s+for|for)\s*[:\-]?\s*(.+?)(?:\n|$)',
        r'(?:client)\s*[:\-]?\s*(.+?)(?:\n|$)',
    ]:
        match = re.search(pattern, first_pages_text, re.IGNORECASE)
        if match and not metadata.prepared_for:
            client = re.sub(r'\s+', ' ', match.group(1)).strip()
            if (3 < len(client) < 120 and not re.search(r'[/\\]', client)
                and not client.lower().endswith(('.pdf', '.docx', '.xlsx'))
                and not re.match(r'^(date|project|location|report)', client, re.IGNORECASE)):
                metadata.prepared_for = client
                break

    identifier_patterns = [
        (r'Project\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-\.]+)', 'Project Number'),
        (r'File\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-\.]+)', 'File Number'),
        (r'Job\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-\.]+)', 'Job Number'),
        (r'Work\s+Order\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-\.]+)', 'Work Order'),
        (r'Reference\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-\.]+)', 'Reference Number'),
    ]

    found_identifiers = {}

    for pattern, label in identifier_patterns:
        match = re.search(pattern, first_pages_text, re.IGNORECASE)
        if match:
            identifier = match.group(1).strip()
            if 2 < len(identifier) < 50:
                found_identifiers[label] = identifier

    if not found_identifiers:
        for page_num, page_text in pages_text.items():
            footer_area = page_text[-500:] if len(page_text) > 500 else page_text
            for pattern, label in identifier_patterns:
                match = re.search(pattern, footer_area, re.IGNORECASE)
                if match:
                    identifier = match.group(1).strip()
                    if 2 < len(identifier) < 50:
                        found_identifiers[label] = identifier
                        break
            if found_identifiers:
                break

    if found_identifiers:
        metadata.reference_number = list(found_identifiers.values())[0]
        metadata.all_identifiers = found_identifiers
    else:
        metadata.all_identifiers = {}

    metadata.source_pages = list(range(1, min(4, len(pages_text) + 1)))
    return metadata


# =============================================================================
# TABLE EXTRACT MODE
# =============================================================================

def detect_table_title(text: str) -> List[Tuple[str, str, int]]:
    """Detect table titles. Returns list of (title, type, position)."""
    tables_found = []
    seen_table_nums = set()

    pattern = r'(TABLE\s*(\d+)[.\-:\s–—]+([A-Za-z][A-Za-z\s\-\n]+?)(?=\n\s*\n|\n[A-Z][a-z]|\n\d|$))'

    for match in re.finditer(pattern, text, re.IGNORECASE):
        full_title = match.group(1).strip()
        table_num = match.group(2)
        title_text = match.group(3).strip().lower().replace('\n', ' ')
        position = match.start()

        if len(title_text) < 10:
            extended = text[position:position + 200]
            better_match = re.search(r'TABLE\s*\d+[.\-:\s]+([A-Z][A-Z\s\-]{10,})', extended, re.IGNORECASE)
            if better_match:
                title_text = better_match.group(1).strip().lower()
                full_title = better_match.group(0).strip()

        table_type = "unknown"
        if any(kw in title_text for kw in ['settlement', 'deformation']):
            table_type = "settlement"
        elif 'bearing' in title_text or 'allowable' in title_text or 'capacity' in title_text:
            table_type = "bearing_capacity"

        tables_found.append((full_title, table_type, position))
        seen_table_nums.add(table_num)

    pattern2 = r'(Table\s*(\d+)\s+presents\s+[^.]+)'
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        full_title = match.group(1).strip()
        table_num = match.group(2)
        position = match.start()

        if table_num in seen_table_nums:
            continue

        context = text[position:position + 150].lower()
        table_type = "unknown"
        if 'settlement' in context:
            table_type = "settlement"
        elif 'bearing' in context and 'capacity' in context:
            table_type = "bearing_capacity"

        if table_type != "unknown":
            tables_found.append((full_title, table_type, position))
            seen_table_nums.add(table_num)

    return tables_found


def extract_bearing_capacity(
        pages_text: Dict[int, str],
        tables_by_page: Dict[int, List[List[str]]],
        design_tables: Dict[str, List[ExtractedTable]]
) -> List[ExtractedValue]:
    """Extract bearing capacity from tables AND text."""
    results = []

    if design_tables.get('bearing_capacity'):
        for table in design_tables['bearing_capacity']:
            for row in table.rows:
                if 'bearing_capacity' in row:
                    capacity = row['bearing_capacity']
                    width_text = row.get('footing_width', 'unspecified')
                    validation = validate_value("allowable_bearing_capacity", capacity, "psf")

                    results.append(ExtractedValue(
                        parameter="allowable_bearing_capacity",
                        value=capacity,
                        units="psf",
                        units_normalized=f"{capacity:,} psf (footing width: {width_text})",
                        source_pages=[table.source_page],
                        confidence="high",
                        extraction_method="table",
                        source_table_name=table.table_name,
                        source_row_text=row.get('raw_row', ''),
                        validation_status=validation.status,
                        validation_message=validation.message,
                        conditions_text='; '.join(table.notes) if table.notes else None
                    ))

    foundation_sections = extract_section_text(pages_text, [
        'Foundation Design and Construction',
        'Foundation Recommendations',
        'Allowable Bearing'
    ])

    for page_num, text in foundation_sections.items():
        results.extend(extract_bearing_capacity_from_text(text, page_num))

    summary_sections = extract_section_text(pages_text, [
        'Executive Summary',
        'Conclusions and Recommendations',
        'Summary'
    ])

    for page_num, text in summary_sections.items():
        results.extend(extract_bearing_capacity_from_text(text, page_num))

    seen = set()
    unique = []
    for r in results:
        key = (r.value, r.source_pages[0], r.extraction_method, r.source_table_name)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def extract_bearing_capacity_table(
        text: str,
        pdfplumber_tables: List[List[str]],
        page_num: int,
        table_title: str
) -> Optional[ExtractedTable]:
    """Extract a bearing capacity table as structured data."""
    rows = []
    notes = []
    headers = []

    for table in pdfplumber_tables:
        if not table or len(table) < 2:
            continue

        if table[0]:
            headers = [str(c).strip() if c else '' for c in table[0]]

        for row in table[1:]:
            if not row or all(c is None or str(c).strip() == '' for c in row):
                continue

            row_data = {}
            row_text = ' | '.join([str(c) if c else '' for c in row])

            for i, cell in enumerate(row):
                cell_str = str(cell).strip() if cell else ''

                if i == 0:
                    width_match = re.search(r'(\d+(?:\.\d+)?)', cell_str)
                    if width_match:
                        row_data['footing_width'] = cell_str
                        row_data['footing_width_value'] = float(width_match.group(1))

                capacity_match = re.search(r'(\d{1,3}(?:,\d{3})*)', cell_str)
                if capacity_match and i > 0:
                    try:
                        capacity_value = float(capacity_match.group(1).replace(',', ''))
                        if 500 <= capacity_value <= 100000:
                            row_data['bearing_capacity'] = int(capacity_value)
                    except ValueError:
                        pass

            row_data['raw_row'] = row_text
            if row_data.get('footing_width') or row_data.get('bearing_capacity'):
                rows.append(row_data)

        if rows:
            notes_match = re.search(r'Notes?:\s*(.+?)(?=\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
            if notes_match:
                notes = [n.strip() for n in re.split(r'\n\s*\d+\s*', notes_match.group(1)) if n.strip()]

            return ExtractedTable(
                table_name=table_title,
                table_type="bearing_capacity",
                headers=headers,
                rows=rows,
                notes=notes[:3],
                source_page=page_num,
                raw_text=text[:300]
            )

    return None


def extract_settlement_table(
        text: str,
        pdfplumber_tables: List[List[str]],
        page_num: int,
        table_title: str
) -> Optional[ExtractedTable]:
    """Extract a settlement table as structured data."""
    rows = []
    notes = []
    headers = []

    for table in pdfplumber_tables:
        if not table or len(table) < 2:
            continue

        first_rows_text = ' '.join([str(c) if c else '' for row in table[:3] for c in row]).lower()

        if any(kw in first_rows_text for kw in ['settlement', 'pressure', 'width', 'deformation']):
            if table[0]:
                headers = [str(c).strip() if c else '' for c in table[0]]

            for row in table[1:]:
                if not row or all(c is None or str(c).strip() == '' for c in row):
                    continue

                row_data = {'cells': []}
                row_text = ' | '.join([str(c) if c else '' for c in row])

                for i, cell in enumerate(row):
                    cell_str = str(cell).strip() if cell else ''
                    row_data['cells'].append(cell_str)

                    if i == 0:
                        width_match = re.search(r'(\d+(?:\.\d+)?)', cell_str)
                        if width_match:
                            row_data['width'] = cell_str
                            row_data['width_value'] = float(width_match.group(1))

                row_data['raw_row'] = row_text
                if row_data.get('width') or any(c for c in row_data['cells'] if c):
                    rows.append(row_data)

            if rows:
                notes_match = re.search(r'Notes?:\s*(.+?)(?=\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
                if notes_match:
                    notes = [n.strip() for n in re.split(r'\n\s*\d+\s*', notes_match.group(1)) if n.strip()]

                return ExtractedTable(
                    table_name=table_title,
                    table_type="settlement",
                    headers=headers,
                    rows=rows,
                    notes=notes[:3],
                    source_page=page_num,
                    raw_text=text[:300]
                )

    return None


@st.cache_data(show_spinner=False)
def extract_design_tables(
        pdf_bytes: bytes,
        pages_text: Dict[int, str],
        tables_by_page: Dict[int, List[List[str]]]
) -> Dict[str, List[ExtractedTable]]:
    """Extract all high-value design tables."""
    extracted = {'bearing_capacity': [], 'settlement': []}
    seen_titles = set()

    for page_num, text in pages_text.items():
        table_titles = detect_table_title(text)
        pdfplumber_tables = tables_by_page.get(page_num, [])

        for title, table_type, position in table_titles:
            table_num_match = re.search(r'TABLE\s*(\d+)', title, re.IGNORECASE)
            if table_num_match:
                title_key = f"TABLE_{table_num_match.group(1)}"
            else:
                title_key = re.sub(r'\s+', ' ', title.upper().strip())

            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            section_text = text[position:position + 2000]

            if table_type == "bearing_capacity":
                table = extract_bearing_capacity_table(section_text, pdfplumber_tables, page_num, title)
                if table:
                    extracted['bearing_capacity'].append(table)
            elif table_type == "settlement":
                table = extract_settlement_table(section_text, pdfplumber_tables, page_num, title)
                if table:
                    extracted['settlement'].append(table)

    return extracted


# =============================================================================
# GROUNDWATER EXTRACTION
# =============================================================================

@st.cache_data(show_spinner=False)
def extract_groundwater_depth(_pages_text: Dict[int, str]) -> List[ExtractedValue]:
    """Extract groundwater depth with keyword gating."""
    results = []

    gw_keywords = ['groundwater', 'water table', 'water level', 'ground water',
                   'encountered water', 'observed water', 'seepage', 'static water',
                   'water depth', 'free water', 'water was']

    for page_num, text in _pages_text.items():
        text_lower = text.lower()

        # Skip pages without groundwater keywords
        if not any(kw in text_lower for kw in gw_keywords):
            continue

        sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if not any(kw in sentence_lower for kw in gw_keywords):
                continue

            if 'not encountered' in sentence_lower or 'not observed' in sentence_lower:
                results.append(ExtractedValue(
                    parameter="groundwater_depth",
                    value=None,
                    value_text="Not encountered",
                    units_normalized="Not encountered",
                    source_pages=[page_num],
                    confidence="high",
                    extraction_method="text",
                    source_row_text=sentence.strip()[:200]
                ))
                continue

            depth_patterns = [
                r'(?:at\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*(?:feet|ft)\s*(?:below|bgs|depth)',
                r'(?:depth|level).*?(\d+(?:\.\d+)?)\s*(?:feet|ft)',
                r'(\d+(?:\.\d+)?)\s*(?:feet|ft).*?(?:below|depth)'
            ]

            for pattern in depth_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        if value > 200:
                            continue

                        location_match = re.search(r'\b(B-\d+|BH-\d+|TP-\d+)\b', sentence, re.IGNORECASE)
                        location = location_match.group(0).upper() if location_match else None

                        validation = validate_value("groundwater_depth", value)

                        results.append(ExtractedValue(
                            parameter="groundwater_depth",
                            value=value,
                            units="ft",
                            units_normalized=f"{value} ft bgs",
                            location=location,
                            source_pages=[page_num],
                            confidence="high",
                            extraction_method="text",
                            source_row_text=sentence.strip()[:200],
                            validation_status=validation.status,
                            validation_message=validation.message
                        ))
                        break
                    except ValueError:
                        continue

    seen = set()
    unique = []
    for r in results:
        key = (r.value, r.location)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# =============================================================================
# BORING INVENTORY
# =============================================================================

@st.cache_data(show_spinner=False)
def extract_boring_inventory(
        pdf_bytes: bytes,
        pages_text: Dict[int, str]
) -> Tuple[List[BoringInfo], List[BoringLogPage]]:
    """Extract boring inventory and identify graphical log pages."""
    borings = {}
    log_pages = []

    full_text = "\n".join(pages_text.values())
    boring_patterns = [r'\b(B-\d+[A-Za-z]?)\b', r'\b(WA-\d+)\b', r'\b(WB-\d+)\b', r'\b(BH-\d+[A-Za-z]?)\b', r'\b(TP-\d+[A-Za-z]?)\b']

    all_boring_ids = set()
    for pattern in boring_patterns:
        for match in re.findall(pattern, full_text, re.IGNORECASE):
            all_boring_ids.add(match.upper())

    for boring_id in all_boring_ids:
        borings[boring_id] = BoringInfo(boring_id=boring_id)

    log_keywords = ['log of boring', 'boring log', 'field data', 'material description',
                    'graphic log', 'blows/foot', 'sample name', 'recovered']

    for page_num, text in pages_text.items():
        text_lower = text.lower()
        has_log_keywords = any(kw in text_lower for kw in log_keywords)

        current_boring_id = None
        boring_header = re.search(r'(?:Log\s+of\s+)?Boring\s+(B-\d+|BH-\d+)', text, re.IGNORECASE)
        if boring_header:
            current_boring_id = boring_header.group(1).upper()

        if has_log_keywords:
            has_fragmented = len(re.findall(r'\b\d{1,3}\b', text)) > 20
            has_headers = bool(re.search(r'depth.*(?:feet|ft)|blows.*foot', text_lower))

            if has_headers and has_fragmented:
                log_pages.append(BoringLogPage(
                    page_num=page_num,
                    boring_id=current_boring_id,
                    text_length=len(text.strip()),
                    has_log_keywords=has_log_keywords,
                    needs_vision=True,
                    reason="Graphical boring log format detected"
                ))

        total_depth_match = re.search(r'Total\s*Depth\s*\(ft\)\s*[:\s]*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if total_depth_match and current_boring_id and current_boring_id in borings:
            try:
                depth = float(total_depth_match.group(1))
                if borings[current_boring_id].depth is None:
                    borings[current_boring_id].depth = depth
                    if page_num not in borings[current_boring_id].source_pages:
                        borings[current_boring_id].source_pages.append(page_num)
            except ValueError:
                pass

        if current_boring_id and current_boring_id in borings and borings[current_boring_id].depth is None:
            notes_depth_match = re.search(r'Notes:\s*\n(\d+(?:\.\d+)?)\s*\n[A-Z]{2,4}\n', text)
            if notes_depth_match:
                try:
                    depth = float(notes_depth_match.group(1))
                    if 10 <= depth <= 500:
                        borings[current_boring_id].depth = depth
                        if page_num not in borings[current_boring_id].source_pages:
                            borings[current_boring_id].source_pages.append(page_num)
                except ValueError:
                    pass

        if current_boring_id and current_boring_id in borings and borings[current_boring_id].depth is None:
            if 'Total\nDepth (ft)' in text or 'Total\nDepth' in text:
                borings[current_boring_id].needs_vision_review = True

        gw_pattern = r'[Gg]roundwater\s+(?:was\s+)?(?:encountered|observed)\s+(?:at\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*(?:feet|ft)'
        gw_match = re.search(gw_pattern, text)
        if gw_match and current_boring_id and current_boring_id in borings:
            try:
                gw_depth = float(gw_match.group(1))
                borings[current_boring_id].groundwater_depth = gw_depth
                borings[current_boring_id].groundwater_note = gw_match.group(0)
            except ValueError:
                pass

        if 'groundwater not observed' in text_lower:
            if current_boring_id and current_boring_id in borings:
                borings[current_boring_id].groundwater_note = "Not observed"

    def sort_key(b):
        num_match = re.search(r'\d+', b.boring_id)
        return (b.boring_id.split('-')[0], int(num_match.group()) if num_match else 0)

    return sorted(borings.values(), key=sort_key), log_pages


# =============================================================================
# OTHER PARAMETER EXTRACTION
# =============================================================================

@st.cache_data(show_spinner=False)
def extract_friction_angle(_pages_text: Dict[int, str]) -> List[ExtractedValue]:
    """Extract friction angle from text."""
    results = []
    keywords = ['friction angle', 'phi', 'internal friction', 'φ']

    for page_num, text in _pages_text.items():
        if not any(kw in text.lower() for kw in keywords):
            continue

        patterns = [
            r'(?:friction\s+angle|phi|φ)[:\s=]*(\d+(?:\.\d+)?)\s*(?:degrees|°)?',
            r'(\d+(?:\.\d+)?)\s*(?:degrees|°)\s*(?:friction|phi|φ)'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    if 10 <= value <= 55:
                        validation = validate_value("friction_angle", value)
                        results.append(ExtractedValue(
                            parameter="friction_angle",
                            value=value,
                            units="degrees",
                            units_normalized=f"{value:.1f}°",
                            source_pages=[page_num],
                            confidence="high",
                            extraction_method="text",
                            source_row_text=match.group(0),
                            validation_status=validation.status,
                            validation_message=validation.message
                        ))
                except ValueError:
                    continue

    seen = set()
    unique = []
    for r in results:
        if r.value not in seen:
            seen.add(r.value)
            unique.append(r)

    return unique


@st.cache_data(show_spinner=False)
def extract_unit_weight(_pages_text: Dict[int, str]) -> List[ExtractedValue]:
    """Extract unit weight from text."""
    results = []

    for page_num, text in _pages_text.items():
        # Skip pages without unit weight keywords
        if 'pcf' not in text.lower() and 'unit weight' not in text.lower():
            continue
        patterns = [
            r'(?:unit\s+weight|density)[:\s=]*(\d+(?:\.\d+)?)\s*pcf',
            r'(\d+(?:\.\d+)?)\s*pcf'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    if 80 <= value <= 170:
                        validation = validate_value("unit_weight_total", value)
                        results.append(ExtractedValue(
                            parameter="unit_weight_total",
                            value=value,
                            units="pcf",
                            units_normalized=f"{value:.1f} pcf",
                            source_pages=[page_num],
                            confidence="medium",
                            extraction_method="text",
                            source_row_text=match.group(0),
                            validation_status=validation.status,
                            validation_message=validation.message
                        ))
                except ValueError:
                    continue

    seen = set()
    unique = []
    for r in results:
        if r.value not in seen:
            seen.add(r.value)
            unique.append(r)

    return unique


# =============================================================================
# QC AND EXPORT
# =============================================================================

TRACKED_PARAMETERS = [
    ("allowable_bearing_capacity", "Allowable Bearing Capacity"),
    ("friction_angle", "Friction Angle"),
    ("unit_weight_total", "Unit Weight"),
    ("groundwater_depth", "Groundwater Depth"),
]


def generate_not_found_report(results: List[ExtractedValue]) -> List[Dict[str, Any]]:
    """Generate found/not-found report based on actual extraction results."""
    report = []
    by_param = defaultdict(list)
    for r in results:
        by_param[r.parameter].append(r)

    for param_key, param_display in TRACKED_PARAMETERS:
        param_results = by_param.get(param_key, [])
        found = [r for r in param_results if r.value is not None or r.value_text is not None]

        if found:
            report.append({
                "parameter": param_display,
                "status": "found",
                "count": len(found),
                "reason": None
            })
        else:
            report.append({
                "parameter": param_display,
                "status": "not_found",
                "count": 0,
                "reason": f"No {param_display.lower()} found in document"
            })

    return report


def export_to_csv(
        results: List[ExtractedValue],
        metadata: ReportMetadata,
        borings: List[BoringInfo],
        design_tables: Dict[str, List[ExtractedTable]],
        filename: str
) -> str:
    """Export to CSV with full tables."""
    csv_data = []

    csv_data.append({'Category': 'Metadata', 'Parameter': 'Project Name', 'Value': metadata.project_name or 'Not found',
                     'Table Name': '', 'Footing Width': '', 'Case Type': '', 'Conditions': '', 'Source Page': '', 'Source': 'text'})
    csv_data.append(
        {'Category': 'Metadata', 'Parameter': 'Reference Number', 'Value': metadata.reference_number or 'Not found',
         'Table Name': '', 'Footing Width': '', 'Case Type': '', 'Conditions': '', 'Source Page': '', 'Source': 'footer'})

    for table in design_tables.get('bearing_capacity', []):
        table_conditions = '; '.join(table.notes[:2]) if table.notes else ''
        for row in table.rows:
            csv_data.append({
                'Category': 'Bearing Capacity Table',
                'Parameter': 'Allowable Bearing',
                'Value': row.get('bearing_capacity', ''),
                'Table Name': table.table_name,
                'Footing Width': row.get('footing_width', ''),
                'Case Type': row.get('case_type', ''),
                'Conditions': table_conditions,
                'Source Page': f'p.{table.source_page}',
                'Source': 'table'
            })

    for result in results:
        if result.value is not None:
            csv_data.append({
                'Category': 'Parameter',
                'Parameter': result.parameter.replace('_', ' ').title(),
                'Value': result.value,
                'Table Name': result.source_table_name or '',
                'Footing Width': '',
                'Case Type': result.case_type or '',
                'Conditions': result.conditions_text or '',
                'Source Page': ','.join([f'p.{p}' for p in result.source_pages]),
                'Source': result.extraction_method
            })

    for boring in borings:
        csv_data.append({
            'Category': 'Boring',
            'Parameter': 'Boring Inventory',
            'Value': boring.boring_id,
            'Table Name': '',
            'Footing Width': '',
            'Case Type': '',
            'Conditions': '',
            'Source Page': ','.join([f'p.{p}' for p in boring.source_pages]),
            'Source': f"Depth: {boring.depth or '—'} ft, GW: {boring.groundwater_note or '—'}"
        })

    return pd.DataFrame(csv_data).to_csv(index=False)


def create_pdf_report(
        results: List[ExtractedValue],
        metadata: ReportMetadata,
        borings: List[BoringInfo],
        design_tables: Dict[str, List[ExtractedTable]],
        not_found_report: List[Dict],
        log_pages: List[BoringLogPage],
        pdf_filename: str
) -> bytes:
    """Create PDF report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18,
                                 textColor=colors.HexColor('#243A47'), spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14,
                                   textColor=colors.HexColor('#2c3e50'), spaceAfter=10)

    story.append(Paragraph("Geotechnical Report Extraction Summary", title_style))
    story.append(Paragraph(f"Source: {pdf_filename}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Report Information", heading_style))
    meta_data = [['Field', 'Value'],
                 ['Project Name', metadata.project_name or 'Not found'],
                 ['File/Reference No.', metadata.reference_number or 'Not found'],
                 ['Location', metadata.project_location or 'Not found'],
                 ['Report Date', metadata.report_date or 'Not found'],
                 ['Prepared For', metadata.prepared_for or 'Not found']]

    meta_table = Table(meta_data, colWidths=[1.5 * inch, 4.5 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#243A47')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.2 * inch))

    if design_tables.get('bearing_capacity'):
        story.append(Paragraph("Bearing Capacity (from Tables)", heading_style))

        for table in design_tables['bearing_capacity']:
            story.append(Paragraph(f"<b>{table.table_name}</b> (Page {table.source_page})", styles['Normal']))

            bearing_data = [['Footing Width (ft)', 'Allowable Bearing (psf)']]
            for row in table.rows:
                bearing_data.append([
                    row.get('footing_width', '—'),
                    f"{row.get('bearing_capacity', '—'):,}" if isinstance(row.get('bearing_capacity'), int) else '—'
                ])

            bearing_table = Table(bearing_data, colWidths=[3 * inch, 3 * inch])
            bearing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#51776C')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ALIGN', (1, 0), (1, -1), 'LEFT')
            ]))
            story.append(bearing_table)

            if table.notes:
                story.append(Spacer(1, 0.1 * inch))
                combined_notes = ' '.join(table.notes[:2])
                story.append(Paragraph(f"<i>Conditions: {combined_notes[:300]}</i>", styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

    if design_tables.get('settlement'):
        story.append(Paragraph("Settlement (from Tables)", heading_style))

        for table in design_tables['settlement']:
            story.append(Paragraph(f"<b>{table.table_name}</b> (Page {table.source_page})", styles['Normal']))

            if table.headers:
                settlement_data = [table.headers]
            else:
                settlement_data = [['Width/Diameter', 'Settlement Data']]

            for row in table.rows:
                if row.get('cells'):
                    settlement_data.append(row['cells'])
                else:
                    settlement_data.append([row.get('width', '—'), row.get('raw_row', '—')])

            num_cols = len(settlement_data[0]) if settlement_data else 2
            col_width = 6.0 / num_cols * inch

            settlement_table = Table(settlement_data, colWidths=[col_width] * num_cols)
            settlement_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3E5C5C')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ALIGN', (1, 0), (-1, -1), 'LEFT')
            ]))
            story.append(settlement_table)

            if table.notes:
                for note in table.notes[:2]:
                    story.append(Paragraph(f"<i>Note: {note[:100]}</i>", styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

    if borings:
        story.append(Paragraph("Exploration Program", heading_style))
        boring_data = [['ID', 'Depth', 'Groundwater']]
        for b in borings[:15]:
            boring_data.append([
                b.boring_id,
                f"{b.depth} ft" if b.depth else "—",
                f"{b.groundwater_depth} ft" if b.groundwater_depth else (b.groundwater_note or "—")
            ])

        boring_table = Table(boring_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
        boring_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A6670')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(boring_table)
        story.append(Spacer(1, 0.2 * inch))

    vision_pages = [lp for lp in log_pages if lp.needs_vision]
    if vision_pages:
        story.append(Paragraph("Pages Requiring Manual Review", heading_style))
        page_list = ', '.join([f"p.{lp.page_num}" for lp in vision_pages])
        story.append(Paragraph(f"Graphical boring logs detected on: {page_list}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Extraction Status", heading_style))
    status_data = [['Parameter', 'Status']]
    for item in not_found_report:
        status = f"Found ({item['count']})" if item['status'] == 'found' else "Not Found"
        status_data.append([item['parameter'], status])

    status_table = Table(status_data, colWidths=[3 * inch, 3 * inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(status_table)

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("<i>This summary does not replace professional review.</i>",
                           ParagraphStyle('Footer', fontSize=8, textColor=colors.grey)))

    doc.build(story)
    return buffer.getvalue()


# =============================================================================
# Custom UI
# =============================================================================
def render_custom_header():
    """Render title and subtitle."""
    st.markdown("""
    <h1 style="color: #ffffff; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.25rem;">
        Geotechnical Report Summary Extractor
    </h1>
    <p style="color: #c0d0c8; font-size: 1.1rem; margin-bottom: 1.5rem;">
        Extract key design parameters from text-based geotechnical reports in seconds.
    </p>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Geotech Extractor", layout="wide")

    load_custom_css()
    render_custom_header()

    # Sidebar
    with st.sidebar:
        use_tables = st.checkbox("Prioritize table extraction", value=True)
        st.divider()
        st.markdown("""

            <p style="color: #ffffff; font-weight: 600;">Summary includes:</p>
                <div style="
                    border-left: 3px solid #87CEEB; 
                    padding-left: 1rem; 
                    margin: 1rem 0;
                    color: #b6dbd0; 
                    font-size: 0.9rem; 
                    line-height: 2;">
            <p style="margin: 0 0 1rem 0; line-height: 1;">Bearing capacity tables</p>
            <p style="margin: 0 0 1rem 0; line-height: 1;">Settlement tables</p>
            <p style="margin: 0 0 1rem 0; line-height: 1;">Boring inventory</p>
            <p style="margin: 0 0 1rem 0; line-height: 1;">Groundwater depths</p>
            <p style="margin: 0; line-height: 1;">Project metadata</p>
            """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="visible")

    if uploaded_file:
        try:
            pdf_bytes = uploaded_file.getvalue()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            st.stop()

        st.success(f"PDF loaded: {page_count} pages")

        # Extract
        with st.spinner("Extracting..."):
            pages_text = extract_text_by_page(pdf_bytes)

            total_chars = sum(len(text.strip()) for text in pages_text.values())
            avg_chars_per_page = total_chars / len(pages_text) if pages_text else 0

            if avg_chars_per_page < 100:
                st.error("⚠️ **Scanned PDF Detected**")
                st.warning(
                    "This PDF appears to be image-based (no readable text). Please upload a text-based PDF."
                )
                st.stop()

            tables_by_page = extract_tables_from_pdf(pdf_bytes, pages_text) if use_tables else {}
            design_tables = extract_design_tables(pdf_bytes, pages_text, tables_by_page)
            metadata = extract_metadata(pdf_bytes, pages_text)
            borings, log_pages = extract_boring_inventory(pdf_bytes, pages_text)

            bearing_results = extract_bearing_capacity(pages_text, tables_by_page, design_tables)
            friction_results = extract_friction_angle(pages_text)
            unit_weight_results = extract_unit_weight(pages_text)
            gw_results = extract_groundwater_depth(pages_text)

            all_results = bearing_results + friction_results + unit_weight_results + gw_results

        not_found_report = generate_not_found_report(all_results)

        # Display tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Metadata", "📊 Bearing Tables", "🔩 Borings", "⚠️ QC"])

        with tab1:
            st.subheader("Report Information")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Project:** {metadata.project_name or '❌ Not found'}")
                st.markdown(f"**Location:** {metadata.project_location or '❌ Not found'}")
                st.markdown(f"**Date:** {metadata.report_date or '❌ Not found'}")
            with col2:
                st.markdown(f"**File No:** {metadata.reference_number or '❌ Not found'}")
                st.markdown(f"**Prepared For:** {metadata.prepared_for or '❌ Not found'}")

        with tab2:
            narrative_bearing = [r for r in bearing_results if r.extraction_method == "text"]
            has_tables = bool(design_tables.get('bearing_capacity'))

            if narrative_bearing and not has_tables:
                st.subheader("Bearing Capacity (from Narrative Text)")
                for r in narrative_bearing:
                    # Build display with case_type context
                    display_parts = [f"**{r.units_normalized}**"]
                    if r.case_type and r.case_type != "general":
                        display_parts.append(f"— {r.case_type}")
                    display_parts.append(f"*(p.{r.source_pages[0] if r.source_pages else '?'})*")
                    st.markdown(" ".join(display_parts))

                    # Show conditions if present
                    if r.conditions_text:
                        st.caption(f"Conditions: {r.conditions_text}")
                    if r.source_row_text:
                        st.caption(f'Source: "{r.source_row_text.strip()[:150]}..."')
                st.divider()

            st.subheader("Bearing Capacity Tables")
            if has_tables:
                for table in design_tables['bearing_capacity']:
                    st.markdown(f"### {table.table_name}")
                    st.caption(f"Page {table.source_page}")

                    df_data = [{'Footing Width (ft)': row.get('footing_width', '—'),
                                'Allowable Bearing (psf)': f"{row.get('bearing_capacity', '—'):,}"
                                if isinstance(row.get('bearing_capacity'), int) else '—'}
                               for row in table.rows]
                    st.dataframe(pd.DataFrame(df_data), hide_index=True)

                    if table.notes:
                        with st.expander("Table Notes"):
                            for note in table.notes:
                                st.caption(f"• {note}")
            else:
                st.info("No bearing capacity tables found")

            if narrative_bearing and has_tables:
                st.divider()
                st.subheader("Additional Bearing Capacity (from Text)")
                for r in narrative_bearing:
                    # Build display with case_type context
                    display_parts = [f"**{r.units_normalized}**"]
                    if r.case_type and r.case_type != "general":
                        display_parts.append(f"— {r.case_type}")
                    display_parts.append(f"*(p.{r.source_pages[0] if r.source_pages else '?'})*")
                    st.markdown(" ".join(display_parts))

                    # Show conditions if present
                    if r.conditions_text:
                        st.caption(f"Conditions: {r.conditions_text}")
                    if r.source_row_text:
                        st.caption(f'Source: "{r.source_row_text.strip()[:150]}..."')

            st.divider()
            st.subheader("Groundwater")
            gw_found = [r for r in gw_results if r.value is not None or r.value_text is not None]
            if gw_found:
                for r in gw_found:
                    st.markdown(f"**{r.units_normalized}** (p.{r.source_pages[0] if r.source_pages else '?'})")
                    if r.source_row_text:
                        st.caption(f'Source: "{r.source_row_text[:100]}..."')
            else:
                st.info("No groundwater depth found")

        with tab3:
            st.subheader("Exploration Program")
            if borings:
                boring_df = pd.DataFrame([{
                    'ID': b.boring_id,
                    'Depth': f"{b.depth} ft" if b.depth else "—",
                    'Groundwater': f"{b.groundwater_depth} ft" if b.groundwater_depth else (b.groundwater_note or "—")
                } for b in borings])
                st.dataframe(boring_df, hide_index=True)
            else:
                st.warning("No borings found")

            vision_pages = [lp for lp in log_pages if lp.needs_vision]
            if vision_pages:
                st.divider()
                st.warning(f"⚠️ {len(vision_pages)} pages detected as graphical boring logs")
                st.caption("Manual review recommended for detailed layer data, SPT N-values.")
                st.markdown(f"Pages: {', '.join([f'p.{lp.page_num}' for lp in vision_pages])}")

        with tab4:
            st.subheader("Extraction QC")

            col1, col2, col3 = st.columns(3)
            col1.metric("Parameters Found", sum(1 for r in not_found_report if r['status'] == 'found'))
            col2.metric("Tables Extracted", sum(len(v) for v in design_tables.values()))
            col3.metric("Borings", len(borings))

            st.divider()
            for item in not_found_report:
                if item['status'] == 'found':
                    st.markdown(f"✅ **{item['parameter']}** — {item['count']} found")
                else:
                    st.markdown(f"❌ **{item['parameter']}** — {item['reason'] or 'Not found'}")

        # Export
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate CSV"):
                csv_data = export_to_csv(all_results, metadata, borings, design_tables, uploaded_file.name)
                st.download_button("📄 Download CSV", csv_data,
                                   f"{uploaded_file.name.replace('.pdf', '')}_summary.csv", "text/csv")

        with col2:
            if st.button("Generate PDF"):
                pdf_data = create_pdf_report(all_results, metadata, borings, design_tables,
                                             not_found_report, log_pages, uploaded_file.name)
                st.download_button("📄 Download PDF", pdf_data,
                                   f"{uploaded_file.name.replace('.pdf', '')}_summary.pdf", "application/pdf")

        # Bottom bar with checklist and footer
    st.markdown("---")
    st.markdown("""
     <div style="background-color: rgba(39, 63, 77, .6); border-radius: 10px; padding: 1rem 1.5rem .5rem;">
         <p class="footer" style="background: #3E5C5C; padding: 30px 20px 20px; border-radius: 4px; margin: -16px -35px 14px;">⚠️  <strong>Disclaimer:</strong> Please review extracted results carefully. Summaries must be verified against the original report by a licensed professional engineer.</p>
         <p class="footer" style="display: flex; justify-content: space-between;"><a href="https://forms.gle/PX4ZQJFn6nc58vdV9" target="_blank">I'd love your suggestions. Please send feedback.</a>
            <span> Created by <a href="https://alexengineered.com" target="_blank" >AlexEngineered</a></span>
         </p>
     </div>
     """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()