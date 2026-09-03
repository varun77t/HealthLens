"""Phase 10 tests: document extraction reads the document, and never invents a value.

The central test is :func:`test_extraction_round_trips_every_demo_report`. The extractor is
handed **only the PDF bytes** — it never sees the source row — and the extracted values are
compared against the record the document was generated from. If that passes, extraction
genuinely read the page. Without it, a demo that appears to work could simply be echoing
data it already had, and the whole upload flow would be theatre.

The rest of the file pins the refusals: a missing label stays missing, "Not performed" stays
missing, an unrecognised coded value is flagged rather than coerced, and an out-of-range
number is surfaced rather than clipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from config import MODELS_DIR, ROOT_DIR
from ml.extraction.extractor import extract, pdf_lines
from ml.reports.demo_report import build_report

DISEASES = ("kidney", "heart", "diabetes")
DEMO_DIR = ROOT_DIR / "demo_reports"


def _cases(disease: str) -> list[dict]:
    return json.loads(
        (MODELS_DIR / disease / "samples.json").read_text(encoding="utf-8")
    )["cases"]


def _report_bytes(disease: str, case: dict, tmp_path: Path) -> bytes:
    """Build the report fresh so the test never depends on a stale committed PDF."""
    path = build_report(disease, case, tmp_path / f"{case['id']}.pdf")
    return path.read_bytes()


def _pdf_with_lines(lines: list[str], tmp_path: Path, name: str = "t.pdf") -> bytes:
    p = tmp_path / name
    c = canvas.Canvas(str(p), pagesize=A4)
    y = A4[1] - 60
    for line in lines:
        c.setFont("Helvetica", 10)
        c.drawString(60, y, line)
        y -= 18
    c.showPage()
    c.save()
    return p.read_bytes()


# --- the central test ------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_extraction_round_trips_every_demo_report(disease, tmp_path):
    """Extract from the PDF alone; every value must equal the record it was rendered from.

    The extractor is given bytes and a disease name. It has no access to the source row, so
    agreement can only come from actually reading the document.
    """
    for case in _cases(disease):
        data = _report_bytes(disease, case, tmp_path)
        result = extract(disease, data)
        source = case["features"]
        mismatched = {
            f.name: (source[f.name], f.value)
            for f in result.fields
            if source[f.name] != f.value
        }
        assert not mismatched, f"{case['id']} mis-read: {mismatched}"


@pytest.mark.parametrize("disease", DISEASES)
def test_extraction_covers_every_field_the_model_needs(disease, tmp_path):
    case = _cases(disease)[0]
    result = extract(disease, _report_bytes(disease, case, tmp_path))
    schema = json.loads(
        (MODELS_DIR / disease / "serving.json").read_text(encoding="utf-8")
    )
    assert [f.name for f in result.fields] == schema["feature_order"]
    assert result.n_expected == len(schema["features"])
    assert result.n_found + result.n_needs_review + result.n_missing == result.n_expected


def test_missing_values_in_the_source_row_stay_missing(tmp_path):
    """kidney-positive-1 genuinely has unrecorded labs; they must not be filled in."""
    case = next(c for c in _cases("kidney") if c["id"] == "kidney-positive-1")
    absent = {k for k, v in case["features"].items() if v is None}
    assert absent, "fixture expected to contain unrecorded values"
    result = extract("kidney", _report_bytes("kidney", case, tmp_path))
    for f in result.fields:
        if f.name in absent:
            assert f.value is None and f.status == "missing"


# --- refusals --------------------------------------------------------------------------


def test_a_label_absent_from_the_document_is_missing_not_guessed(tmp_path):
    data = _pdf_with_lines(["Haemoglobin 12.4 g/dL", "Packed cell volume 40 %"], tmp_path)
    result = extract("kidney", data)
    by_name = {f.name: f for f in result.fields}
    assert by_name["hemo"].value == 12.4
    assert by_name["sc"].value is None
    assert by_name["sc"].status == "missing"
    # Nothing anywhere may acquire a value that was not on the page.
    assert sum(f.value is not None for f in result.fields) == 2


def test_not_performed_is_recorded_as_missing(tmp_path):
    data = _pdf_with_lines(["Serum creatinine Not performed", "Sodium N/A"], tmp_path)
    by_name = {f.name: f for f in extract("kidney", data).fields}
    assert by_name["sc"].status == "missing" and by_name["sc"].value is None
    assert by_name["sod"].status == "missing"
    assert "not performed" in by_name["sc"].note.lower()


def test_unrecognised_coded_value_is_flagged_not_coerced(tmp_path):
    data = _pdf_with_lines(["Appetite Ravenous"], tmp_path)
    field = next(f for f in extract("kidney", data).fields if f.name == "appet")
    assert field.value is None
    assert field.status == "needs_review"
    assert "Good" in field.note and "Poor" in field.note


def test_out_of_range_number_is_surfaced_not_clipped(tmp_path):
    """Haemoglobin 250 g/dL is a unit error; the person decides, not the extractor."""
    data = _pdf_with_lines(["Haemoglobin 250 g/dL"], tmp_path)
    field = next(f for f in extract("kidney", data).fields if f.name == "hemo")
    assert field.value == 250.0
    assert field.status == "needs_review"
    assert "outside the range" in field.note


def test_unreadable_value_is_flagged(tmp_path):
    data = _pdf_with_lines(["Serum creatinine pending review"], tmp_path)
    field = next(f for f in extract("kidney", data).fields if f.name == "sc")
    assert field.value is None and field.status == "needs_review"


def test_a_pdf_with_no_text_layer_says_so(tmp_path):
    p = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    c.showPage()
    c.save()
    result = extract("kidney", p.read_bytes())
    assert result.n_found == 0
    assert any("no text layer" in w.lower() for w in result.warnings)
    assert all(f.value is None for f in result.fields)


# --- the blood-pressure trap ---------------------------------------------------------------


def test_blood_pressure_pair_splits_per_model(tmp_path):
    """Heart's trestbps is systolic; kidney's bp is diastolic. Same text, different field.

    Reading "142/90" into both as the same number would be silently, confidently wrong —
    the training medians are 130 and 80 respectively.
    """
    kidney = extract("kidney", _pdf_with_lines(["Blood pressure 142/90"], tmp_path, "k.pdf"))
    bp = next(f for f in kidney.fields if f.name == "bp")
    assert bp.value == 90.0 and bp.status == "found"
    assert "diastolic" in bp.note

    heart = extract("heart", _pdf_with_lines(["Blood pressure 142/90"], tmp_path, "h.pdf"))
    tb = next(f for f in heart.fields if f.name == "trestbps")
    assert tb.value == 142.0 and tb.status == "found"
    assert "systolic" in tb.note


# --- label matching ---------------------------------------------------------------------------


def test_longer_label_wins_over_a_shorter_one_it_contains(tmp_path):
    """"Red blood cell count" and "Red blood cells in urine" are different kidney fields."""
    data = _pdf_with_lines(
        ["Red blood cell count 4.5 millions/cmm", "Red blood cells in urine Normal"],
        tmp_path,
    )
    by_name = {f.name: f for f in extract("kidney", data).fields}
    assert by_name["rbcc"].value == 4.5
    assert by_name["rbc"].value == 0.0


def test_a_section_heading_does_not_swallow_the_field_below_it(tmp_path):
    """Regression: "RESTING ELECTROCARDIOGRAM" is an alias for restecg and carries no value."""
    data = _pdf_with_lines(
        ["RESTING ELECTROCARDIOGRAM", "Resting ECG result Normal"], tmp_path
    )
    field = next(f for f in extract("heart", data).fields if f.name == "restecg")
    assert field.value == 0.0 and field.status == "found"


def test_alias_wordings_from_other_report_styles_resolve(tmp_path):
    data = _pdf_with_lines(
        ["Hemoglobin 11.2 g/dL", "Hematocrit 34 %", "Creatinine 1.4 mg/dL", "BUN 40 mg/dL"],
        tmp_path,
    )
    by_name = {f.name: f for f in extract("kidney", data).fields}
    assert by_name["hemo"].value == 11.2
    assert by_name["pcv"].value == 34.0
    assert by_name["sc"].value == 1.4
    assert by_name["bu"].value == 40.0


# --- the document itself -----------------------------------------------------------------------


@pytest.mark.parametrize("disease", DISEASES)
def test_demo_report_declares_itself_synthetic(disease, tmp_path):
    case = _cases(disease)[0]
    text = " ".join(
        line for _, line in pdf_lines(_report_bytes(disease, case, tmp_path))
    )
    assert "SYNTHETIC DEMONSTRATION DOCUMENT" in text
    assert "NOT A REAL PATIENT RECORD" in text
    assert "held-out test split" in text


@pytest.mark.parametrize("disease", DISEASES)
def test_demo_report_contains_no_feature_outside_the_model_schema(disease, tmp_path):
    """The document may only mention fields the model actually has."""
    case = _cases(disease)[0]
    result = extract(disease, _report_bytes(disease, case, tmp_path))
    schema = json.loads(
        (MODELS_DIR / disease / "serving.json").read_text(encoding="utf-8")
    )
    assert {f.name for f in result.fields} == {f["name"] for f in schema["features"]}


def test_committed_demo_reports_exist_and_still_round_trip():
    """The PDFs shipped in demo_reports/ must match the sample cases they claim to be."""
    for disease in DISEASES:
        for case in _cases(disease):
            path = DEMO_DIR / f"{case['id']}.pdf"
            if not path.exists():
                pytest.skip(f"{path} not generated yet")
            result = extract(disease, path.read_bytes())
            for f in result.fields:
                assert case["features"][f.name] == f.value, f"{case['id']}.{f.name}"


def test_a_number_must_follow_the_label_directly(tmp_path):
    """Regression: "Cholesterol checked in the last 5 years  Yes" yielded chol = 5.

    "Cholesterol" is a legitimate alias for `chol` and prefix-matches that line, so the
    parser went on to harvest the 5 out of "5 years". A value follows its label
    immediately; searching the whole remainder invents readings out of other questions.
    """
    data = _pdf_with_lines(["Cholesterol checked in the last 5 years Yes"], tmp_path)
    field = next(f for f in extract("heart", data).fields if f.name == "chol")
    assert field.value is None
    assert field.status == "needs_review"
    assert "must follow the label" in field.note


def test_a_range_is_flagged_not_halved(tmp_path):
    """Regression: "Age group 75-79" yielded a heart age of 75 marked `found`.

    Taking either end of a band claims a precision the document does not have.
    """
    data = _pdf_with_lines(["Age group 75-79"], tmp_path)
    field = next(f for f in extract("heart", data).fields if f.name == "age")
    assert field.value is None
    assert field.status == "needs_review"


def test_a_bare_range_where_a_number_is_expected_is_refused(tmp_path):
    data = _pdf_with_lines(["Haemoglobin 9.5-10.5 g/dL"], tmp_path)
    field = next(f for f in extract("kidney", data).fields if f.name == "hemo")
    assert field.value is None
    assert field.status == "needs_review"
    assert "range" in field.note


def test_a_label_seen_but_unreadable_is_not_reported_as_absent(tmp_path):
    """"We saw this and couldn't read it" and "it wasn't there" are different statements."""
    seen = next(
        f for f in extract("kidney", _pdf_with_lines(["Appetite Ravenous"], tmp_path)).fields
        if f.name == "appet"
    )
    assert seen.status == "needs_review"
    assert "Good" in seen.note and "Poor" in seen.note

    absent = next(
        f for f in extract("kidney", _pdf_with_lines(["Haemoglobin 12"], tmp_path)).fields
        if f.name == "appet"
    )
    assert absent.status == "missing"
    assert "not found" in absent.note.lower()
