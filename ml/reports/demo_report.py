"""Synthetic demonstration reports, generated from real held-out test records.

Why generate them at all: a demonstration needs a document to upload, and a real medical
report cannot be used (it would be someone's health data) while an invented one would make
the prediction at the end meaningless — a fabricated input produces a fabricated result.

The compromise here keeps the whole pipeline honest. The *values* come from a real row of
the public UCI dataset, drawn from the **held-out test split**, so the model was never fitted
on them. The *document* around those values is fabricated, and says so on its face. So the
extraction is exercised on a genuine document, and the prediction at the end is a genuine
prediction on a record the model has never seen. Nothing downstream is staged.

The field list, labels, units and coded value names all come from ``models/<disease>/
serving.json`` — the same file the API and the web form are generated from. No feature name
or value is invented here; if it is not in the model's schema, it does not appear on the page.

Layout is deliberately plain: one ``label ....... value unit`` row per line, at fixed
columns. That is what a lab report looks like, and it is also what makes deterministic
extraction reliable — no LLM is needed to read it back.

Run: ``python -m ml.reports.demo_report``
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from config import MODELS_DIR, ROOT_DIR

OUT_DIR = ROOT_DIR / "demo_reports"

NOT_MEASURED = "Not performed"

SYNTHETIC_BANNER = "SYNTHETIC DEMONSTRATION DOCUMENT - NOT A REAL PATIENT RECORD"

# Section headings for the printed document. The model's own group ids drive which sections
# appear and in what order; this only supplies the wording a lab report would actually use.
SECTION_TITLES: dict[str, str] = {
    "about_you": "Patient Information",
    "vitals": "Vitals",
    "urine_test": "Urinalysis",
    "blood_test": "Blood Investigation",
    "symptoms": "Clinical Observations",
    "history": "Medical History",
    "resting_ecg": "Resting Electrocardiogram",
    "exercise_test": "Exercise Stress Test",
    "cardiac_imaging": "Cardiac Imaging",
    "wellbeing": "General Wellbeing",
    "lifestyle": "Lifestyle",
    "healthcare_access": "Healthcare Access",
}

DOCUMENT_TITLES: dict[str, str] = {
    "kidney": "RENAL FUNCTION PANEL",
    "heart": "CARDIOLOGY ASSESSMENT",
    "diabetes": "HEALTH INDICATOR ASSESSMENT",
}

SOURCE_NOTE = (
    "Values below are taken from a single record of the {dataset} held in the "
    "held-out test split, which the model behind this demonstration was never trained on. "
    "The document itself is fabricated. No real person is described."
)

DATASETS: dict[str, str] = {
    "kidney": "UCI Chronic Kidney Disease dataset (id 336)",
    "heart": "UCI Heart Disease dataset, Cleveland processed (id 45)",
    "diabetes": "CDC BRFSS Diabetes Health Indicators dataset (UCI id 891)",
}

# Page geometry. Fixed columns matter: the extractor locates a value by finding its label
# and reading the words to the right of it on the same line.
LEFT = 22 * mm
VALUE_X = 118 * mm
RIGHT = A4[0] - 22 * mm
TOP = A4[1] - 16 * mm
BOTTOM = 16 * mm
LINE = 5.4 * mm


def _fmt_number(v: float) -> str:
    """Shortest exact decimal form. Never scientific notation — that would not round-trip."""
    if float(v).is_integer():
        return str(int(v))
    s = f"{v:.10g}"
    if "e" in s or "E" in s:
        s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def display_value(feature: dict, value: float | None) -> str:
    """What the printed report shows: a coded field's label, else the number.

    A real report says "Poor", not "1", so the document prints the human label and the
    extractor maps it back. That exercises the value-alias path rather than side-stepping it.
    """
    if value is None:
        return NOT_MEASURED
    opts = feature.get("options")
    if opts:
        for o in opts:
            if o["value"] == value:
                return o["label"]
    return _fmt_number(value)


@dataclass
class ReportSpec:
    disease: str
    case_id: str
    recorded_label: int
    path: Path


class Page:
    """A tiny y-cursor over a reportlab canvas, with automatic page breaks."""

    def __init__(self, c: canvas.Canvas):
        self.c = c
        self.y = TOP

    def _space(self, needed: float) -> None:
        if self.y - needed < BOTTOM:
            self.c.showPage()
            self.y = TOP

    def title(self, text: str, size: int = 15) -> None:
        self._space(12 * mm)
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawCentredString(A4[0] / 2, self.y, text)
        self.y -= 7 * mm

    def banner(self, text: str) -> None:
        self._space(10 * mm)
        self.c.setFont("Helvetica-Bold", 9)
        self.c.drawCentredString(A4[0] / 2, self.y, text)
        self.y -= 5 * mm

    def paragraph(self, text: str, size: int = 7.5, leading: float = 4.0) -> None:
        self.c.setFont("Helvetica", size)
        words, line = text.split(), ""
        for w in words:
            trial = f"{line} {w}".strip()
            if self.c.stringWidth(trial, "Helvetica", size) > (RIGHT - LEFT):
                self._space(leading * mm)
                self.c.drawString(LEFT, self.y, line)
                self.y -= leading * mm
                line = w
            else:
                line = trial
        if line:
            self._space(leading * mm)
            self.c.drawString(LEFT, self.y, line)
            self.y -= leading * mm

    def rule(self, gap: float = 3.0) -> None:
        self._space(4 * mm)
        self.c.setLineWidth(0.4)
        self.c.line(LEFT, self.y, RIGHT, self.y)
        self.y -= gap * mm

    def section(self, text: str) -> None:
        self._space(12 * mm)
        self.y -= 1.5 * mm
        self.c.setFont("Helvetica-Bold", 10)
        self.c.drawString(LEFT, self.y, text.upper())
        self.y -= 1.5 * mm
        self.c.setLineWidth(0.6)
        self.c.line(LEFT, self.y, RIGHT, self.y)
        self.y -= 5.0 * mm

    def row(self, label: str, value: str, unit: str) -> None:
        self._space(LINE)
        self.c.setFont("Helvetica", 9.5)
        self.c.drawString(LEFT, self.y, label)
        self.c.setFont("Helvetica-Bold", 9.5)
        # No unit after "Not performed" - nothing was measured, so nothing has units.
        printed = f"{value} {unit}".strip() if unit and value != NOT_MEASURED else value
        self.c.drawString(VALUE_X, self.y, printed)
        self.y -= LINE

    def kv_line(self, left: str, right: str) -> None:
        self._space(LINE)
        self.c.setFont("Helvetica", 8.5)
        self.c.drawString(LEFT, self.y, left)
        self.c.drawRightString(RIGHT, self.y, right)
        self.y -= LINE


def build_report(disease: str, case: dict, out_path: Path) -> Path:
    """Render one synthetic report for ``case`` (a record from ``samples.json``)."""
    serving = json.loads(
        (MODELS_DIR / disease / "serving.json").read_text(encoding="utf-8")
    )
    features = {f["name"]: f for f in serving["features"]}
    groups = serving["groups"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle(f"{DOCUMENT_TITLES.get(disease, disease.title())} — synthetic demonstration")
    c.setAuthor("HealthLens (research/education demonstration)")
    c.setSubject(SYNTHETIC_BANNER)

    p = Page(c)
    p.title(DOCUMENT_TITLES.get(disease, disease.title()))
    p.banner(SYNTHETIC_BANNER)
    p.paragraph(SOURCE_NOTE.format(dataset=DATASETS.get(disease, "source dataset")))
    p.y -= 2 * mm
    p.rule()
    p.kv_line(f"Specimen ID: {case['id'].upper()}", f"Report date: {date.today().isoformat()}")
    p.kv_line("Referring unit: Demonstration", "Laboratory: HealthLens demo set")
    p.rule()

    for g in groups:
        names = [n for n, f in features.items() if f["group"] == g["id"]]
        names.sort(key=lambda n: features[n]["shap_rank"])
        if not names:
            continue
        p.section(SECTION_TITLES.get(g["id"], g["label"]))
        for n in names:
            f = features[n]
            p.row(f["label"], display_value(f, case["features"].get(n)), f.get("unit", ""))

    p.y -= 3 * mm
    p.rule()
    p.paragraph(
        f"{SYNTHETIC_BANNER}. Produced by the HealthLens demonstration platform for "
        "research and educational purposes only. This document does not describe a real "
        "person, is not a clinical record, and must not be used for any medical purpose.",
        size=7,
    )
    c.showPage()
    c.save()
    return out_path


def generate_for_disease(disease: str, out_dir: Path = OUT_DIR) -> list[ReportSpec]:
    """One report per exported sample case."""
    samples = json.loads(
        (MODELS_DIR / disease / "samples.json").read_text(encoding="utf-8")
    )
    out: list[ReportSpec] = []
    for case in samples["cases"]:
        path = out_dir / f"{case['id']}.pdf"
        build_report(disease, case, path)
        out.append(
            ReportSpec(disease, case["id"], int(case["recorded_label"]), path)
        )
    return out


def main() -> None:
    from config import DISEASES

    for disease in DISEASES:
        for spec in generate_for_disease(disease):
            size = spec.path.stat().st_size
            print(
                f"[{disease}] {spec.path.relative_to(ROOT_DIR).as_posix()} "
                f"({size:,} bytes, recorded="
                f"{'positive' if spec.recorded_label else 'negative'})"
            )


if __name__ == "__main__":
    main()
