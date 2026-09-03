"""The take-away PDF: one analysis, written down.

This is the artifact most likely to outlive the screen that produced it — forwarded, printed,
read months later by someone who never saw the app. So it carries its caveats on the page
rather than assuming context:

* the disclaimer appears at the top *and* the bottom, not once in small print;
* the decision is stated at the model's own threshold, with the threshold shown, so the
  probability cannot be re-read against an assumed 50%;
* every imputed field is listed by name, so a reader can see what the estimate did not know;
* the model's external-validation position is stated in full;
* the source document, when there was one, is named as synthetic.

Nothing here is computed. Every number is taken from the prediction response the API already
returned, so this document cannot disagree with the screen it came from.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from config import DISCLAIMER
from ml.reports.demo_report import Page

RESEARCH_BANNER = "RESEARCH & EDUCATION OUTPUT - NOT A MEDICAL DIAGNOSIS"


def _factor_sentence(contribution: float) -> str:
    return "pushed the estimate higher" if contribution >= 0 else "pushed the estimate lower"


def build_result_report(
    prediction: dict,
    out_path: Path,
    *,
    schema: dict | None = None,
    source_document: str | None = None,
    top_factors: int = 8,
) -> Path:
    """Render one analysis to PDF from a ``/predict`` response. Computes nothing."""
    labels: dict[str, str] = {}
    units: dict[str, str] = {}
    if schema:
        for f in schema.get("features", []):
            labels[f["name"]] = f.get("label", f["name"])
            units[f["name"]] = f.get("unit", "")

    def label_of(name: str) -> str:
        return labels.get(name, name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle(f"{prediction['module']} — analysis")
    c.setSubject(RESEARCH_BANNER)
    c.setAuthor("Multi-Disease AI (research/education platform)")

    p = Page(c)
    p.title(prediction["module"])
    p.banner(RESEARCH_BANNER)
    p.paragraph(DISCLAIMER, size=7.5)
    p.y -= 2 * mm
    p.rule()
    p.kv_line(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Model: {prediction['model_name']} ({prediction['calibration']}) "
        f"v{prediction['model_version']}",
    )
    if source_document:
        p.kv_line(f"Source document: {source_document}", "")
    p.rule()

    # --- the result -----------------------------------------------------------------
    p.section("Analysis")
    flagged = prediction["flagged"]
    headline = (
        "This model would flag this case for follow-up."
        if flagged
        else "This model would not flag this case."
    )
    c.setFont("Helvetica-Bold", 12)
    p._space(8 * mm)
    c.drawString(22 * mm, p.y, headline)
    p.y -= 7 * mm

    p.row("Estimated probability", f"{prediction['probability'] * 100:.1f}", "%")
    p.row("Decision threshold used", f"{prediction['threshold'] * 100:.1f}", "%")
    p.row("Category", str(prediction["risk_band"]["label"]).title(), "")
    p.y -= 1 * mm
    p.paragraph(
        "The category above is a presentation label whose upper boundary is this model's own "
        "operating threshold; it is not a clinical category. The threshold was chosen on "
        "training data, never on the held-out test set. A probability below it does not mean "
        "'healthy', and one above it does not mean 'diagnosed'.",
        size=7.5,
    )

    # --- what influenced it ---------------------------------------------------------
    explanation = prediction.get("explanation")
    if explanation and explanation.get("contributions"):
        p.section("What influenced this estimate")
        p.paragraph(
            "Ranked by how much each value moved this particular estimate, computed with "
            "SHAP on the model itself. These describe the model's reasoning, not a cause of "
            "disease, and must not be read as things to change.",
            size=7.5,
        )
        p.y -= 1 * mm
        for con in explanation["contributions"][:top_factors]:
            name = con["feature"]
            value = con.get("value")
            unit = units.get(name, "")
            shown = "not supplied" if value is None else (
                f"{value:g} {unit}".strip() if unit else f"{value:g}"
            )
            p.row(label_of(name), f"{shown} — {_factor_sentence(con['contribution'])}", "")

    # --- what it did and did not know ------------------------------------------------
    p.section("Information used")
    p.row(
        "Values you supplied",
        f"{prediction['n_features_provided']} of {prediction['n_features_expected']}",
        "",
    )
    imputed = prediction.get("imputed_features") or []
    if imputed:
        p.paragraph(
            "Filled from the training data because they were not supplied: "
            + ", ".join(label_of(n) for n in imputed)
            + ". The estimate did not know these values.",
            size=7.5,
        )
    extrapolated = prediction.get("extrapolated_features") or []
    if extrapolated:
        p.paragraph(
            "Outside the range the model was trained on: "
            + ", ".join(label_of(e["feature"]) for e in extrapolated)
            + ". The model has no support for values there and its output is unreliable.",
            size=7.5,
        )

    # --- caveats ---------------------------------------------------------------------
    warnings = prediction.get("warnings") or []
    if warnings:
        p.section("Caveats that apply to this result")
        for w in warnings:
            p.paragraph(f"- {w}", size=7.5)
            p.y -= 0.5 * mm

    # --- footer ----------------------------------------------------------------------
    p.y -= 2 * mm
    p.rule()
    p.paragraph(
        f"{RESEARCH_BANNER}. {DISCLAIMER} This document was produced by a research and "
        "education platform for demonstrating machine-learning methodology. It describes a "
        "model's output on the information provided, has not been reviewed by any clinician, "
        "and must not be used to make any health decision.",
        size=7,
    )
    c.showPage()
    c.save()
    return out_path
