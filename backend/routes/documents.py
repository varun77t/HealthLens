"""Document endpoints: upload for extraction, demo reports, and the take-away PDF.

Three rules govern everything here.

**Uploads are never stored.** The PDF is read into memory, parsed, and dropped when the
request ends. Nothing is written to disk, nothing is logged, and no extracted value appears
in a log line. Health data should not outlive the request that carried it.

**Extraction never predicts.** This module turns a document into candidate values. It does
not score anything. The user reviews those values, and only then does the existing
``POST /predict/{disease}`` run — unchanged from Phase 8.

**Nothing is invented.** A field the extractor cannot find comes back ``missing`` with a null
value, never a plausible-looking default.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.routes.deps import get_bundle, get_registry
from backend.services.registry import Registry
from config import DISCLAIMER, MODELS_DIR, ROOT_DIR
from ml.extraction.extractor import extract
from ml.reports.result_report import build_result_report

router = APIRouter(tags=["documents"])

DEMO_DIR = ROOT_DIR / "demo_reports"

# A generous cap for a lab report. Large enough for a scanned multi-page document, small
# enough that an oversized upload is rejected before it is read into memory.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

PDF_MAGIC = b"%PDF-"


@router.post("/extract/{disease}", summary="Read model fields out of an uploaded report")
async def extract_document(
    disease: str,
    file: UploadFile = File(..., description="A PDF with a text layer."),
    registry: Registry = Depends(get_registry),
):
    """Extract this model's fields from a PDF. Returns candidates for review, not a prediction.

    The response lists **every** field the model needs, each marked `found`, `needs_review`
    or `missing`, with the page it came from. Values are matched against the model's own
    schema, so a coded field can only take a level the model was trained on, and a number
    outside the training range is flagged rather than silently accepted.
    """
    get_bundle(registry, disease)  # 404 on an unknown module before reading any bytes

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(data) / 1e6:.1f} MB; the limit is "
                   f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if not data.startswith(PDF_MAGIC):
        # Sniff the bytes rather than trusting the filename or the client's content type.
        raise HTTPException(
            status_code=415,
            detail="That file is not a PDF. Image and scanned-document extraction is not "
                   "available yet — upload a PDF, or enter the values manually.",
        )

    try:
        result = extract(disease, data)
    except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed into a fake result
        raise HTTPException(
            status_code=422,
            detail=f"Could not read this PDF: {type(exc).__name__}. It may be encrypted or "
                   f"damaged. You can enter the values manually instead.",
        ) from exc
    finally:
        del data  # not persisted anywhere; drop the bytes as soon as parsing is done

    return {**result.to_dict(), "disclaimer": DISCLAIMER}


@router.get("/demo-reports/{disease}", summary="List the synthetic demonstration reports")
def list_demo_reports(disease: str, registry: Registry = Depends(get_registry)):
    """Synthetic reports built from held-out test records, for demonstrating the flow."""
    get_bundle(registry, disease)
    path = MODELS_DIR / disease / "samples.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No sample cases for '{disease}'.")
    samples = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        {
            "id": c["id"],
            "recorded_label": c["recorded_label"],
            "recorded_label_meaning": c["recorded_label_meaning"],
            "filename": f"{c['id']}.pdf",
            "available": (DEMO_DIR / f"{c['id']}.pdf").exists(),
        }
        for c in samples["cases"]
    ]
    return {
        "disease": disease,
        "cases": cases,
        "note": (
            "These documents are fabricated, and say so on their face. The values inside "
            "them come from real records in the held-out test split, which the model was "
            "never trained on — so extraction runs on a genuine document and the prediction "
            "at the end is a genuine prediction on an unseen record."
        ),
        "disclaimer": DISCLAIMER,
    }


@router.get("/demo-reports/{disease}/{case_id}", summary="Download one demonstration report")
def get_demo_report(disease: str, case_id: str, registry: Registry = Depends(get_registry)):
    get_bundle(registry, disease)
    # Matched against the files actually present rather than joined onto a path, so a
    # crafted case_id cannot escape the directory.
    available = {p.stem: p for p in DEMO_DIR.glob(f"{disease}-*.pdf")}
    if case_id not in available:
        raise HTTPException(
            status_code=404,
            detail=f"No demonstration report '{case_id}'. Available: "
                   f"{', '.join(sorted(available)) or 'none — run '
                   'python -m ml.reports.demo_report'}.",
        )
    return FileResponse(
        available[case_id], media_type="application/pdf",
        filename=f"{case_id}.pdf",
    )


@router.post("/report/{disease}", summary="Generate the take-away PDF for one analysis")
def build_report(
    disease: str,
    prediction: dict,
    registry: Registry = Depends(get_registry),
):
    """Render a ``/predict`` response as a PDF.

    Computes nothing: every figure is taken from the response body supplied, so the document
    cannot disagree with the screen that produced it. The disclaimer, the threshold used and
    the list of imputed fields are all written into the page, because this file will be read
    away from the app that made it.
    """
    bundle = get_bundle(registry, disease)
    required = {"module", "probability", "flagged", "threshold", "risk_band", "model_name"}
    missing = required - set(prediction)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Not a prediction response — missing {', '.join(sorted(missing))}.",
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = build_result_report(
            prediction,
            Path(tmp) / f"{disease}-analysis.pdf",
            schema=bundle.serving,
            source_document=prediction.get("source_document"),
        )
        payload = path.read_bytes()

    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{disease}-analysis.pdf"'
        },
    )
