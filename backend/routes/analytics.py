"""Analytics — the saved evaluation artifacts, served as JSON.

Nothing here is computed at request time. Every number was written by the training run into
``reports/<disease>/`` and is read back verbatim, so the API cannot report a metric that no
experiment produced.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from backend.routes.deps import get_bundle, get_registry
from backend.services import prediction_service as svc
from backend.services.registry import Registry
from config import REPORTS_DIR

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/{disease}", summary="Saved metrics, model comparison, SHAP and fairness")
def analytics(disease: str, registry: Registry = Depends(get_registry)):
    return svc.analytics(get_bundle(registry, disease))


@router.get("/{disease}/figures/{name}", summary="One saved evaluation figure (PNG)")
def figure(disease: str, name: str, registry: Registry = Depends(get_registry)):
    """Serve a figure by name.

    The name is matched against the files actually present in the disease's figures
    directory rather than being joined onto a path, so a crafted ``name`` cannot escape
    the directory.
    """
    get_bundle(registry, disease)  # 404s on an unknown disease before touching the disk
    figures_dir = REPORTS_DIR / disease / "figures"
    available = {p.name: p for p in figures_dir.glob("*.png")}
    if name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"No figure '{name}' for '{disease}'. Available: "
                   f"{', '.join(sorted(available)) or 'none'}.",
        )
    return FileResponse(available[name], media_type="image/png")
