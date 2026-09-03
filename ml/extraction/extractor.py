"""Deterministic extraction of model fields from a text-layer PDF.

No LLM. For documents with a real text layer and a label/value layout, pattern matching
against the model's own schema is more reliable than a language model and cannot hallucinate
a value — which is the property that matters most here.

The pipeline:

    PDF bytes -> words with positions (pdfplumber)
              -> lines, grouped by vertical position
              -> longest-alias label match at the start of a line
              -> value text = the remainder of the line
              -> parse against THAT field's schema (number, or one of its coded options)
              -> ExtractedField with page, confidence and status

Three rules hold throughout:

* **Nothing is invented.** A field whose label is not found, or whose value will not parse,
  comes back ``missing`` with ``value=None``. There is no fallback to a median, a mode or a
  plausible-looking number.
* **A coded field can only take a level the model was trained on**, because values resolve
  against that field's own option list from ``serving.json``.
* **Out-of-range values are surfaced, not corrected.** A number outside the training range
  is returned with ``status="needs_review"`` so a person decides, rather than being clipped.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from config import MODELS_DIR
from ml.extraction.aliases import ALIASES, BP_COMPONENT, NOT_MEASURED_TEXTS

# Vertical tolerance when grouping words into a line, in PDF points.
LINE_TOLERANCE = 2.5

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BP_PAIR = re.compile(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$")


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for label comparison only."""
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text.lower())).strip()


# Normalised through the same function as the text they are compared against, so entries
# like "n/a" cannot silently fail to match once punctuation is stripped to whitespace.
_NOT_MEASURED = frozenset(_norm(t) for t in NOT_MEASURED_TEXTS) - {""}


@dataclass
class ExtractedField:
    name: str
    label: str
    unit: str
    value: float | None
    display: str
    raw_text: str
    page: int | None
    status: str          # "found" | "needs_review" | "missing"
    confidence: str      # "high" | "medium" | "low"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    disease: str
    module: str
    n_expected: int
    n_found: int
    n_needs_review: int
    n_missing: int
    pages: int
    fields: list[ExtractedField] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    note: str = ""

    def values(self) -> dict[str, float | None]:
        """The payload shape ``POST /predict/{disease}`` expects."""
        return {f.name: f.value for f in self.fields}

    def to_dict(self) -> dict:
        return {
            "disease": self.disease,
            "module": self.module,
            "n_expected": self.n_expected,
            "n_found": self.n_found,
            "n_needs_review": self.n_needs_review,
            "n_missing": self.n_missing,
            "pages": self.pages,
            "note": self.note,
            "warnings": self.warnings,
            "fields": [f.to_dict() for f in self.fields],
        }


@lru_cache(maxsize=None)
def _schema(disease: str) -> dict:
    path = MODELS_DIR / disease / "serving.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m scripts.export_serving_assets`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _label_index(disease: str) -> list[tuple[str, str]]:
    """``(normalised label, field name)`` sorted longest-first.

    Longest-first is what keeps "Red blood cell count" from being captured by the shorter
    "Red blood cells", which is a different field on the same report.
    """
    schema = _schema(disease)
    pairs: set[tuple[str, str]] = set()
    for f in schema["features"]:
        pairs.add((_norm(f["label"]), f["name"]))
        for alias in ALIASES.get(disease, {}).get(f["name"], []):
            pairs.add((_norm(alias), f["name"]))
    return sorted(pairs, key=lambda p: -len(p[0]))


def pdf_lines(data: bytes) -> list[tuple[int, str]]:
    """``(page number, line text)`` for every line of a text-layer PDF.

    Lines are rebuilt from word positions rather than taken from ``extract_text`` so that a
    label and its value are reliably on the same line even in a two-column layout.
    """
    import pdfplumber

    out: list[tuple[int, str]] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            rows: dict[float, list[dict]] = {}
            for w in words:
                key = next(
                    (k for k in rows if abs(k - w["top"]) <= LINE_TOLERANCE), w["top"]
                )
                rows.setdefault(key, []).append(w)
            for top in sorted(rows):
                line = " ".join(w["text"] for w in sorted(rows[top], key=lambda w: w["x0"]))
                if line.strip():
                    out.append((page_no, line.strip()))
    return out


def page_count(data: bytes) -> int:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return len(pdf.pages)


def _match_label(disease: str, line: str) -> tuple[str, str] | None:
    """Find the longest known label that this line *starts with*. Returns (field, rest)."""
    norm_line = _norm(line)
    for norm_label, name in _label_index(disease):
        if norm_line == norm_label:
            return name, ""
        if norm_line.startswith(norm_label + " "):
            # Map back to the original line so the value keeps its original casing/format.
            words = line.split()
            for take in range(len(words), 0, -1):
                if _norm(" ".join(words[:take])) == norm_label:
                    return name, " ".join(words[take:])
            return name, norm_line[len(norm_label):].strip()
    return None


def _strip_unit(text: str, unit: str) -> str:
    """Remove a trailing unit so "9.6 g/dL" parses as 9.6."""
    if not unit:
        return text.strip()
    pattern = re.compile(re.escape(unit) + r"\s*$", re.IGNORECASE)
    return pattern.sub("", text).strip()


def _parse_value(disease: str, feature: dict, raw: str) -> tuple[float | None, str, str]:
    """``(value, status, note)`` for one field's raw text, judged against its own schema."""
    text = raw.strip()
    if not text:
        return None, "missing", "No value found next to this label in the document."
    if _norm(text) in _NOT_MEASURED:
        return None, "missing", f"The report records this as '{text}'."

    # Coded field: resolve against its own options, never a free number.
    options = feature.get("options")
    if options:
        for o in options:
            if _norm(str(o["label"])) == _norm(text):
                return float(o["value"]), "found", ""
        number = _NUMBER.search(text)
        if number:
            v = float(number.group())
            if any(o["value"] == v for o in options):
                return v, "found", ""
        allowed = ", ".join(str(o["label"]) for o in options)
        return None, "needs_review", (
            f"'{text}' is not one of the values this model recognises ({allowed})."
        )

    stripped = _strip_unit(text, feature.get("unit", ""))

    # A blood pressure printed as one pair: take the half this specific field was trained on.
    pair = _BP_PAIR.match(stripped)
    if pair:
        component = BP_COMPONENT.get(f"{disease}.{feature['name']}")
        if component:
            v = float(pair.group(1) if component == "systolic" else pair.group(2))
            return v, "found", f"Read the {component} value from '{stripped}'."
        return None, "needs_review", (
            f"'{stripped}' is a blood-pressure pair and this field expects a single number."
        )

    number = _NUMBER.search(stripped)
    if not number:
        return None, "needs_review", f"Could not read a number from '{text}'."
    v = float(number.group())

    lo, hi = feature.get("observed_min"), feature.get("observed_max")
    if lo is not None and hi is not None and not (lo <= v <= hi):
        return v, "needs_review", (
            f"{v:g} is outside the range this model was trained on ({lo:g}-{hi:g}). "
            "Check the units on the report."
        )
    return v, "found", ""


def extract(disease: str, data: bytes) -> ExtractionResult:
    """Read every field the model needs out of one PDF. Never invents a value."""
    schema = _schema(disease)
    features = {f["name"]: f for f in schema["features"]}
    lines = pdf_lines(data)

    # Ranked so a stronger later match can displace a weaker earlier one. Section headings
    # are the reason: a heading like "RESTING ELECTROCARDIOGRAM" is a legitimate alias for
    # `restecg`, sits above the data row, and carries no value — taking the first match
    # outright would let a heading swallow the field it introduces.
    strength = {"found": 2, "needs_review": 1, "missing": 0}
    hits: dict[str, tuple[float | None, str, str, str, int]] = {}
    for page_no, line in lines:
        matched = _match_label(disease, line)
        if not matched:
            continue
        name, rest = matched
        value, status, note = _parse_value(disease, features[name], rest)
        if name in hits and strength[status] <= strength[hits[name][1]]:
            continue  # keep the earlier match unless this one is strictly better evidence
        hits[name] = (value, status, note, line, page_no)

    fields: list[ExtractedField] = []
    for name in schema["feature_order"]:
        f = features[name]
        if name in hits:
            value, status, note, line, page_no = hits[name]
            confidence = {"found": "high", "needs_review": "low", "missing": "medium"}[status]
            display = _display(f, value) if value is not None else "—"
            fields.append(ExtractedField(
                name=name, label=f["label"], unit=f.get("unit", ""), value=value,
                display=display, raw_text=line, page=page_no,
                status=status, confidence=confidence, note=note,
            ))
        else:
            fields.append(ExtractedField(
                name=name, label=f["label"], unit=f.get("unit", ""), value=None,
                display="—", raw_text="", page=None, status="missing", confidence="low",
                note="Not found in this document.",
            ))

    n_found = sum(f.status == "found" for f in fields)
    n_review = sum(f.status == "needs_review" for f in fields)
    n_missing = sum(f.status == "missing" for f in fields)

    warnings: list[str] = []
    if not lines:
        warnings.append(
            "No text layer was found in this PDF. It is probably a scan or a photograph; "
            "image extraction is not available yet, so please enter the values manually."
        )
    if n_missing:
        warnings.append(
            f"{n_missing} of {len(fields)} fields were not found in the document. They are "
            "left blank rather than guessed — complete or mark them on the next screen."
        )
    if n_review:
        warnings.append(
            f"{n_review} value{'s' if n_review != 1 else ''} could not be read confidently "
            "and need checking."
        )

    return ExtractionResult(
        disease=disease,
        module=schema["module"],
        n_expected=len(fields),
        n_found=n_found,
        n_needs_review=n_review,
        n_missing=n_missing,
        pages=page_count(data) if data else 0,
        fields=fields,
        warnings=warnings,
        note=(
            "Values were read directly from the uploaded document by pattern matching "
            "against this model's own field list. Nothing was inferred, completed or "
            "guessed: anything not found is left blank for you to supply."
        ),
    )


def _display(feature: dict, value: float) -> str:
    opts = feature.get("options")
    if opts:
        for o in opts:
            if o["value"] == value:
                return str(o["label"])
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
