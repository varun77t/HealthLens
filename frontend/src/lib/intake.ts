import type {
  ExtractionResult,
  FeatureSpec,
  FieldState,
  IntakeValues,
  SchemaResponse,
} from "../types";

/** Every field starts blank — nothing is pre-filled with a plausible-looking default. */
export function emptyIntake(schema: SchemaResponse): IntakeValues {
  const out: IntakeValues = {};
  for (const f of schema.features) out[f.name] = { value: null, status: "blank" };
  return out;
}

/**
 * Turn an extraction result into form values.
 *
 * A field the document did not yield becomes `blank`, not `unavailable` — even when the
 * report said "Not performed". Both reach the model as missing, but `blank` is what the
 * review screen refuses to run on, so the person has to make that call rather than inherit
 * it from a parser. For kidney that matters more than it looks: missingness there is
 * confounded with the outcome, so an unconsidered blank is a real risk, not a formality.
 */
export function fromExtraction(
  schema: SchemaResponse,
  result: ExtractionResult,
): IntakeValues {
  const byName = new Map(result.fields.map((f) => [f.name, f]));
  const out: IntakeValues = {};
  for (const f of schema.features) {
    const got = byName.get(f.name);
    if (!got || got.value === null) {
      out[f.name] = { value: null, status: "blank" };
    } else {
      out[f.name] = {
        value: got.value,
        status: got.status === "found" ? "extracted" : "flagged",
      };
    }
  }
  return out;
}

/** What actually goes to the model: a value, or null for both "unavailable" and "blank". */
export function toPayload(values: IntakeValues): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  for (const [k, s] of Object.entries(values)) out[k] = s.status === "blank" ? null : s.value;
  return out;
}

export function hasValue(s: FieldState): boolean {
  return s.value !== null && s.status !== "blank";
}

export function byGroup(schema: SchemaResponse): Map<string, FeatureSpec[]> {
  const map = new Map<string, FeatureSpec[]>();
  for (const g of schema.groups) map.set(g.id, []);
  for (const f of schema.features) {
    if (!map.has(f.group)) map.set(f.group, []);
    map.get(f.group)!.push(f);
  }
  // Within a group, most important first — attention goes where it changes the answer.
  for (const list of map.values()) list.sort((a, b) => a.shap_rank - b.shap_rank);
  return map;
}

export function formatValue(f: FeatureSpec, v: number | null): string {
  if (v === null) return "—";
  const opt = f.options?.find((o) => o.value === v);
  if (opt) return opt.label;
  return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(3)));
}

export function outOfObservedRange(f: FeatureSpec, v: number | null): boolean {
  if (v === null || f.kind !== "numeric") return false;
  if (f.observed_min === undefined || f.observed_max === undefined) return false;
  return v < f.observed_min || v > f.observed_max;
}

