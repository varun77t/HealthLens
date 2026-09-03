import type { FeatureSpec, FieldState, IntakeValues, SchemaResponse } from "../types";

/** Every field starts blank — nothing is pre-filled with a plausible-looking default. */
export function emptyIntake(schema: SchemaResponse): IntakeValues {
  const out: IntakeValues = {};
  for (const f of schema.features) out[f.name] = { value: null, status: "blank" };
  return out;
}

export function fromSample(
  schema: SchemaResponse,
  features: Record<string, number | null>,
): IntakeValues {
  const out: IntakeValues = {};
  for (const f of schema.features) {
    const v = features[f.name];
    out[f.name] =
      v === null || v === undefined
        ? { value: null, status: "unavailable" }
        : { value: v, status: "sample" };
  }
  return out;
}

/** What actually goes to the model: a value, or null for both "unavailable" and "blank". */
export function toPayload(values: IntakeValues): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  for (const [k, s] of Object.entries(values)) out[k] = s.status === "blank" ? null : s.value;
  return out;
}

export function isAnswered(s: FieldState): boolean {
  return s.status !== "blank";
}

export function hasValue(s: FieldState): boolean {
  return s.value !== null && s.status !== "blank";
}

/**
 * Share of the model's total attribution mass covered by the fields that have values.
 *
 * This is the honest progress bar. "12 of 24 fields" implies every field is worth the
 * same; summing `shap_mass_share` says how much of what the model actually uses is
 * present, so supplying kidney's `sg` (16.4%) counts for more than supplying `su` (0.2%).
 */
export function coverage(schema: SchemaResponse, values: IntakeValues): number {
  let sum = 0;
  for (const f of schema.features) {
    const s = values[f.name];
    if (s && hasValue(s)) sum += f.shap_mass_share;
  }
  return sum;
}

export function coreCoverage(schema: SchemaResponse, values: IntakeValues): number {
  const core = schema.features.filter((f) => f.tier === "core");
  if (!core.length) return 1;
  const have = core.filter((f) => values[f.name] && hasValue(values[f.name])).length;
  return have / core.length;
}

export function unansweredCount(schema: SchemaResponse, values: IntakeValues): number {
  return schema.features.filter((f) => !values[f.name] || !isAnswered(values[f.name])).length;
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

export const pct = (x: number) => `${Math.round(x * 100)}%`;
