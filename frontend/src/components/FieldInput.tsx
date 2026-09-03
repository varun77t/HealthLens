import type { FeatureSpec, FieldState } from "../types";
import { outOfObservedRange } from "../lib/intake";

/**
 * One input, rendered from the model's exported feature dictionary.
 *
 * Nothing about a field is hard-coded: label, help text, unit, allowed options and numeric
 * bounds all arrive from `GET /models/{disease}/schema`, which is generated from the
 * training data. A hand-built form would drift from the model the first time anything was
 * retrained.
 *
 * "I don't have this" is a first-class answer, distinct from an empty box. Both reach the
 * model as missing, but only the first is a decision, and review refuses to run while
 * anything is merely empty.
 */
export function FieldInput({
  feature,
  state,
  onChange,
  compact = false,
}: {
  feature: FeatureSpec;
  state: FieldState;
  onChange: (next: FieldState) => void;
  compact?: boolean;
}) {
  const unavailable = state.status === "unavailable";
  const extrapolated = outOfObservedRange(feature, state.value);
  const id = `f-${feature.name}`;

  return (
    <div className={compact ? "" : "py-4"}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <label htmlFor={id} className="text-sm font-medium text-ink">
          {feature.label}
          {feature.unit && <span className="ml-1.5 text-xs text-faint">({feature.unit})</span>}
        </label>
        <button
          type="button"
          className={`text-xs underline-offset-4 hover:underline ${
            unavailable ? "font-medium text-accent" : "text-faint hover:text-muted"
          }`}
          aria-pressed={unavailable}
          onClick={() =>
            onChange(
              unavailable
                ? { value: null, status: "blank" }
                : { value: null, status: "unavailable" },
            )
          }
        >
          {unavailable ? "I do have this" : "I don't have this"}
        </button>
      </div>

      {!unavailable && (
        <div className="mt-2">
          {feature.options ? (
            <select
              id={id}
              className="input"
              value={state.value === null ? "" : String(state.value)}
              onChange={(e) =>
                onChange(
                  e.target.value === ""
                    ? { value: null, status: "blank" }
                    : { value: Number(e.target.value), status: "entered" },
                )
              }
            >
              <option value="">Select…</option>
              {feature.options.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          ) : (
            <input
              id={id}
              className="input tnum"
              type="number"
              inputMode="decimal"
              step="any"
              min={feature.hard_min}
              max={feature.hard_max}
              placeholder={
                feature.observed_median !== undefined
                  ? `typically around ${feature.observed_median}`
                  : ""
              }
              value={state.value === null ? "" : state.value}
              onChange={(e) =>
                onChange(
                  e.target.value === ""
                    ? { value: null, status: "blank" }
                    : { value: Number(e.target.value), status: "entered" },
                )
              }
            />
          )}
        </div>
      )}

      {extrapolated && (
        <p className="mt-2 text-xs text-caution">
          Outside the range seen in training ({feature.observed_min}–{feature.observed_max}
          {feature.unit ? ` ${feature.unit}` : ""}). Worth checking the units.
        </p>
      )}
      {unavailable && (
        <p className="mt-2 text-xs text-muted">
          Recorded as unavailable. The analysis will note that it didn't know this.
        </p>
      )}
    </div>
  );
}
