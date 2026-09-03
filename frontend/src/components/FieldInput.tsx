import type { FeatureSpec, FieldState } from "../types";
import { outOfObservedRange } from "../lib/intake";

/**
 * One input, rendered from the model's exported feature dictionary.
 *
 * Nothing about a field is hard-coded here — label, help text, allowed options and numeric
 * bounds all arrive from `GET /models/{disease}/schema`, which is generated from the
 * training data. A form built by hand would drift from the model the first time anything
 * was retrained.
 *
 * "Not available" is a first-class answer, distinct from an empty box. Both reach the model
 * as missing, but only the first is a decision, and the review step refuses to run while
 * anything is merely empty.
 */
export function FieldInput({
  feature,
  state,
  onChange,
}: {
  feature: FeatureSpec;
  state: FieldState;
  onChange: (next: FieldState) => void;
}) {
  const unavailable = state.status === "unavailable";
  const extrapolated = outOfObservedRange(feature, state.value);

  return (
    <div className="py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <label
          htmlFor={`f-${feature.name}`}
          className="text-sm font-medium leading-snug text-ink"
        >
          {feature.label}
          {feature.tier === "core" && (
            <span className="ml-2 chip bg-accent/10 text-accent">key field</span>
          )}
        </label>
        <button
          type="button"
          className={`text-xs underline-offset-2 hover:underline ${
            unavailable ? "font-medium text-accent" : "text-muted"
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
        <div className="mt-1.5">
          {feature.options ? (
            <select
              id={`f-${feature.name}`}
              className="field-input"
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
              id={`f-${feature.name}`}
              className="field-input tnum"
              type="number"
              inputMode="decimal"
              step="any"
              min={feature.hard_min}
              max={feature.hard_max}
              placeholder={
                feature.observed_median !== undefined
                  ? `typical: ${feature.observed_median}`
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

      <p className="mt-1.5 text-xs leading-relaxed text-muted">{feature.description}</p>

      {extrapolated && (
        <p className="mt-1.5 text-xs leading-relaxed text-warn">
          Outside the range seen in training ({feature.observed_min}–{feature.observed_max}).
          The model was never fitted on values here, so its output is unreliable. Check the
          units before continuing.
        </p>
      )}
      {unavailable && (
        <p className="mt-1.5 text-xs text-muted">
          Recorded as unavailable. The model will fill it with the value it learned from the
          training data, and the result will say it did.
        </p>
      )}
    </div>
  );
}
