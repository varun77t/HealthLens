import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { Banner } from "../components/Chrome";
import { formatValue, pct } from "../lib/intake";
import { useSession } from "../state";
import type { Disease, Explanation, ModelSummary } from "../types";

export default function Result() {
  const { disease } = useParams<{ disease: Disease }>();
  const nav = useNavigate();
  const session = useSession();
  const [card, setCard] = useState<ModelSummary | null>(null);

  const p = session.prediction;
  const schema = session.schema;

  useEffect(() => {
    if (!disease) return;
    api
      .models()
      .then((ms) => setCard(ms.find((m) => m.disease === disease) ?? null))
      .catch(() => setCard(null));
  }, [disease]);

  if (!p || !schema || !session.confirmed) {
    nav(`/${disease ?? ""}`, { replace: true });
    return null;
  }

  const sample = session.sampleId;

  return (
    <div className="space-y-6">
      <div>
        <button
          className="btn-ghost -ml-2 mb-2 text-xs"
          onClick={() => nav(`/${disease}/review`)}
        >
          ← Change an answer
        </button>
        <h1 className="text-2xl font-semibold tracking-tight">{p.module}</h1>
      </div>

      {/* The decision, not the probability, is the headline — it is what the model
          actually concluded, taken at its own threshold rather than at 0.5. */}
      <section
        className={`card p-6 ${p.flagged ? "border-flagged/30 bg-flaggedBg" : "border-clear/30 bg-clearBg"}`}
      >
        <p className="text-xs uppercase tracking-wide text-muted">
          At this model&apos;s operating threshold
        </p>
        <p className="mt-2 text-xl font-semibold leading-snug">
          {p.flagged
            ? "This model would flag this case for follow-up."
            : "This model would not flag this case."}
        </p>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed">
          It is not a diagnosis and not a statement about what will happen. It means the
          case does or does not resemble the records the model was trained to identify as{" "}
          <em>{p.positive_class_meaning}</em>.
        </p>

        <dl className="mt-5 grid gap-4 border-t border-line/60 pt-4 sm:grid-cols-3">
          <div>
            <dt className="text-xs text-muted">Estimated probability</dt>
            <dd className="tnum mt-0.5 text-lg font-semibold">
              {(p.probability * 100).toFixed(1)}%
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Decision threshold</dt>
            <dd className="tnum mt-0.5 text-lg font-semibold">
              {(p.threshold * 100).toFixed(1)}%
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Presentation band</dt>
            <dd className="mt-0.5 text-lg font-semibold capitalize">{p.risk_band.label}</dd>
          </div>
        </dl>
      </section>

      <Banner title="Why the threshold is not 50%">
        <p>{p.threshold_rule}</p>
        <p className="mt-2">{p.risk_band.note}</p>
      </Banner>

      {sample && (
        <Banner title="This was an example case">
          You loaded <span className="tnum font-medium">{sample}</span> — a record from the
          held-out test split, which the model was never trained on. The outcome recorded in
          the source dataset is shown on the form screen. One case agreeing or disagreeing
          with the model proves nothing either way; the model card&apos;s metrics come from{" "}
          <span className="tnum">{card?.n_test.toLocaleString() ?? "all"}</span> such rows.
        </Banner>
      )}

      <section className="card p-5">
        <h2 className="text-sm font-semibold">What the model used</h2>
        <p className="mt-1 text-xs leading-relaxed text-muted">
          {p.n_features_provided} of {p.n_features_expected} fields came from you.
          {p.imputed_features.length > 0 && (
            <>
              {" "}
              The remaining {p.imputed_features.length} were filled from the training data:{" "}
              <span className="text-ink">
                {p.imputed_features
                  .map((n) => schema.features.find((f) => f.name === n)?.label ?? n)
                  .join(", ")}
              </span>
              .
            </>
          )}
        </p>
        {p.extrapolated_features.length > 0 && (
          <p className="mt-2 text-xs leading-relaxed text-warn">
            Outside the training range:{" "}
            {p.extrapolated_features.map((e) => e.feature).join(", ")}. The model has no
            support for values there.
          </p>
        )}
      </section>

      {p.warnings.length > 0 && (
        <section className="card p-5">
          <h2 className="text-sm font-semibold">Caveats that apply to this result</h2>
          <ul className="mt-3 space-y-2.5">
            {p.warnings.map((w, i) => (
              <li key={i} className="flex gap-2.5 text-xs leading-relaxed text-muted">
                <span aria-hidden className="text-warn">
                  ▲
                </span>
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {p.explanation && <ShapPanel explanation={p.explanation} schema={schema} />}

      {card && (
        <section className="card p-5">
          <h2 className="text-sm font-semibold">Known limitations of this model</h2>
          <ul className="mt-3 space-y-2.5">
            {card.limitations.map((l, i) => (
              <li key={i} className="text-xs leading-relaxed text-muted">
                • {l}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="flex flex-wrap gap-3">
        <button className="btn-secondary" onClick={() => nav(`/${disease}/review`)}>
          Change an answer
        </button>
        <button
          className="btn-ghost"
          onClick={() => {
            session.reset();
            nav("/");
          }}
        >
          Start over
        </button>
      </div>
    </div>
  );
}

/**
 * Per-feature SHAP contributions, drawn from a shared centre so direction reads at a glance.
 *
 * The panel says plainly that these explain the uncalibrated model rather than the
 * calibrated probability above — the API returns both numbers precisely so the UI does not
 * have to imply the bars sum to the headline figure.
 */
function ShapPanel({
  explanation,
  schema,
}: {
  explanation: Explanation;
  schema: { features: { name: string; label: string; options: { value: number; label: string }[] | null; kind: string }[] };
}) {
  const top = explanation.contributions.slice(0, 10);
  const max = Math.max(...top.map((c) => Math.abs(c.contribution)), 1e-9);

  return (
    <section className="card p-5">
      <h2 className="text-sm font-semibold">What pushed this estimate</h2>
      <p className="mt-1 text-xs leading-relaxed text-muted">
        The ten fields that moved this case most, from SHAP on the model itself. Bars to the
        right pushed towards {""}
        <span className="text-ink">the positive class</span>; bars to the left pushed away.
      </p>

      <ul className="mt-4 space-y-2">
        {top.map((c) => {
          const f = schema.features.find((x) => x.name === c.feature);
          const w = (Math.abs(c.contribution) / max) * 50;
          const positive = c.contribution >= 0;
          return (
            <li key={c.feature} className="grid grid-cols-[minmax(0,11rem)_1fr_auto] items-center gap-3">
              <span className="truncate text-xs" title={f?.label ?? c.feature}>
                {f?.label ?? c.feature}
              </span>
              <span className="relative h-4 rounded bg-canvas">
                <span className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <span
                  className={`absolute inset-y-0.5 rounded ${positive ? "bg-flagged/70" : "bg-clear/70"}`}
                  style={
                    positive
                      ? { left: "50%", width: `${w}%` }
                      : { right: "50%", width: `${w}%` }
                  }
                />
              </span>
              <span className="tnum w-20 text-right text-xs text-muted">
                {f ? formatValue(f as never, c.value) : (c.value ?? "—")}
              </span>
            </li>
          );
        })}
      </ul>

      <p className="mt-4 border-t border-line pt-3 text-xs leading-relaxed text-muted">
        {explanation.note}
      </p>
      <p className="mt-2 text-xs text-muted">
        Explainer: <span className="text-ink">{explanation.explainer}</span> · uncalibrated
        model output{" "}
        <span className="tnum text-ink">{pct(explanation.uncalibrated_probability)}</span>
      </p>
    </section>
  );
}
