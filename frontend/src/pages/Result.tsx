import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { Disclosure, Note, PageHead } from "../components/Chrome";
import { formatValue } from "../lib/intake";
import { useSession } from "../state";
import type { Contribution, Disease, FeatureSpec, ModelSummary } from "../types";

const TITLES: Record<string, string> = {
  heart: "Your heart health analysis",
  kidney: "Your kidney health analysis",
  diabetes: "Your diabetes health analysis",
};

/**
 * The result.
 *
 * The **category** is the headline rather than the percentage, and that is a deliberate
 * departure from showing a bare number first. The category's upper boundary is the model's
 * own operating threshold, so it cannot contradict the model's decision — whereas a
 * percentage read against an assumed 50% can. For diabetes the threshold is 13.9%, so
 * "14.5%" looks reassuring and means the opposite.
 *
 * SHAP is translated into plain sentences with a direction. The raw values, the methodology
 * and every measured metric stay available underneath, in the disclosure.
 */
export default function Result() {
  const { disease } = useParams<{ disease: Disease }>();
  const nav = useNavigate();
  const session = useSession();
  const [card, setCard] = useState<ModelSummary | null>(null);
  const [downloading, setDownloading] = useState(false);

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
    nav(`/app/${disease ?? ""}`, { replace: true });
    return null;
  }

  const featureOf = (name: string): FeatureSpec | undefined =>
    schema.features.find((f) => f.name === name);

  const download = async () => {
    setDownloading(true);
    try {
      const blob = await api.reportPdf(disease!, {
        ...p,
        ...(session.sourceDocument
          ? { source_document: `${session.sourceDocument} (synthetic demonstration document)` }
          : {}),
      } as never);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${disease}-analysis.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  };

  const band = p.risk_band.label;
  const tone = p.flagged
    ? "border-attention-line bg-attention-soft"
    : "border-steady-line bg-steady-soft";

  return (
    <div>
      <PageHead
        title={TITLES[disease!] ?? p.module}
        back={{ to: `/app/${disease}/review`, label: "Change an answer" }}
      />

      {/* Category first, percentage second. See the note at the top of this file. */}
      <section className={`rounded-2xl border p-8 ${tone}`}>
        <p className="eyebrow">Estimated risk</p>
        <p className="mt-2 text-5xl font-semibold capitalize tracking-tight text-ink">
          {band}
        </p>
        <p className="tnum mt-4 text-lg text-body">
          {(p.probability * 100).toFixed(1)}% estimated probability
        </p>
        <p className="mt-1 text-sm text-muted">
          {p.flagged ? "Above" : "Below"} the threshold this model uses (
          <span className="tnum">{(p.threshold * 100).toFixed(1)}%</span>)
        </p>
        <p className="mt-6 max-w-prose border-t border-line/60 pt-5 text-sm text-body">
          This is a model-estimated probability based on the information you provided. It is
          not a diagnosis, and it does not say what will happen in future.
        </p>
      </section>

      {session.sourceDocument && (
        <p className="mt-4 text-xs text-faint">
          Based on {session.sourceDocument}, reviewed by you before analysis.
        </p>
      )}

      {/* --- factors, in plain language ------------------------------------------- */}
      {p.explanation && (
        <section className="mt-12">
          <h2 className="text-xl font-semibold">What influenced this estimate?</h2>
          <p className="mt-2 max-w-prose text-sm text-muted">
            The information that moved this estimate most. These describe how the model
            weighed your details — not causes of disease, and not things to change.
          </p>
          <ul className="mt-6 space-y-3">
            {p.explanation.contributions.slice(0, 6).map((c) => (
              <Factor key={c.feature} contribution={c} feature={featureOf(c.feature)} />
            ))}
          </ul>
        </section>
      )}

      {/* --- what it knew ---------------------------------------------------------- */}
      <section className="mt-12">
        <h2 className="text-xl font-semibold">What this analysis knew</h2>
        <p className="mt-2 max-w-prose text-sm text-body">
          You provided {p.n_features_provided} of {p.n_features_expected} pieces of
          information.
          {p.imputed_features.length > 0 && (
            <>
              {" "}
              The rest were filled from the training data, so the estimate didn't know your{" "}
              {p.imputed_features
                .map((n) => (featureOf(n)?.label ?? n).toLowerCase())
                .join(", ")}
              .
            </>
          )}
        </p>
        {p.extrapolated_features.length > 0 && (
          <p className="mt-3 max-w-prose text-sm text-caution">
            Some values sit outside the range this model was trained on (
            {p.extrapolated_features.map((e) => featureOf(e.feature)?.label ?? e.feature).join(", ")}
            ), so its output there is unreliable.
          </p>
        )}
      </section>

      {/* --- actions --------------------------------------------------------------- */}
      <div className="mt-12 flex flex-wrap items-center gap-3 border-t border-line pt-8">
        <button className="btn-primary" onClick={download} disabled={downloading}>
          {downloading ? "Preparing…" : "↓ Download report"}
        </button>
        <button className="btn-secondary" onClick={() => nav(`/app/${disease}/review`)}>
          Change an answer
        </button>
        <button
          className="btn-quiet"
          onClick={() => {
            session.reset();
            nav("/app");
          }}
        >
          Start over
        </button>
      </div>

      {/* --- the research layer, one click away ------------------------------------ */}
      <div className="mt-12 space-y-4">
        <Disclosure summary="How was this estimated?">
          <div className="space-y-5 text-sm text-body">
            <Facts
              rows={[
                ["Model", `${p.model_name} (${p.calibration} calibration)`],
                ["Version", p.model_version],
                ["Decision threshold", `${(p.threshold * 100).toFixed(2)}%`],
                ["Estimated probability", `${(p.probability * 100).toFixed(2)}%`],
                [
                  "Category boundaries",
                  `${(p.risk_band.lower * 100).toFixed(1)}% – ${(p.risk_band.upper * 100).toFixed(1)}%`,
                ],
                ["Positive class", p.positive_class_meaning],
              ]}
            />
            <p className="max-w-prose">{p.threshold_rule}</p>
            <p className="max-w-prose">{p.risk_band.note}</p>
            {p.explanation && (
              <>
                <p className="max-w-prose border-t border-line pt-4">{p.explanation.note}</p>
                <div>
                  <p className="eyebrow mb-2">Raw SHAP contributions</p>
                  <dl className="text-sm">
                    {p.explanation.contributions.slice(0, 12).map((c) => (
                      <div
                        key={c.feature}
                        className="flex justify-between gap-4 border-b border-hairline py-1.5"
                      >
                        <dt className="text-muted">{featureOf(c.feature)?.label ?? c.feature}</dt>
                        <dd className="tnum text-ink">
                          {c.contribution >= 0 ? "+" : ""}
                          {c.contribution.toFixed(4)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </>
            )}
            <p>
              <Link to={`/app/${disease}/about`} className="btn-link">
                Full model card, dataset and validation details →
              </Link>
            </p>
          </div>
        </Disclosure>

        {p.warnings.length > 0 && (
          <Disclosure summary={`Caveats that apply to this result (${p.warnings.length})`}>
            <ul className="max-w-prose space-y-3 text-sm text-body">
              {p.warnings.map((w, i) => (
                <li key={i}>• {w}</li>
              ))}
            </ul>
          </Disclosure>
        )}

        {card && card.limitations.length > 0 && (
          <Disclosure summary={`Known limitations of this model (${card.limitations.length})`}>
            <ul className="max-w-prose space-y-3 text-sm text-body">
              {card.limitations.map((l, i) => (
                <li key={i}>• {l}</li>
              ))}
            </ul>
          </Disclosure>
        )}
      </div>

      <div className="mt-8">
        <Note>
          This is a research and education tool. It cannot diagnose anything and is not a
          substitute for a clinician. If you have a health concern, speak to a doctor.
        </Note>
      </div>
    </div>
  );
}

function Factor({
  contribution,
  feature,
}: {
  contribution: Contribution;
  feature?: FeatureSpec;
}) {
  const up = contribution.contribution >= 0;
  const label = feature?.label ?? contribution.feature;
  const shown =
    contribution.value === null
      ? "not provided"
      : feature
        ? `${formatValue(feature, contribution.value)}${feature.unit ? ` ${feature.unit}` : ""}`
        : String(contribution.value);

  return (
    <li className="surface flex items-start gap-4 p-5">
      <span
        aria-hidden
        className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
          up ? "bg-attention-soft text-attention" : "bg-steady-soft text-steady"
        }`}
      >
        {up ? "↑" : "↓"}
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-medium text-ink">
          {label}
          <span className="tnum ml-2 font-normal text-muted">{shown}</span>
        </span>
        <span className="mt-1 block text-sm text-muted">
          {up
            ? "pushed the estimate higher"
            : "pushed the estimate lower"}
        </span>
      </span>
    </li>
  );
}

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4 border-b border-hairline py-1.5">
          <dt className="text-muted">{k}</dt>
          <dd className="tnum text-right text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
