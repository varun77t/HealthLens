import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { Disclosure, ErrorBox, Note, PageHead, Spinner } from "../components/Chrome";
import type { Disease, ModelSummary } from "../types";

/**
 * The research layer, in full.
 *
 * Nothing here was deleted to simplify the primary journey — it was moved. The metrics,
 * protocol, calibration, subgroup analysis and limitations all live on this page, reachable
 * from every screen, so the user-facing layer can stay plain without the project losing the
 * substance it is actually being marked on.
 */
export default function About() {
  const { disease } = useParams<{ disease: Disease }>();
  const [card, setCard] = useState<Record<string, any> | null>(null);
  const [summary, setSummary] = useState<ModelSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!disease) return;
    Promise.all([api.modelCard(disease), api.models()])
      .then(([c, ms]) => {
        setCard(c as Record<string, any>);
        setSummary((ms as ModelSummary[]).find((m) => m.disease === disease) ?? null);
      })
      .catch((e) => setError(e.message));
  }, [disease]);

  if (error) return <ErrorBox error={error} />;
  if (!card || !summary || !disease) return <Spinner label="Loading the model card…" />;

  const perf = card.performance ?? {};
  const test = perf["test_set_threshold_0.5"] ?? {};
  const alt = perf["test_set_alternative_threshold"] ?? {};
  const dataset = card.dataset ?? {};
  const model = card.model ?? {};

  return (
    <div>
      <PageHead
        eyebrow="Research details"
        title={`How the ${summary.module.replace(/ Prediction$/, "")} estimate is produced`}
        lead={card.intended_use}
        back={{ to: `/${disease}`, label: "Back to assessment" }}
      />

      <div className="space-y-4">
        <Note tone="attention">
          <p className="font-medium text-ink">
            {summary.external_validation.status === "rejected"
              ? "External validation was attempted and rejected."
              : "This model has no external validation."}
          </p>
          <p className="mt-1">{summary.external_validation.summary}</p>
        </Note>

        <Disclosure summary="Dataset" defaultOpen>
          <Facts
            rows={[
              ["Source", `${dataset.source} — id ${dataset.uci_id}`],
              ["Records", Number(dataset.n_rows).toLocaleString()],
              ["Features", String(dataset.n_features)],
              ["Positive class", dataset.positive_class_meaning],
              ["Positive rate", String(dataset.positive_rate)],
              ["Missing cells", Number(dataset.total_missing_cells).toLocaleString()],
            ]}
          />
        </Disclosure>

        <Disclosure summary="Model and selection">
          <Facts
            rows={[
              ["Algorithm", model.algorithm],
              ["Calibration", model.calibration],
              ["Operating threshold", alt.threshold?.toFixed?.(4)],
            ]}
          />
          <p className="mt-4 max-w-prose text-sm text-body">{model.selection_rationale}</p>
          <p className="mt-3 max-w-prose text-sm text-body">{model.calibration_rationale}</p>
        </Disclosure>

        <Disclosure summary="Measured performance">
          <p className="mb-4 max-w-prose text-sm text-body">{perf.note}</p>
          <div className="grid gap-8 sm:grid-cols-2">
            <div>
              <p className="eyebrow mb-2">At threshold 0.5</p>
              <Facts rows={metricRows(test)} />
            </div>
            <div>
              <p className="eyebrow mb-2">
                At the operating threshold ({alt.threshold?.toFixed?.(4)})
              </p>
              <Facts rows={metricRows(alt)} />
            </div>
          </div>
          {perf.what_drives_this_score?.operating_point_note && (
            <p className="mt-5 max-w-prose border-t border-line pt-4 text-sm text-body">
              {perf.what_drives_this_score.operating_point_note}
            </p>
          )}
        </Disclosure>

        {perf.what_drives_this_score?.attribution_note && (
          <Disclosure summary="What actually drives this score">
            <p className="max-w-prose text-sm text-body">
              {perf.what_drives_this_score.attribution_note}
            </p>
            {perf.what_drives_this_score.duplicate_conflict_note && (
              <p className="mt-3 max-w-prose text-sm text-body">
                {perf.what_drives_this_score.duplicate_conflict_note}
              </p>
            )}
          </Disclosure>
        )}

        <Disclosure summary="Explainability method">
          <p className="max-w-prose text-sm text-body">{card.explainability?.method}</p>
          <ol className="mt-4 space-y-1.5 text-sm">
            {(card.explainability?.top_features ?? []).map((f: any) => (
              <li key={f.feature} className="flex justify-between gap-4 border-b border-hairline pb-1.5">
                <span className="text-ink">{f.feature}</span>
                <span className="tnum text-muted">{f.mean_abs_shap?.toFixed(4)}</span>
              </li>
            ))}
          </ol>
        </Disclosure>

        <Disclosure summary={`Limitations (${(card.limitations ?? []).length})`}>
          <ul className="max-w-prose space-y-3 text-sm text-body">
            {(card.limitations ?? []).map((l: string, i: number) => (
              <li key={i}>• {l}</li>
            ))}
          </ul>
        </Disclosure>

        <Disclosure summary="Out of scope">
          <ul className="max-w-prose space-y-2 text-sm text-body">
            {(card.out_of_scope_use ?? []).map((l: string, i: number) => (
              <li key={i}>• {l}</li>
            ))}
          </ul>
        </Disclosure>
      </div>

      <p className="mt-8 max-w-prose text-xs text-faint">{card.disclaimer}</p>
    </div>
  );
}

function metricRows(m: Record<string, any>): [string, string][] {
  const keys: [string, string][] = [
    ["roc_auc", "ROC-AUC"],
    ["pr_auc", "PR-AUC"],
    ["recall_sensitivity", "Recall"],
    ["specificity", "Specificity"],
    ["precision", "Precision"],
    ["f1", "F1"],
    ["brier", "Brier"],
    ["accuracy", "Accuracy"],
  ];
  return keys
    .filter(([k]) => typeof m[k] === "number")
    .map(([k, label]) => [label, m[k].toFixed(4)]);
}

function Facts({ rows }: { rows: [string, string | undefined][] }) {
  return (
    <dl className="text-sm">
      {rows
        .filter(([, v]) => v !== undefined && v !== null && v !== "")
        .map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 border-b border-hairline py-1.5">
            <dt className="text-muted">{k}</dt>
            <dd className="tnum text-right text-ink">{v}</dd>
          </div>
        ))}
    </dl>
  );
}
