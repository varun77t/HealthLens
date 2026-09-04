import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { Disclosure, ErrorBox, Note, PageHead, Spinner, TextField } from "../components/Chrome";
import { formatValue } from "../lib/intake";
import * as routes from "../lib/routes";
import type { AnalysisDetail, FeatureSpec, SchemaResponse } from "../types";

/**
 * One saved analysis, rendered from the stored record.
 *
 * Nothing on this page is recomputed. The probability, the threshold, the band boundaries
 * and the SHAP contributions are the ones the model produced at the time, and they are
 * shown against the threshold that was in force then — not today's. If the model has since
 * changed, the page says so and offers a fresh assessment rather than quietly re-scoring a
 * record against a model that never saw it.
 */
export default function HistoryDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();

  const [item, setItem] = useState<AnalysisDetail | null>(null);
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [label, setLabel] = useState("");

  useEffect(() => {
    setError(null);
    api.analyses
      .get(id)
      .then((got) => {
        setItem(got);
        setLabel(got.label ?? "");
        return api.schema(got.disease).then(setSchema);
      })
      .catch((e) => setError(e.message));
  }, [id]);

  if (error) return <ErrorBox error={error} />;
  if (!item || !schema) return <Spinner label="Loading…" />;

  const featureOf = (name: string): FeatureSpec | undefined =>
    schema.features.find((f) => f.name === name);
  const labelOf = (name: string) => featureOf(name)?.label ?? name;

  // A saved record can name a feature the current schema no longer has — that is the whole
  // point of pinning the model version. Render the stored number rather than crashing on a
  // lookup that legitimately fails.
  const show = (name: string, value: number | null): string => {
    const spec = featureOf(name);
    if (!spec) return value === null ? "—" : String(value);
    return formatValue(spec, value);
  };
  const unitOf = (name: string) => (featureOf(name)?.unit ? ` ${featureOf(name)!.unit}` : "");

  const contributions = (item.explanation?.contributions ?? []).slice(0, 6);
  const provided = Object.entries(item.features).filter(([, v]) => v !== null);

  async function rename() {
    try {
      setItem(await api.analyses.rename(id, label.trim() || null));
      setRenaming(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not rename this entry.");
    }
  }

  async function remove() {
    try {
      await api.analyses.remove(id);
      nav(routes.history, { replace: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete this entry.");
    }
  }

  return (
    <>
      <PageHead
        eyebrow={item.module}
        title={item.label || "Saved analysis"}
        lead={new Date(item.created_at).toLocaleString(undefined, {
          dateStyle: "full",
          timeStyle: "short",
        })}
        back={{ to: routes.history, label: "All saved analyses" }}
      />

      {!item.model_is_current && (
        <div className="mb-8">
          <Note tone="attention">
            <p className="font-medium text-ink">This model has changed since</p>
            <p className="mt-1">
              The result below is exactly what version {item.model_version} produced, at the
              threshold it used. It has not been re-scored against version{" "}
              {item.current_model_version}, because that would attribute a new model's
              decision to an old record.{" "}
              <Link to={routes.start(item.disease)} className="btn-link">
                Run a new assessment
              </Link>{" "}
              to see what the current model says.
            </p>
          </Note>
        </div>
      )}

      <section className="surface p-8">
        <p className="eyebrow">Estimated risk</p>
        <p className="mt-2 text-4xl font-semibold capitalize">{item.band_label}</p>
        <p className="tnum mt-2 text-lg text-body">
          {(item.probability * 100).toFixed(1)}% estimated probability
        </p>
        <p className="mt-1 text-sm text-muted">
          {item.flagged ? "Above" : "Below"} the threshold this model used (
          {(item.threshold * 100).toFixed(1)}%)
        </p>
        {item.source_document && (
          <p className="mt-4 text-sm text-faint">Based on {item.source_document}.</p>
        )}
      </section>

      {contributions.length > 0 && (
        <section className="mt-10">
          <h2 className="text-lg font-semibold">What influenced this estimate?</h2>
          <ul className="mt-4 space-y-2">
            {contributions.map((c) => (
              <li key={c.feature} className="flex items-baseline gap-3 border-b border-hairline py-2.5">
                <span aria-hidden className="text-muted">
                  {c.contribution >= 0 ? "↑" : "↓"}
                </span>
                <span className="flex-1 text-sm text-ink">
                  {labelOf(c.feature)}
                  <span className="ml-2 text-muted">{show(c.feature, c.value)}</span>
                </span>
                <span className="text-xs text-faint">
                  pushed the estimate {c.contribution >= 0 ? "higher" : "lower"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {item.imputed_features.length > 0 && (
        <p className="mt-8 max-w-prose text-sm text-body">
          This analysis did not know your{" "}
          {item.imputed_features.map(labelOf).join(", ")} — those were filled from the
          training data.
        </p>
      )}

      <div className="mt-12 flex flex-wrap items-center gap-3 border-t border-line pt-8">
        <Link to={routes.start(item.disease)} className="btn-primary">
          Run a new assessment
        </Link>
        <button className="btn-secondary" onClick={() => setRenaming((v) => !v)}>
          {renaming ? "Cancel" : "Rename"}
        </button>
        <button className="btn-quiet" onClick={remove}>
          Delete this entry
        </button>
      </div>

      {renaming && (
        <div className="mt-6 max-w-sm space-y-4">
          <TextField label="Label" value={label} onChange={setLabel} autoFocus />
          <button className="btn-primary" onClick={rename}>
            Save label
          </button>
          <p className="text-xs text-muted">
            The label is the only thing a saved analysis will accept a change to. The values
            and the result are a record of what was run, not a draft.
          </p>
        </div>
      )}

      <div className="mt-12 space-y-4">
        <Disclosure summary="What was submitted">
          <dl className="text-sm">
            {provided.map(([name, value]) => (
              <div key={name} className="flex justify-between gap-4 border-b border-hairline py-1.5">
                <dt className="text-muted">{labelOf(name)}</dt>
                <dd className="text-ink">
                  {show(name, value)}
                  {unitOf(name)}
                </dd>
              </div>
            ))}
          </dl>
        </Disclosure>

        <Disclosure summary="How this was estimated">
          <dl className="text-sm">
            {[
              ["Model", `${item.model_name} (${item.calibration} calibration)`],
              ["Version at the time", item.model_version],
              ["Decision threshold", `${(item.threshold * 100).toFixed(2)}%`],
              [
                "Category boundaries",
                `${(item.band_lower * 100).toFixed(1)}% – ${(item.band_upper * 100).toFixed(1)}%`,
              ],
              ["Source", item.source_kind],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 border-b border-hairline py-1.5">
                <dt className="text-muted">{k}</dt>
                <dd className="text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-4">
            <Link to={routes.about(item.disease)} className="btn-link">
              Full model card, dataset and validation details →
            </Link>
          </p>
        </Disclosure>

        {item.warnings.length > 0 && (
          <Disclosure summary={`Caveats that applied to this result (${item.warnings.length})`}>
            <ul className="space-y-2 text-sm text-body">
              {item.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </Disclosure>
        )}
      </div>

      <p className="mt-10 max-w-prose border-t border-line pt-5 text-xs text-faint">
        {item.disclaimer}
      </p>
    </>
  );
}
