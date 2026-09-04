import { Fragment, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api";
import { ErrorBox, Note, PageHead } from "../components/Chrome";
import { FieldInput } from "../components/FieldInput";
import { byGroup, formatValue, outOfObservedRange, toPayload } from "../lib/intake";
import { useSession } from "../state";
import type { Disease, ExtractedField, FeatureSpec, FieldState } from "../types";

/**
 * The verification step. Nothing is predicted until every field has been settled.
 *
 * Grouped the way the information physically arrives (patient details, blood tests, urine
 * tests) rather than by model importance, because this screen is read against a document.
 * A field that is merely untouched blocks the run: leaving it blank and saying you don't
 * have it both reach the model as missing, but only one of them is a decision a person made.
 */
export default function Review() {
  const { disease } = useParams<{ disease: Disease }>();
  const nav = useNavigate();
  const session = useSession();
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const schema = session.schema;
  const values = session.values;
  const extraction = session.extraction;

  const groups = useMemo(
    () => (schema ? byGroup(schema) : new Map<string, FeatureSpec[]>()),
    [schema],
  );
  const extractedByName = useMemo(
    () => new Map((extraction?.fields ?? []).map((f) => [f.name, f])),
    [extraction],
  );

  if (!schema || !disease || session.disease !== disease) {
    nav(`/app/${disease ?? ""}`, { replace: true });
    return null;
  }

  const unsettled = schema.features.filter(
    (f) => (values[f.name]?.status ?? "blank") === "blank",
  );
  const fromDocument = Boolean(extraction);

  const update = (name: string, next: FieldState) =>
    session.setValues({ ...values, [name]: next });

  const markRemainingUnavailable = () => {
    const next = { ...values };
    for (const f of unsettled) next[f.name] = { value: null, status: "unavailable" };
    session.setValues(next);
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const prediction = await api.predict(disease, toPayload(values));
      session.setPrediction(prediction);
      session.setConfirmed(true);
      nav(`/app/${disease}/result`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <PageHead
        title={fromDocument ? "We found your information" : "Review your information"}
        lead="Please review the details before we run the analysis."
        back={{
          to: fromDocument ? `/app/${disease}` : `/app/${disease}/enter`,
          label: fromDocument ? "Upload a different report" : "Back to the form",
        }}
      />

      {extraction && (
        <div className="mb-8">
          <Note>
            <p>
              Read <span className="font-medium text-ink">{extraction.n_found}</span> of{" "}
              {extraction.n_expected} pieces of information from{" "}
              <span className="font-medium text-ink">{session.sourceDocument}</span>
              {extraction.n_missing > 0 && (
                <>
                  {" "}
                  · {extraction.n_missing} not found in the document
                </>
              )}
              {extraction.n_needs_review > 0 && (
                <>
                  {" "}
                  · {extraction.n_needs_review} to check
                </>
              )}
              .
            </p>
            <p className="mt-1.5 text-xs text-muted">
              Anything not found is left blank rather than guessed.
            </p>
          </Note>
        </div>
      )}

      {error && (
        <div className="mb-8">
          <ErrorBox error={error} />
        </div>
      )}

      <div className="space-y-4">
        {schema.groups.map((g) => {
          const fields = groups.get(g.id) ?? [];
          if (!fields.length) return null;
          return (
            <section key={g.id} className="surface overflow-hidden">
              <h2 className="border-b border-line px-6 py-4 text-sm font-semibold text-ink">
                {g.label}
              </h2>
              <dl className="divide-y divide-hairline">
                {fields.map((f) => (
                  <Fragment key={f.name}>
                    <ValueRow
                      feature={f}
                      state={values[f.name] ?? { value: null, status: "blank" }}
                      extracted={extractedByName.get(f.name)}
                      open={editing === f.name}
                      onToggle={() => setEditing(editing === f.name ? null : f.name)}
                    />
                    {editing === f.name && (
                      <div className="bg-hairline px-6 py-4">
                        <FieldInput
                          feature={f}
                          state={values[f.name] ?? { value: null, status: "blank" }}
                          onChange={(next) => update(f.name, next)}
                          compact
                        />
                        <button
                          className="btn-secondary mt-4 py-2 text-xs"
                          onClick={() => setEditing(null)}
                        >
                          Done
                        </button>
                      </div>
                    )}
                  </Fragment>
                ))}
              </dl>
            </section>
          );
        })}
      </div>

      <div className="mt-10 border-t border-line pt-8">
        {unsettled.length > 0 ? (
          <div className="max-w-prose">
            <Note tone="attention">
              <p className="font-medium text-ink">
                {unsettled.length} {unsettled.length === 1 ? "item still needs" : "items still need"}{" "}
                your attention
              </p>
              <p className="mt-1">
                Add a value, or say you don't have it. Both are fine — but the analysis
                shouldn't guess on your behalf.
              </p>
              <button className="btn-secondary mt-4 py-2 text-xs" onClick={markRemainingUnavailable}>
                I don't have any of these
              </button>
            </Note>
          </div>
        ) : (
          <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center">
            <button className="btn-primary" disabled={busy} onClick={run}>
              {busy ? "Analyzing…" : "Everything looks correct — Analyze →"}
            </button>
            <p className="text-sm text-muted">Nothing has been analyzed yet.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function ValueRow({
  feature,
  state,
  extracted,
  open,
  onToggle,
}: {
  feature: FeatureSpec;
  state: FieldState;
  extracted?: ExtractedField;
  open: boolean;
  onToggle: () => void;
}) {
  const outOfRange = outOfObservedRange(feature, state.value);
  const status = statusOf(state, outOfRange);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-6 py-3.5">
      <dt className="min-w-0 flex-1 text-sm text-body">{feature.label}</dt>

      <dd className="tnum text-sm font-medium text-ink">
        {state.status === "blank" ? (
          <span className="font-normal text-faint">—</span>
        ) : state.status === "unavailable" ? (
          <span className="font-normal text-muted">Not available</span>
        ) : (
          <>
            {formatValue(feature, state.value)}
            {feature.unit && state.value !== null && (
              <span className="ml-1 text-xs font-normal text-faint">{feature.unit}</span>
            )}
          </>
        )}
      </dd>

      <span className={`pill w-32 justify-center ${status.className}`}>{status.label}</span>

      <div className="flex w-24 justify-end gap-3">
        {extracted?.page && state.status !== "blank" && (
          <details className="group relative">
            <summary className="cursor-pointer list-none text-xs text-faint hover:text-muted">
              Source
            </summary>
            <div className="absolute right-0 z-10 mt-2 w-72 rounded-lg border border-line bg-surface p-3 text-xs shadow-lift">
              <p className="text-muted">Extracted from page {extracted.page}</p>
              <p className="mt-1.5 break-words font-mono text-2xs text-body">
                “{extracted.raw_text}”
              </p>
              {extracted.note && <p className="mt-2 text-muted">{extracted.note}</p>}
            </div>
          </details>
        )}
        <button className="text-xs text-accent underline-offset-4 hover:underline" onClick={onToggle}>
          {open ? "Close" : "Edit"}
        </button>
      </div>
    </div>
  );
}

function statusOf(state: FieldState, outOfRange: boolean): { label: string; className: string } {
  if (state.status === "blank")
    return { label: "⚠ Needs review", className: "bg-attention-soft text-attention" };
  if (state.status === "flagged")
    return { label: "⚠ Please verify", className: "bg-attention-soft text-attention" };
  if (outOfRange)
    return { label: "⚠ Check units", className: "bg-caution-soft text-caution" };
  if (state.status === "unavailable")
    return { label: "Not available", className: "bg-hairline text-muted" };
  if (state.status === "extracted")
    return { label: "✓ Found", className: "bg-steady-soft text-steady" };
  return { label: "✓ Confirmed", className: "bg-steady-soft text-steady" };
}
