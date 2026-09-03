import { Fragment, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api";
import { Banner, ErrorBox } from "../components/Chrome";
import { CoverageMeter } from "../components/CoverageMeter";
import { FieldInput } from "../components/FieldInput";
import {
  coverage,
  formatValue,
  hasValue,
  outOfObservedRange,
  toPayload,
} from "../lib/intake";
import { useSession } from "../state";
import type { Disease, FieldState } from "../types";

/**
 * The verification step. Nothing is predicted until the user has settled every field.
 *
 * Fields are ordered by how much the model actually leans on them, not by the order they
 * were entered, so attention goes where a mistake would matter most. An untouched field
 * blocks the run: leaving it blank and marking it unavailable both reach the model as
 * missing, but only one of them is a decision the user made.
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

  const rows = useMemo(
    () => (schema ? [...schema.features].sort((a, b) => a.shap_rank - b.shap_rank) : []),
    [schema],
  );

  if (!schema || !disease || session.disease !== disease) {
    nav(`/${disease ?? ""}`, { replace: true });
    return null;
  }

  const blanks = rows.filter((f) => (values[f.name]?.status ?? "blank") === "blank");
  const core = schema.features.filter((f) => f.tier === "core");
  const coreDone = core.filter((f) => values[f.name] && hasValue(values[f.name])).length;
  const covered = coverage(schema, values);

  const update = (name: string, next: FieldState) =>
    session.setValues({ ...values, [name]: next });

  const markRemainingUnavailable = () => {
    const next = { ...values };
    for (const f of blanks) next[f.name] = { value: null, status: "unavailable" };
    session.setValues(next);
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const prediction = await api.predict(disease, toPayload(values));
      session.setPrediction(prediction);
      session.setConfirmed(true);
      nav(`/${disease}/result`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <button className="btn-ghost -ml-2 mb-2 text-xs" onClick={() => nav(`/${disease}`)}>
          ← Back to the form
        </button>
        <h1 className="text-2xl font-semibold tracking-tight">Review your information</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          Check every line before anything is analysed. Fields are listed by how much this
          model relies on them, so the ones at the top are where a wrong value would change
          the answer most.
        </p>
      </div>

      {error && <ErrorBox error={error} />}

      <div className="grid gap-4 md:grid-cols-[2fr_1fr] md:items-start">
        <div className="card overflow-hidden md:order-1">
          <table className="w-full text-sm">
            <caption className="sr-only">Extracted and entered values for review</caption>
            <thead className="border-b border-line bg-canvas text-left text-xs text-muted">
              <tr>
                <th scope="col" className="px-4 py-2 font-medium">Field</th>
                <th scope="col" className="px-4 py-2 font-medium">Value</th>
                <th scope="col" className="px-4 py-2 font-medium">Status</th>
                <th scope="col" className="px-4 py-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {rows.map((f) => {
                const st = values[f.name] ?? { value: null, status: "blank" as const };
                const open = editing === f.name;
                const extrapolated = outOfObservedRange(f, st.value);
                return (
                  <Fragment key={f.name}>
                    <tr className={open ? "bg-canvas" : undefined}>
                      <td className="px-4 py-2.5">
                        <span className="font-medium">{f.label}</span>
                        {f.tier === "core" && (
                          <span className="ml-2 chip bg-accent/10 text-accent">key</span>
                        )}
                      </td>
                      <td className="tnum px-4 py-2.5">
                        {st.status === "blank" ? (
                          <span className="text-muted">—</span>
                        ) : st.status === "unavailable" ? (
                          <span className="text-muted">Unavailable</span>
                        ) : (
                          formatValue(f, st.value)
                        )}
                      </td>
                      <td className="px-4 py-2.5">
                        {st.status === "blank" ? (
                          <span className="chip bg-warnBg text-warn">⚠ Not answered</span>
                        ) : st.status === "unavailable" ? (
                          <span className="chip bg-canvas text-muted">— Unavailable</span>
                        ) : extrapolated ? (
                          <span className="chip bg-warnBg text-warn">⚠ Out of range</span>
                        ) : (
                          <span className="chip bg-clearBg text-clear">✓ Confirmed</span>
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <button
                          className="text-xs text-accent underline-offset-2 hover:underline"
                          onClick={() => setEditing(open ? null : f.name)}
                        >
                          {open ? "Done" : "Edit"}
                        </button>
                      </td>
                    </tr>
                    {open && (
                      <tr className="bg-canvas">
                        <td colSpan={4} className="px-4 pb-3">
                          <FieldInput
                            feature={f}
                            state={st}
                            onChange={(next) => update(f.name, next)}
                          />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        <aside className="space-y-4 md:sticky md:top-4 md:order-2">
          <CoverageMeter covered={covered} coreDone={coreDone} coreTotal={core.length} />

          {blanks.length > 0 ? (
            <Banner tone="warn" title={`${blanks.length} field${blanks.length === 1 ? "" : "s"} still unanswered`}>
              <p>
                Leaving a field blank and saying you do not have it both reach the model as
                missing — but only one of them is a decision you made. Settle each one before
                running.
              </p>
              <button className="btn-secondary mt-3 w-full" onClick={markRemainingUnavailable}>
                I don&apos;t have any of these
              </button>
            </Banner>
          ) : (
            <Banner tone="info" title="Everything is settled">
              You can run the analysis. Values marked unavailable will be filled from the
              training data, and the result will list exactly which ones.
            </Banner>
          )}

          <button
            className="btn-primary w-full"
            disabled={blanks.length > 0 || busy}
            onClick={run}
          >
            {busy ? "Analysing…" : "Confirm and run analysis"}
          </button>
          <p className="text-center text-xs text-muted">
            Nothing has been predicted yet.
          </p>
        </aside>
      </div>
    </div>
  );
}
