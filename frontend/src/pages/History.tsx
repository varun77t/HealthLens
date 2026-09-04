import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { ErrorBox, Note, PageHead, Spinner } from "../components/Chrome";
import * as routes from "../lib/routes";
import type { AnalysisPage, AnalysisSummary, Disease } from "../types";

const PAGE = 10;

const MODULE_NAMES: Record<Disease, string> = {
  heart: "Heart Health",
  kidney: "Kidney Health",
  diabetes: "Diabetes",
};

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/**
 * A saved entry, shown with the threshold that defined its category.
 *
 * The percentage never appears without it. "14.5%" against diabetes's 13.9% threshold means
 * the model flagged the case; read against an assumed 50% it means the opposite, and a
 * history list is exactly where someone skims numbers out of context.
 */
function Row({ item, onDelete }: { item: AnalysisSummary; onDelete: (id: string) => void }) {
  const [confirming, setConfirming] = useState(false);

  return (
    <li className="surface flex flex-wrap items-center gap-x-6 gap-y-3 p-5">
      <div className="min-w-0 flex-1">
        <Link to={routes.historyItem(item.id)} className="block">
          <p className="text-sm font-medium text-ink">
            {item.label || MODULE_NAMES[item.disease]}
            {item.label && (
              <span className="ml-2 font-normal text-muted">{MODULE_NAMES[item.disease]}</span>
            )}
          </p>
          <p className="mt-1 text-xs text-faint">
            {when(item.created_at)}
            {item.source_document && ` · ${item.source_document}`}
          </p>
        </Link>
      </div>

      <div className="text-right">
        <p className="text-sm font-semibold capitalize text-ink">{item.band_label}</p>
        <p className="tnum text-xs text-muted">
          {(item.probability * 100).toFixed(1)}% · threshold {(item.threshold * 100).toFixed(1)}%
        </p>
      </div>

      <div className="flex items-center gap-2">
        {!item.model_is_current && (
          <span className="pill bg-attention-soft text-body" title="This model has changed since">
            Earlier model
          </span>
        )}
        {confirming ? (
          <>
            <button className="btn-quiet text-xs" onClick={() => onDelete(item.id)}>
              Delete
            </button>
            <button className="btn-quiet text-xs" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </>
        ) : (
          <button className="btn-quiet text-xs" onClick={() => setConfirming(true)}>
            Remove
          </button>
        )}
      </div>
    </li>
  );
}

export default function History() {
  const [page, setPage] = useState<AnalysisPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<Disease | "">("");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    api.analyses
      .list({ limit: PAGE, offset, disease: filter || undefined })
      .then(setPage)
      .catch((e) => setError(e.message));
  }, [offset, filter]);

  useEffect(load, [load]);

  async function remove(id: string) {
    try {
      await api.analyses.remove(id);
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete that entry.");
    }
  }

  if (error) return <ErrorBox error={error} onRetry={load} />;
  if (!page) return <Spinner label="Loading your history…" />;

  const lastPage = offset + PAGE >= page.total;

  return (
    <>
      <PageHead
        eyebrow="History"
        title="Your saved analyses"
        lead="Only what you chose to save. Assessments you ran without saving were never recorded."
        back={{ to: routes.home, label: "Assessments" }}
      />

      <div className="mb-6 flex flex-wrap items-center gap-2">
        {(["", "heart", "kidney", "diabetes"] as const).map((d) => (
          <button
            key={d || "all"}
            className={`pill border ${
              filter === d ? "border-accent-line bg-accent/10 text-ink" : "border-line text-muted"
            }`}
            onClick={() => {
              setFilter(d);
              setOffset(0);
            }}
          >
            {d ? MODULE_NAMES[d] : "All"}
          </button>
        ))}
      </div>

      {page.items.length === 0 ? (
        <Note>
          <p className="font-medium text-ink">Nothing saved yet</p>
          <p className="mt-1">
            Run an assessment and choose <em>Save to my history</em> on the result screen.
            Nothing is stored unless you ask for it.
          </p>
          <p className="mt-3">
            <Link to={routes.home} className="btn-link">
              Start an assessment →
            </Link>
          </p>
        </Note>
      ) : (
        <>
          <ul className="space-y-3">
            {page.items.map((item) => (
              <Row key={item.id} item={item} onDelete={remove} />
            ))}
          </ul>

          {page.total > PAGE && (
            <div className="mt-6 flex items-center gap-3 text-sm">
              <button
                className="btn-secondary py-2 text-xs"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE))}
              >
                ← Newer
              </button>
              <span className="text-muted">
                {offset + 1}–{Math.min(offset + PAGE, page.total)} of {page.total}
              </span>
              <button
                className="btn-secondary py-2 text-xs"
                disabled={lastPage}
                onClick={() => setOffset(offset + PAGE)}
              >
                Older →
              </button>
            </div>
          )}
        </>
      )}

      <p className="mt-10 max-w-prose border-t border-line pt-5 text-xs text-faint">
        {page.note}
      </p>
    </>
  );
}
