import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Note, TextField } from "./Chrome";
import * as routes from "../lib/routes";
import type { Disease, SourceKind } from "../types";

/**
 * Saving is a separate, explicit act.
 *
 * Nothing about running an assessment writes to the database — signing in unlocked the
 * flow, it did not start a record. This is the only control in the application that stores
 * health information, so it says what it keeps *before* it is pressed rather than in a
 * policy page afterwards.
 *
 * The inputs are what get sent. The server re-runs the model and stores its own answer, so
 * a saved entry is a result the model produced, not a number this page reported.
 */
export function SaveAnalysis({
  disease,
  features,
  sourceKind,
  sourceDocument,
}: {
  disease: Disease;
  features: Record<string, number | null>;
  sourceKind: SourceKind;
  sourceDocument: string | null;
}) {
  const [open, setOpen] = useState(false);
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    setBusy(true);
    setError(null);
    try {
      const saved = await api.analyses.save({
        disease,
        features,
        source_kind: sourceKind,
        source_document: sourceDocument,
        label: label.trim() || null,
      });
      setSavedId(saved.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save this analysis.");
    } finally {
      setBusy(false);
    }
  }

  if (savedId) {
    return (
      <Note tone="attention">
        <p className="font-medium text-ink">Saved to your history</p>
        <p className="mt-1">
          <Link to={routes.historyItem(savedId)} className="btn-link">
            Open it
          </Link>{" "}
          ·{" "}
          <Link to={routes.history} className="btn-link">
            All saved analyses
          </Link>
        </p>
      </Note>
    );
  }

  if (!open) {
    return (
      <button className="btn-secondary" onClick={() => setOpen(true)}>
        Save to my history
      </button>
    );
  }

  return (
    <div className="surface w-full max-w-lg p-6">
      <h3 className="text-sm font-semibold text-ink">Save this analysis</h3>
      <p className="mt-2 text-sm text-body">
        This stores the values you entered, the estimate, and the model version and threshold
        that produced it — so the result stays readable later even if the model changes.
        The uploaded document itself is not stored. You can delete this entry, or your whole
        account, at any time.
      </p>

      <div className="mt-5">
        <TextField
          label="Label"
          value={label}
          onChange={setLabel}
          placeholder="Optional — e.g. 'March check-up'"
        />
      </div>

      {error && (
        <div className="mt-4">
          <Note tone="caution">{error}</Note>
        </div>
      )}

      <div className="mt-5 flex flex-wrap gap-3">
        <button className="btn-primary" onClick={save} disabled={busy}>
          {busy ? "Saving…" : "Save to my history"}
        </button>
        <button className="btn-quiet" onClick={() => setOpen(false)} disabled={busy}>
          Not now
        </button>
      </div>
    </div>
  );
}
