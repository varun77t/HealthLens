import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Banner, ErrorBox, Spinner } from "../components/Chrome";
import type { ModelSummary } from "../types";

const BLURB: Record<string, string> = {
  heart:
    "Needs results from a coronary angiogram and a thallium scan to answer well — those are its strongest inputs. Without them the estimate is reported as reduced-coverage.",
  kidney:
    "Answerable from a routine urinalysis and full blood count. The best fit of the three for someone holding a recent lab report.",
  diabetes:
    "No lab report needed. It is built from survey answers, so about eight short questions cover most of what it uses.",
};

export default function Dashboard() {
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const nav = useNavigate();

  const load = () => {
    setError(null);
    setModels(null);
    api
      .models()
      .then(setModels)
      .catch((e) => setError(e.message));
  };
  useEffect(load, []);

  if (error) return <ErrorBox error={error} onRetry={load} />;
  if (!models) return <Spinner label="Loading model cards…" />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">What would you like to check?</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          Pick one module. Each is a separate model trained on a separate public dataset, so
          you will only be asked for what that model actually uses — and never for the other
          two.
        </p>
      </div>

      <Banner tone="warn" title="Read this before you start">
        This is a research and education tool, not a medical device. It estimates how closely
        your details resemble records in a historical dataset. It cannot diagnose anything,
        cannot tell you what will happen in future, and is not a substitute for a clinician.
      </Banner>

      <div className="grid gap-4 md:grid-cols-3">
        {models.map((m) => (
          <button
            key={m.disease}
            onClick={() => nav(`/${m.disease}`)}
            className="card group flex flex-col p-5 text-left transition-shadow hover:shadow-md"
          >
            <h2 className="text-base font-semibold leading-snug">{m.module}</h2>
            <p className="mt-2 flex-1 text-xs leading-relaxed text-muted">
              {BLURB[m.disease]}
            </p>
            <dl className="mt-4 space-y-1 border-t border-line pt-3 text-xs">
              <div className="flex justify-between">
                <dt className="text-muted">Test ROC-AUC</dt>
                <dd className="tnum font-medium">{m.roc_auc_test.toFixed(4)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">Trained on</dt>
                <dd className="tnum font-medium">{m.n_train.toLocaleString()} records</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">External validation</dt>
                <dd className="font-medium text-warn">
                  {m.external_validation.status === "rejected" ? "rejected" : "none"}
                </dd>
              </div>
            </dl>
            <span className="btn-primary mt-4 w-full group-hover:bg-accent/90">Start</span>
          </button>
        ))}
      </div>

      <Banner title="No module here has been externally validated">
        Every figure above comes from a single held-out split of a single dataset. The heart
        module's intended external cohort was tested and rejected — it turned out to be a
        redistributed subset of its own training data. Nothing establishes that any of these
        models transfers to another hospital, country or population.
      </Banner>
    </div>
  );
}
