import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";
import { Disclosure, ErrorBox, Spinner } from "../components/Chrome";
import type { Disease, ModelSummary } from "../types";

/**
 * Module selection.
 *
 * The research figures — ROC-AUC, training counts, external-validation status — are not
 * gone; they moved into the disclosure at the foot of the page. On the primary journey they
 * were answering a question nobody had yet asked, and they made a health tool read like a
 * benchmark table.
 */

const MODULES: { disease: Disease; icon: string; name: string; blurb: string }[] = [
  {
    disease: "heart",
    icon: "❤️",
    name: "Heart Health",
    blurb: "Explore factors associated with heart disease risk.",
  },
  {
    disease: "kidney",
    icon: "🫘",
    name: "Kidney Health",
    blurb: "Explore factors associated with chronic kidney disease risk.",
  },
  {
    disease: "diabetes",
    icon: "🩸",
    name: "Diabetes",
    blurb: "Explore factors associated with diabetes-related health risk.",
  },
];

export default function Dashboard() {
  const { user } = useAuth();
  const [models, setModels] = useState<ModelSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setError(null);
    setModels(null);
    api.models().then(setModels).catch((e) => setError(e.message));
  };
  useEffect(load, []);

  if (error) return <ErrorBox error={error} onRetry={load} />;
  if (!models) return <Spinner label="Getting things ready…" />;

  const available = new Set(models.map((m) => m.disease));

  return (
    <div>
      <div className="mb-12 max-w-prose">
        <p className="eyebrow mb-2">Welcome back{user ? `, ${user.greeting_name}` : ""}</p>
        <h1 className="text-3xl font-semibold">What would you like to check?</h1>
        <p className="mt-3 text-lg text-body">
          Each assessment is a separate model with its own dataset and its own limits. They
          are never combined, and there is no overall score.
        </p>
      </div>

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {MODULES.filter((m) => available.has(m.disease)).map((m) => (
          <Link
            key={m.disease}
            to={`/app/${m.disease}`}
            className="surface group flex flex-col p-7 transition-all duration-200 hover:-translate-y-0.5 hover:border-accent-line hover:shadow-lift"
          >
            <span aria-hidden className="text-2xl">
              {m.icon}
            </span>
            <h2 className="mt-4 text-lg font-semibold">{m.name}</h2>
            <p className="mt-2 flex-1 text-sm text-muted">{m.blurb}</p>
            <span className="mt-6 inline-flex items-center gap-1.5 text-sm font-medium text-accent">
              Start assessment
              <span className="transition-transform group-hover:translate-x-0.5">→</span>
            </span>
          </Link>
        ))}
      </div>

      <div className="mt-14">
        <Disclosure summary="How these estimates are produced">
          <div className="space-y-6">
            <p className="max-w-prose text-sm text-body">
              Three separate machine-learning models, each trained on its own public research
              dataset. They are never combined, and this platform produces no overall health
              score. Every figure below comes from an actual training run.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-line text-left text-xs text-muted">
                  <tr>
                    <th scope="col" className="pb-2 pr-4 font-medium">Module</th>
                    <th scope="col" className="pb-2 pr-4 font-medium">Model</th>
                    <th scope="col" className="pb-2 pr-4 font-medium">Records</th>
                    <th scope="col" className="pb-2 pr-4 font-medium">Test ROC-AUC</th>
                    <th scope="col" className="pb-2 font-medium">External validation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hairline">
                  {models.map((m) => (
                    <tr key={m.disease}>
                      <td className="py-2.5 pr-4 text-ink">{m.module}</td>
                      <td className="py-2.5 pr-4">{m.model_name}</td>
                      <td className="tnum py-2.5 pr-4">
                        {(m.n_train + m.n_test).toLocaleString()}
                      </td>
                      <td className="tnum py-2.5 pr-4">{m.roc_auc_test.toFixed(4)}</td>
                      <td className="py-2.5 text-attention">
                        {m.external_validation.status === "rejected"
                          ? "attempted, rejected"
                          : "none"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="max-w-prose space-y-3 text-sm text-body">
              <p>
                <span className="font-medium text-ink">
                  No module has been externally validated.
                </span>{" "}
                Every figure above comes from a single held-out split of a single dataset.
                The heart module's intended external cohort was tested and rejected — it
                turned out to be a redistributed subset of its own training data. Nothing
                establishes that these models transfer to another hospital, country or
                population.
              </p>
              <p>
                These are experimental research models. They estimate how closely the
                information you provide resembles records in a historical dataset. They
                cannot diagnose anything, cannot tell you what will happen in future, and are
                not a substitute for a clinician.
              </p>
            </div>

            <div className="flex flex-wrap gap-x-6 gap-y-2 border-t border-line pt-4 text-xs">
              {models.map((m) => (
                <Link key={m.disease} to={`/app/${m.disease}/about`} className="btn-link text-xs">
                  {m.module} — full model card
                </Link>
              ))}
            </div>
          </div>
        </Disclosure>
      </div>
    </div>
  );
}
