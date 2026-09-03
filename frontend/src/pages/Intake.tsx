import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { Banner, ErrorBox, Spinner } from "../components/Chrome";
import { CoverageMeter } from "../components/CoverageMeter";
import { FieldInput } from "../components/FieldInput";
import {
  byGroup,
  coverage,
  emptyIntake,
  fromSample,
  hasValue,
  unansweredCount,
} from "../lib/intake";
import { useSession } from "../state";
import type {
  Disease,
  FeatureSpec,
  FieldState,
  SamplesResponse,
  SchemaResponse,
} from "../types";

export default function Intake() {
  const { disease } = useParams<{ disease: Disease }>();
  const nav = useNavigate();
  const session = useSession();
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [samples, setSamples] = useState<SamplesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [coreOnly, setCoreOnly] = useState(true);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!disease) return;
    setError(null);
    Promise.all([api.schema(disease), api.samples(disease).catch(() => null)])
      .then(([s, sm]) => {
        setSchema(s);
        setSamples(sm);
        // Preserve values if the user navigates back from review.
        if (session.disease !== disease || !session.schema) {
          session.start(disease, s, emptyIntake(s));
        }
        // Open the first group that is actually visible. The form starts in "key fields
        // only" mode, and for kidney the first group by order (About you) contains no key
        // field — opening it by index would leave every visible section collapsed.
        const firstVisible = s.groups.find((g) =>
          s.features.some((f) => f.group === g.id && f.tier === "core"),
        );
        setOpenGroups(
          Object.fromEntries(s.groups.map((g) => [g.id, g.id === firstVisible?.id])),
        );
      })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disease]);

  const values = session.values;
  const groups = useMemo(
    () => (schema ? byGroup(schema) : new Map<string, FeatureSpec[]>()),
    [schema],
  );

  if (error) return <ErrorBox error={error} />;
  if (!schema || !disease || !session.schema) return <Spinner label="Loading the form…" />;

  const core = schema.features.filter((f) => f.tier === "core");
  const coreDone = core.filter((f) => values[f.name] && hasValue(values[f.name])).length;
  const covered = coverage(schema, values);
  const unanswered = unansweredCount(schema, values);
  const anyValue = schema.features.some((f) => values[f.name] && hasValue(values[f.name]));

  const update = (name: string, next: FieldState) =>
    session.setValues({ ...values, [name]: next });

  const loadSample = (id: string) => {
    const c = samples?.cases.find((x) => x.id === id);
    if (!c || !schema) return;
    session.setValues(fromSample(schema, c.features));
    session.setSampleId(id);
    setCoreOnly(false);
    setOpenGroups(Object.fromEntries(schema.groups.map((g) => [g.id, true])));
  };

  return (
    <div className="space-y-6">
      <div>
        <button className="btn-ghost -ml-2 mb-2 text-xs" onClick={() => nav("/")}>
          ← All modules
        </button>
        <h1 className="text-2xl font-semibold tracking-tight">{schema.module}</h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
          Fill in what you have. Every field is optional — anything you leave out is filled
          from the training data, and the result will tell you which ones. Nothing you type
          leaves your browser except the prediction request.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-[2fr_1fr] md:items-start">
        <div className="space-y-4 md:order-1">
          {schema.groups.map((g) => {
            const all = groups.get(g.id) ?? [];
            const shown = coreOnly ? all.filter((f) => f.tier === "core") : all;
            if (!shown.length) return null;
            const done = all.filter((f) => values[f.name] && hasValue(values[f.name])).length;
            const open = openGroups[g.id] ?? false;
            return (
              <section key={g.id} className="card">
                <button
                  className="flex w-full items-start justify-between gap-4 p-4 text-left"
                  onClick={() => setOpenGroups({ ...openGroups, [g.id]: !open })}
                  aria-expanded={open}
                >
                  <span>
                    <span className="text-sm font-semibold">{g.label}</span>
                    <span className="mt-1 block text-xs leading-relaxed text-muted">
                      {g.help}
                    </span>
                  </span>
                  <span className="tnum shrink-0 text-xs text-muted">
                    {done}/{all.length} {open ? "▲" : "▼"}
                  </span>
                </button>
                {open && (
                  <div className="divide-y divide-line border-t border-line px-4">
                    {shown.map((f) => (
                      <FieldInput
                        key={f.name}
                        feature={f}
                        state={values[f.name] ?? { value: null, status: "blank" }}
                        onChange={(next) => update(f.name, next)}
                      />
                    ))}
                  </div>
                )}
              </section>
            );
          })}
        </div>

        <aside className="space-y-4 md:sticky md:top-4 md:order-2">
          <CoverageMeter covered={covered} coreDone={coreDone} coreTotal={core.length} />

          <div className="card p-4">
            <label className="flex cursor-pointer items-start gap-3">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={coreOnly}
                onChange={(e) => setCoreOnly(e.target.checked)}
              />
              <span className="text-xs leading-relaxed">
                <span className="block text-sm font-medium">Show key fields only</span>
                <span className="text-muted">
                  {core.length} of {schema.features.length} fields carry about 80% of what
                  this model uses. The rest still help — they are just worth less each.
                </span>
              </span>
            </label>
          </div>

          {samples && samples.cases.length > 0 && (
            <div className="card p-4">
              <p className="text-sm font-medium">Try an example</p>
              <p className="mt-1 text-xs leading-relaxed text-muted">
                Real records from the held-out test split — rows this model was never trained
                on, each with the outcome recorded in the source dataset.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {samples.cases.map((c) => (
                  <button
                    key={c.id}
                    className="btn-secondary px-3 py-1.5 text-xs"
                    onClick={() => loadSample(c.id)}
                  >
                    {c.recorded_label === 1 ? "Recorded positive" : "Recorded negative"}
                  </button>
                ))}
              </div>
              {session.sampleId && (
                <p className="mt-3 text-xs text-accent">
                  Loaded {session.sampleId}. These are dataset rows, not patients of this
                  system.
                </p>
              )}
            </div>
          )}

          <button
            className="btn-primary w-full"
            disabled={!anyValue}
            onClick={() => nav(`/${disease}/review`)}
          >
            Review your information
          </button>
          {!anyValue && (
            <p className="text-center text-xs text-muted">
              Enter at least one value to continue.
            </p>
          )}
          {anyValue && unanswered > 0 && (
            <p className="text-center text-xs text-muted">
              {unanswered} field{unanswered === 1 ? "" : "s"} still untouched — you can settle
              them on the next screen.
            </p>
          )}
        </aside>
      </div>

      <Banner title="Where these ranges come from">
        {schema.range_note}
      </Banner>
    </div>
  );
}
