import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { ErrorBox, Note, PageHead, Spinner } from "../components/Chrome";
import { FieldInput } from "../components/FieldInput";
import { byGroup, emptyIntake, hasValue } from "../lib/intake";
import { useSession } from "../state";
import type { Disease, FeatureSpec, FieldState, SchemaResponse } from "../types";

/**
 * Manual entry — the fallback when there is no report to upload.
 *
 * Sections open one at a time and are ordered the way a person holds the information: about
 * you, then vitals, then each test in turn. The importance tiering still runs underneath
 * (fields are ordered within a section by how much the model leans on them) but it is no
 * longer surfaced as badges and percentages — that was the "filling in a dataset" feel.
 */
export default function Intake() {
  const { disease } = useParams<{ disease: Disease }>();
  const nav = useNavigate();
  const session = useSession();
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    if (!disease) return;
    api
      .schema(disease)
      .then((s) => {
        setSchema(s);
        if (session.disease !== disease || !session.schema) {
          session.start(disease, s, emptyIntake(s));
        }
        setOpen(s.groups[0]?.id ?? null);
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

  const anyValue = schema.features.some((f) => values[f.name] && hasValue(values[f.name]));
  const update = (name: string, next: FieldState) =>
    session.setValues({ ...values, [name]: next });

  return (
    <div>
      <PageHead
        title="Enter your information"
        lead="Fill in what you have. Anything you leave out will be noted in the analysis rather than guessed."
        back={{ to: `/${disease}`, label: "Back" }}
      />

      <div className="space-y-3">
        {schema.groups.map((g) => {
          const fields = groups.get(g.id) ?? [];
          if (!fields.length) return null;
          const done = fields.filter((f) => values[f.name] && hasValue(values[f.name])).length;
          const isOpen = open === g.id;
          return (
            <section key={g.id} className="surface overflow-hidden">
              <button
                className="flex w-full items-center justify-between gap-4 px-6 py-5 text-left hover:bg-hairline"
                onClick={() => setOpen(isOpen ? null : g.id)}
                aria-expanded={isOpen}
              >
                <span>
                  <span className="text-base font-medium text-ink">{g.label}</span>
                  <span className="mt-1 block text-sm text-muted">{g.help}</span>
                </span>
                <span className="flex shrink-0 items-center gap-3 text-xs text-faint">
                  {done > 0 && (
                    <span className="tnum">
                      {done}/{fields.length}
                    </span>
                  )}
                  <span className={isOpen ? "rotate-180 transition-transform" : "transition-transform"}>
                    ▾
                  </span>
                </span>
              </button>
              {isOpen && (
                <div className="divide-y divide-hairline border-t border-line px-6">
                  {fields.map((f) => (
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

      <div className="mt-10 flex flex-col items-start gap-3 border-t border-line pt-8 sm:flex-row sm:items-center">
        <button
          className="btn-primary"
          disabled={!anyValue}
          onClick={() => nav(`/${disease}/review`)}
        >
          Review your information
        </button>
        {!anyValue && (
          <p className="text-sm text-muted">Enter at least one value to continue.</p>
        )}
      </div>

      <div className="mt-10">
        <Note>{schema.range_note}</Note>
      </div>
    </div>
  );
}
