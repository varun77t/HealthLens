import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api";
import { ErrorBox, Note, PageHead, Spinner } from "../components/Chrome";
import { UploadDropzone } from "../components/UploadDropzone";
import { emptyIntake, fromExtraction } from "../lib/intake";
import { useSession } from "../state";
import type { Disease, DemoReportsResponse, SchemaResponse } from "../types";

const HEADINGS: Record<string, string> = {
  heart: "Heart Health",
  kidney: "Kidney Health",
  diabetes: "Diabetes",
};

/**
 * The assessment entry screen: upload first, manual entry as the fallback.
 *
 * The promise made here is deliberately modest — "the information available in your
 * report", not "your report". Real documents do not contain every field a model uses, and
 * a screen that implies otherwise sets up the review step to look like a failure rather
 * than the point.
 */
export default function Start() {
  const { disease } = useParams<{ disease: Disease }>();
  const nav = useNavigate();
  const session = useSession();

  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [demos, setDemos] = useState<DemoReportsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!disease) return;
    setError(null);
    Promise.all([api.schema(disease), api.demoReports(disease).catch(() => null)])
      .then(([s, d]) => {
        setSchema(s);
        setDemos(d);
      })
      .catch((e) => setError(e.message));
  }, [disease]);

  if (error) return <ErrorBox error={error} />;
  if (!schema || !disease) return <Spinner label="Getting things ready…" />;

  const upload = async (file: File) => {
    setBusy(true);
    setUploadError(null);
    try {
      const result = await api.extract(disease, file);
      session.start(disease, schema, fromExtraction(schema, result));
      session.setExtraction(result, file.name);
      nav(`/app/${disease}/review`);
    } catch (e) {
      setUploadError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const manual = () => {
    session.start(disease, schema, emptyIntake(schema));
    nav(`/app/${disease}/enter`);
  };

  return (
    <div>
      <PageHead
        eyebrow={HEADINGS[disease] ?? schema.module}
        title="Let's get your information"
        lead="Upload a medical report and we'll extract the information we can use."
        back={{ to: "/app", label: "All assessments" }}
      />

      <UploadDropzone onFile={upload} busy={busy} />

      {uploadError && (
        <div className="mt-4">
          <Note tone="caution">{uploadError}</Note>
        </div>
      )}

      <p className="mt-5 max-w-prose text-sm text-muted">
        We'll extract the information available in your report. You'll be able to review and
        complete anything that's missing before we analyze it. Your document is read once and
        never stored.
      </p>

      <div className="mt-10 flex flex-col gap-2 border-t border-line pt-8 sm:flex-row sm:items-baseline sm:gap-3">
        <span className="text-sm text-muted">Don't have a report?</span>
        <button className="btn-link" onClick={manual}>
          Enter information manually →
        </button>
      </div>

      {demos && demos.cases.some((c) => c.available) && (
        <div className="mt-12">
          <p className="eyebrow mb-3">Trying this out</p>
          <div className="surface p-6">
            <p className="max-w-prose text-sm text-body">
              Download a sample report to see the whole flow. These documents are fabricated
              and say so on their face, but the values inside them come from real records in
              the held-out test split — data this model was never trained on — so the
              analysis at the end is genuine.
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              {demos.cases
                .filter((c) => c.available)
                .map((c) => (
                  <a
                    key={c.id}
                    href={api.demoReportUrl(disease, c.id)}
                    className="btn-secondary py-2 text-xs"
                    download={c.filename}
                  >
                    ↓ Sample report{" "}
                    {c.id.endsWith("1") ? "" : "(2)"}
                    {c.recorded_label === 1 ? " · A" : " · B"}
                  </a>
                ))}
            </div>
            <p className="mt-4 text-xs text-faint">
              The recorded outcome is deliberately not shown here — download one, run it
              through, and compare afterwards.
            </p>
          </div>
        </div>
      )}

      <div className="mt-12">
        <Link to={`/app/${disease}/about`} className="btn-link text-xs">
          How is this estimated? Model, data and limitations →
        </Link>
      </div>
    </div>
  );
}
