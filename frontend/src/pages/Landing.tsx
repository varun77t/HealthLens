import { Link } from "react-router-dom";
import { PublicLayout } from "../components/Chrome";
import * as routes from "../lib/routes";

/**
 * The first screen anyone sees.
 *
 * It says what the tool is, what it is not, and offers exactly one thing to do. There is
 * no module list here and no demonstration: an account is required before any health
 * information is submitted, so showing the assessments before sign-in would advertise a
 * door that is locked.
 */
export default function Landing() {
  return (
    <PublicLayout>
      <section className="pb-4 pt-6">
        <p className="eyebrow mb-3">Research &amp; education platform</p>
        <h1 className="max-w-2xl text-4xl font-semibold leading-tight tracking-tight">
          Understand what a model sees in your health data — and why.
        </h1>
        <p className="mt-5 max-w-prose text-lg text-body">
          Three independent risk models — heart, kidney and diabetes indicators — each with
          its own dataset, its own limitations, and a plain-language account of which of
          your values moved its estimate.
        </p>

        <div className="mt-9 flex flex-wrap items-center gap-3">
          <Link to={routes.SIGN_IN} className="btn-primary">
            Sign in
          </Link>
          <Link to={routes.SIGN_UP} className="btn-secondary">
            Create an account
          </Link>
        </div>
        <p className="mt-4 text-sm text-muted">
          An account is required. Nothing is analysed before you have signed in and reviewed
          the information yourself.
        </p>
      </section>

      <section className="mt-16 grid gap-5 sm:grid-cols-3">
        {[
          {
            title: "Upload a report",
            body: "Values are read from your document by pattern matching against the model's own field list. Nothing is inferred or filled in.",
          },
          {
            title: "Check it yourself",
            body: "Every value is shown with where it was read from, and you correct anything wrong. Nothing runs until you say so.",
          },
          {
            title: "See the reasoning",
            body: "Each estimate comes with the factors that pushed it up or down, what the model did not know, and where it is unreliable.",
          },
        ].map((card) => (
          <div key={card.title} className="surface p-6">
            <h2 className="text-sm font-semibold text-ink">{card.title}</h2>
            <p className="mt-2 text-sm text-body">{card.body}</p>
          </div>
        ))}
      </section>

      <section className="mt-16 max-w-prose">
        <h2 className="text-sm font-semibold text-ink">What this is not</h2>
        <p className="mt-2 text-sm text-body">
          Not a diagnostic system, not a screening tool, and not a forecast of future
          disease. The models are trained on historical public research datasets, none of
          them has been validated on an external cohort, and an estimate here says only what
          one model would conclude about data resembling the cohort it learned from.
          Decisions about your health belong with a clinician.
        </p>
      </section>
    </PublicLayout>
  );
}
