import { Link } from "react-router-dom";
import { PublicLayout } from "../components/Chrome";
import * as routes from "../lib/routes";

/**
 * The first screen anyone sees.
 *
 * Written for someone who has never used the thing and does not care how it works. No
 * jargon: not "model", not "dataset", not "estimate", not "assessment". The research layer
 * has not been deleted — it is where it was always meant to be, behind the sign-in, on each
 * module's own page.
 *
 * The one thing that stays on the surface is that this cannot diagnose anything, because
 * that is the sentence a first-time visitor most needs and least expects.
 */

const STEPS = [
  {
    step: "1",
    title: "Add your details",
    body: "Upload a health report, or answer a few short questions.",
  },
  {
    step: "2",
    title: "Check it looks right",
    body: "You see everything first. Nothing runs until you say so.",
  },
  {
    step: "3",
    title: "See what stands out",
    body: "A clear result, and the things that mattered most.",
  },
];

export default function Landing() {
  return (
    <PublicLayout>
      <section className="pt-6">
        <h1 className="max-w-2xl text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
          Your health results,
          <br />
          explained simply.
        </h1>

        <p className="mt-6 max-w-md text-lg text-body">
          Upload a health report or fill in a short form, and see what stands out — in
          everyday language.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link to={routes.SIGN_IN} className="btn-primary px-7 py-3 text-base">
            Sign in
          </Link>
          <Link to={routes.SIGN_UP} className="btn-secondary px-7 py-3 text-base">
            Create an account
          </Link>
        </div>

        <div className="mt-10 flex flex-wrap items-center gap-2">
          {["❤️ Heart", "🫘 Kidney", "🩸 Diabetes"].map((label) => (
            <span key={label} className="pill border border-line bg-surface text-muted">
              {label}
            </span>
          ))}
        </div>
      </section>

      <section className="mt-20 grid gap-5 sm:grid-cols-3">
        {STEPS.map((s) => (
          <div key={s.step} className="surface p-6">
            <span className="eyebrow">Step {s.step}</span>
            <h2 className="mt-2 text-base font-semibold text-ink">{s.title}</h2>
            <p className="mt-1.5 text-sm text-body">{s.body}</p>
          </div>
        ))}
      </section>

      <section className="mt-20 max-w-prose border-t border-line pt-8">
        <p className="text-sm text-muted">
          This is a learning tool, not a doctor. It cannot diagnose anything, and it is not
          medical advice. Always talk to a health professional about your health.
        </p>
      </section>
    </PublicLayout>
  );
}
