import { Link } from "react-router-dom";
import { PublicLayout } from "../components/Chrome";
import DnaHelix from "../components/DnaHelix";
import * as routes from "../lib/routes";

/**
 * The first screen anyone sees.
 *
 * Written for someone who has never used the thing and does not care how it works. No
 * jargon: not "model", not "dataset", not "estimate", not "assessment". The research layer
 * has not been deleted — it is where it was always meant to be, behind the sign-in, on each
 * module's own page.
 *
 * A headline, a sentence, two buttons. Nothing explains the product to someone who has not
 * agreed to look at it yet, and there is no list of what it covers — that is the first
 * thing on the other side of the door.
 *
 * The one thing that stays on the surface is that this cannot diagnose anything, because
 * that is the sentence a first-time visitor most needs and least expects.
 */
export default function Landing() {
  return (
    <PublicLayout>
      <section className="grid gap-2 py-10 sm:py-20 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] lg:items-center lg:gap-8 lg:py-6">
        <div>
          <h1 className="max-w-2xl text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
            Your health results,
            <br />
            explained simply.
          </h1>

          <p className="mt-6 max-w-md text-lg text-body">
            Upload a health report or fill in a short form, and see what stands out — in
            everyday language.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-3">
            <Link to={routes.SIGN_IN} className="btn-primary px-7 py-3 text-base">
              Sign in
            </Link>
            <Link to={routes.SIGN_UP} className="btn-secondary px-7 py-3 text-base">
              Create an account
            </Link>
          </div>
        </div>

        <DnaHelix className="-mx-2 h-[21rem] sm:h-[26rem] lg:-mr-4 lg:ml-0 lg:h-[36rem]" />
      </section>

      <section className="max-w-prose border-t border-line pt-8">
        <p className="text-sm text-muted">
          This is a learning tool, not a doctor. It cannot diagnose anything, and it is not
          medical advice. Always talk to a health professional about your health.
        </p>
      </section>
    </PublicLayout>
  );
}
