import { Link } from "react-router-dom";
import type { ReactNode } from "react";

/**
 * The page frame.
 *
 * The disclaimer lives here rather than in each screen, so there is no route that can
 * render a prediction without it. It is one line in the footer, not a banner on every
 * screen — repeated at full length it stops being read.
 */
export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-line/70 bg-canvas/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-5xl items-center px-6">
          <Link to="/" className="group flex items-baseline gap-2.5">
            <span className="text-base font-semibold tracking-tight text-ink">
              Multi-Disease AI
            </span>
            <span className="hidden text-xs text-faint sm:inline">
              Explainable health risk analysis
            </span>
          </Link>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-12">{children}</main>

      <footer className="mx-auto w-full max-w-5xl px-6 pb-10">
        <p className="border-t border-line pt-5 text-xs text-faint">
          Research &amp; education only · Not a medical diagnosis
        </p>
      </footer>
    </div>
  );
}

/** A page heading with an optional back link and lead paragraph. */
export function PageHead({
  eyebrow,
  title,
  lead,
  back,
}: {
  eyebrow?: string;
  title: string;
  lead?: string;
  back?: { to: string; label: string };
}) {
  return (
    <div className="mb-10">
      {back && (
        <Link to={back.to} className="btn-quiet -ml-3 mb-3 text-xs">
          ← {back.label}
        </Link>
      )}
      {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
      <h1 className="text-3xl font-semibold">{title}</h1>
      {lead && <p className="mt-3 max-w-prose text-base text-body">{lead}</p>}
    </div>
  );
}

export function Note({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "attention" | "caution";
  children: ReactNode;
}) {
  const tones = {
    neutral: "border-line bg-hairline text-body",
    attention: "border-attention-line bg-attention-soft text-body",
    caution: "border-caution-line bg-caution-soft text-body",
  };
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${tones[tone]}`}>{children}</div>
  );
}

/**
 * Progressive disclosure for the research layer.
 *
 * The methodology, metrics and limitations are not removed — they are the substance of the
 * project — but they are not the first thing a person reads about their own result.
 */
export function Disclosure({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details open={defaultOpen} className="group surface overflow-hidden">
      <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 text-sm font-medium text-ink hover:bg-hairline">
        {summary}
        <span className="text-faint transition-transform group-open:rotate-180">▾</span>
      </summary>
      <div className="border-t border-line px-6 py-5">{children}</div>
    </details>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center gap-4 py-24 text-sm text-muted">
      <span className="h-6 w-6 animate-spin rounded-full border-2 border-line border-t-accent" />
      {label}
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <Note tone="caution">
      <p className="font-medium text-ink">Something went wrong</p>
      <p className="mt-1">{error}</p>
      <p className="mt-2 text-xs text-muted">
        If the service is not running, start it with{" "}
        <code className="rounded bg-surface px-1.5 py-0.5 text-2xs">
          uvicorn backend.main:app --port 8000
        </code>{" "}
        from the project root.
      </p>
      {onRetry && (
        <button className="btn-secondary mt-4 py-2 text-xs" onClick={onRetry}>
          Try again
        </button>
      )}
    </Note>
  );
}
