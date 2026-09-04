import { Link, NavLink, useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import { useAuth } from "../auth";
import * as routes from "../lib/routes";

const DISCLAIMER_LINE = "Research & education only · Not a medical diagnosis";

/**
 * The page frame.
 *
 * The disclaimer lives here rather than in each screen, so there is no route that can
 * render a prediction without it. It is one line in the footer, not a banner on every
 * screen — repeated at full length it stops being read.
 */
function Frame({ header, children }: { header: ReactNode; children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-line/70 bg-canvas/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-5xl items-center gap-6 px-6">{header}</div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-12">{children}</main>

      <footer className="mx-auto w-full max-w-5xl px-6 pb-10">
        <p className="border-t border-line pt-5 text-xs text-faint">{DISCLAIMER_LINE}</p>
      </footer>
    </div>
  );
}

function Wordmark({ to }: { to: string }) {
  return (
    <Link to={to} className="group flex items-baseline gap-2.5">
      <span className="text-base font-semibold tracking-tight text-ink">Multi-Disease AI</span>
      <span className="hidden text-xs text-faint sm:inline">
        Explainable health risk analysis
      </span>
    </Link>
  );
}

/** The frame for everything under `/app` — signed in, with the account controls. */
export function Layout({ children }: { children: ReactNode }) {
  const { user, signOut } = useAuth();
  const nav = useNavigate();

  return (
    <Frame
      header={
        <>
          <Wordmark to={routes.home} />
          <nav className="ml-auto flex items-center gap-1 text-sm">
            <NavLink
              to={routes.home}
              end
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 ${isActive ? "text-ink" : "text-muted hover:text-ink"}`
              }
            >
              Assessments
            </NavLink>
            {user && (
              <>
                <NavLink
                  to={routes.history}
                  className={({ isActive }) =>
                    `rounded-md px-3 py-1.5 ${isActive ? "text-ink" : "text-muted hover:text-ink"}`
                  }
                >
                  History
                </NavLink>
                <NavLink
                  to={routes.account}
                  className={({ isActive }) =>
                    `max-w-[10rem] truncate rounded-md px-3 py-1.5 ${
                      isActive ? "text-ink" : "text-muted hover:text-ink"
                    }`
                  }
                  title={user.email}
                >
                  {user.greeting_name}
                </NavLink>
                <button
                  className="btn-quiet px-3 py-1.5 text-sm"
                  onClick={async () => {
                    await signOut();
                    nav("/", { replace: true });
                  }}
                >
                  Sign out
                </button>
              </>
            )}
          </nav>
        </>
      }
    >
      {children}
    </Frame>
  );
}

/** The frame for the landing page: no account controls beyond the way in. */
export function PublicLayout({ children }: { children: ReactNode }) {
  return (
    <Frame
      header={
        <>
          <Wordmark to="/" />
          <div className="ml-auto flex items-center gap-2 text-sm">
            <Link to={routes.SIGN_IN} className="btn-quiet px-3 py-1.5 text-sm">
              Sign in
            </Link>
            <Link to={routes.SIGN_UP} className="btn-primary px-4 py-1.5 text-sm">
              Create account
            </Link>
          </div>
        </>
      }
    >
      {children}
    </Frame>
  );
}

/**
 * The frame for the sign-in, sign-up and password-reset screens.
 *
 * Narrow and centred, with nothing in the header but the way back to the landing page.
 * A screen asking for a password should not also be offering somewhere else to go.
 */
export function AuthShell({
  title,
  lead,
  children,
  footer,
}: {
  title: string;
  lead?: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-line/70">
        <div className="mx-auto flex h-16 max-w-5xl items-center px-6">
          <Wordmark to="/" />
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center px-6 py-16">
        <h1 className="text-2xl font-semibold">{title}</h1>
        {lead && <p className="mt-2 text-sm text-body">{lead}</p>}
        <div className="mt-8">{children}</div>
        {footer && <div className="mt-8 text-sm text-muted">{footer}</div>}
      </main>

      <footer className="mx-auto w-full max-w-md px-6 pb-10">
        <p className="border-t border-line pt-5 text-xs text-faint">{DISCLAIMER_LINE}</p>
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

/** A labelled text input for the account forms. */
export function TextField({
  label,
  type = "text",
  value,
  onChange,
  autoComplete,
  placeholder,
  hint,
  required,
  autoFocus,
}: {
  label: string;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  autoComplete?: string;
  placeholder?: string;
  hint?: string;
  required?: boolean;
  autoFocus?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-ink">{label}</span>
      <input
        className="input w-full"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoComplete={autoComplete}
        placeholder={placeholder}
        required={required}
        autoFocus={autoFocus}
      />
      {hint && <span className="mt-1.5 block text-xs text-muted">{hint}</span>}
    </label>
  );
}
