import { Link, NavLink, useNavigate } from "react-router-dom";
import { useState, type ReactNode } from "react";
import { useAuth } from "../auth";
import { useTheme } from "../theme";
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
        <div className="nav-type mx-auto flex h-16 max-w-5xl items-center gap-3 px-4 sm:gap-6 sm:px-6">
          {header}
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-5 py-12 sm:px-6">{children}</main>

      <footer className="mx-auto w-full max-w-5xl px-5 pb-10 sm:px-6">
        <p className="border-t border-line pt-5 text-xs text-faint">{DISCLAIMER_LINE}</p>
      </footer>
    </div>
  );
}

/**
 * `tagline` is off on the public screens. "Explainable health risk analysis" describes the
 * project to someone who already knows what it is; on a landing page it is the first thing
 * a first-time visitor reads, and it reads like documentation.
 */
function Wordmark({ to, tagline = false }: { to: string; tagline?: boolean }) {
  return (
    <Link to={to} className="group flex items-baseline gap-2.5">
      <span className="text-base font-semibold tracking-tight text-ink">HealthLens</span>
      {tagline && (
        <span className="hidden font-sans text-xs font-normal tracking-normal text-faint sm:inline">
          A closer look at your results
        </span>
      )}
    </Link>
  );
}

/**
 * The dark-mode button.
 *
 * The icon shows what pressing it gives you, not what you are currently in — a sun while
 * you are in the dark. Showing the current state is the commoner choice and the one people
 * misread, because a button is a thing you press, so its face is read as its outcome.
 */
export function ThemeToggle({ className = "" }: { className?: string }) {
  const { theme, toggle } = useTheme();
  const goingTo = theme === "dark" ? "light" : "dark";

  return (
    <button
      type="button"
      onClick={toggle}
      className={`btn rounded-md p-2 text-muted hover:bg-accent-soft hover:text-ink ${className}`}
      aria-label={`Switch to ${goingTo} mode`}
      title={`Switch to ${goingTo} mode`}
    >
      {theme === "dark" ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="1.8" strokeLinecap="round" aria-hidden>
          <circle cx="12" cy="12" r="4.2" />
          <path d="M12 2.6v2.2M12 19.2v2.2M21.4 12h-2.2M4.8 12H2.6M18.6 5.4l-1.6 1.6M7 17l-1.6 1.6M18.6 18.6L17 17M7 7L5.4 5.4" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M20.5 14.6A8.6 8.6 0 1 1 9.4 3.5a6.9 6.9 0 0 0 11.1 11.1Z" />
        </svg>
      )}
    </button>
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
          <Wordmark to={routes.home} tagline />
          <nav className="ml-auto flex items-center gap-1 text-sm">
            <NavLink
              to={routes.home}
              end
              className={({ isActive }) =>
                `hidden rounded-md px-3 py-1.5 sm:block ${
                  isActive ? "text-ink" : "text-muted hover:text-ink"
                }`
              }
            >
              Assessments
            </NavLink>
            <ThemeToggle />
            {user && (
              <>
                <NavLink
                  to={routes.history}
                  className={({ isActive }) =>
                    `rounded-md px-2 py-1.5 sm:px-3 ${
                      isActive ? "text-ink" : "text-muted hover:text-ink"
                    }`
                  }
                >
                  History
                </NavLink>
                <NavLink
                  to={routes.account}
                  className={({ isActive }) =>
                    `max-w-[5rem] truncate rounded-md px-2 py-1.5 sm:max-w-[10rem] sm:px-3 ${
                      isActive ? "text-ink" : "text-muted hover:text-ink"
                    }`
                  }
                  title={user.email}
                >
                  {user.greeting_name}
                </NavLink>
                <button
                  className="btn-quiet px-2 py-1.5 text-sm sm:px-3"
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

/**
 * The frame for the landing page.
 *
 * No account controls in the header: the page's own two buttons are a few centimetres
 * below it, and the same pair twice on one short screen reads as an interface that has
 * lost track of what it already offered.
 */
export function PublicLayout({ children }: { children: ReactNode }) {
  return (
    <Frame
      header={
        <>
          <Wordmark to="/" />
          <ThemeToggle className="ml-auto" />
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
        <div className="nav-type mx-auto flex h-16 max-w-5xl items-center px-6">
          <Wordmark to="/" />
          <ThemeToggle className="ml-auto" />
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
        <code className="rounded bg-field px-1.5 py-0.5 text-2xs">
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

/**
 * A labelled text input for the account forms.
 *
 * A password field gets a reveal button. The argument for one is not convenience: the
 * alternative to seeing what you typed is guessing, and a person who cannot check a long
 * password picks a short one instead. It is `type="button"`, so it never submits the form
 * it sits in, and it stays inside the label so that clicking it also returns focus to the
 * field rather than stranding the caret.
 *
 * The icon shows the action rather than the current state — an open eye while the password
 * is hidden — and the accessible name says which in words, because the eye and the crossed
 * eye are both used for both conventions and neither one is self-evident.
 */
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
  const [revealed, setRevealed] = useState(false);
  const isPassword = type === "password";
  const action = revealed ? "Hide password" : "Show password";

  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-ink">{label}</span>
      <div className="relative">
        <input
          className={`input w-full ${isPassword ? "pr-12" : ""}`}
          type={isPassword && revealed ? "text" : type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoComplete={autoComplete}
          placeholder={placeholder}
          required={required}
          autoFocus={autoFocus}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setRevealed((v) => !v)}
            className="absolute inset-y-0 right-0 flex items-center rounded-r-lg px-3.5
                       text-muted transition-colors hover:text-ink"
            aria-label={action}
            title={action}
          >
            {revealed ? (
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M9.9 5.7A9.9 9.9 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17.6 17.6 0 0 1-3.4 4.2" />
                <path d="M6.5 7.7A17.2 17.2 0 0 0 2.5 12S6 18.5 12 18.5c1.6 0 3.1-.5 4.4-1.2" />
                <path d="M10.1 10.1a2.7 2.7 0 0 0 3.8 3.8" />
                <path d="M3.5 3.5l17 17" />
              </svg>
            ) : (
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
                <circle cx="12" cy="12" r="3.1" />
              </svg>
            )}
          </button>
        )}
      </div>
      {hint && <span className="mt-1.5 block text-xs text-muted">{hint}</span>}
    </label>
  );
}
