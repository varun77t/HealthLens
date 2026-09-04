import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";
import { Disclosure, Note, PageHead, TextField } from "../components/Chrome";
import * as routes from "../lib/routes";
import type { SessionSummary } from "../types";

const MIN_PASSWORD = 10;

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

/** "Chrome on Windows" from a user-agent string, or nothing rather than a guess. */
function describeBrowser(ua: string | null): string {
  if (!ua) return "Unknown browser";
  const browser =
    /Edg\//.test(ua) ? "Edge"
    : /OPR\//.test(ua) ? "Opera"
    : /Firefox\//.test(ua) ? "Firefox"
    : /Chrome\//.test(ua) ? "Chrome"
    : /Safari\//.test(ua) ? "Safari"
    : null;
  const platform =
    /Windows/.test(ua) ? "Windows"
    : /Android/.test(ua) ? "Android"
    : /iPhone|iPad/.test(ua) ? "iOS"
    : /Mac OS X/.test(ua) ? "macOS"
    : /Linux/.test(ua) ? "Linux"
    : null;
  if (browser && platform) return `${browser} on ${platform}`;
  return browser ?? platform ?? "Unknown browser";
}

export default function Account() {
  const { user, refresh } = useAuth();
  const nav = useNavigate();

  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [changing, setChanging] = useState(false);

  const [deletePassword, setDeletePassword] = useState("");
  const [deleting, setDeleting] = useState(false);

  async function loadSessions() {
    try {
      setSessions(await api.auth.sessions());
    } catch {
      setSessions([]);
    }
  }

  useEffect(() => {
    void loadSessions();
  }, []);

  async function changePassword(event: FormEvent) {
    event.preventDefault();
    setChanging(true);
    setError(null);
    setMessage(null);
    try {
      const r = await api.auth.changePassword(currentPassword, newPassword);
      setMessage(r.message);
      setCurrentPassword("");
      setNewPassword("");
      await loadSessions();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not change the password.");
    } finally {
      setChanging(false);
    }
  }

  async function signOutOthers() {
    setError(null);
    try {
      const r = await api.auth.signOutEverywhereElse();
      setMessage(r.message);
      await loadSessions();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not sign out the other browsers.");
    }
  }

  async function deleteAccount(event: FormEvent) {
    event.preventDefault();
    setDeleting(true);
    setError(null);
    try {
      await api.auth.deleteAccount(deletePassword);
      await refresh();
      nav("/", { replace: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete the account.");
      setDeleting(false);
    }
  }

  if (!user) return null;

  return (
    <>
      <PageHead
        eyebrow="Account"
        title={user.greeting_name}
        lead={user.email}
        back={{ to: routes.home, label: "Assessments" }}
      />

      {message && (
        <div className="mb-6">
          <Note tone="attention">{message}</Note>
        </div>
      )}
      {error && (
        <div className="mb-6">
          <Note tone="caution">{error}</Note>
        </div>
      )}

      <div className="space-y-4">
        <div className="surface p-6">
          <h2 className="text-sm font-semibold text-ink">Signed-in browsers</h2>
          <p className="mt-1 text-sm text-muted">
            Each row is one browser holding a valid session for this account.
          </p>
          <ul className="mt-4 divide-y divide-line text-sm">
            {(sessions ?? []).map((s) => (
              <li key={s.id} className="flex items-baseline justify-between gap-4 py-3">
                <span className="text-ink">
                  {describeBrowser(s.user_agent)}
                  {s.current && (
                    <span className="pill ml-2 bg-hairline text-muted">This browser</span>
                  )}
                </span>
                <span className="text-xs text-faint">Last used {when(s.last_seen_at)}</span>
              </li>
            ))}
            {sessions?.length === 0 && <li className="py-3 text-muted">None.</li>}
          </ul>
          {(sessions?.length ?? 0) > 1 && (
            <button className="btn-secondary mt-5 py-2 text-xs" onClick={signOutOthers}>
              Sign out every other browser
            </button>
          )}
        </div>

        <Disclosure summary="Change your password">
          <form className="max-w-sm space-y-5" onSubmit={changePassword}>
            <TextField
              label="Current password"
              type="password"
              value={currentPassword}
              onChange={setCurrentPassword}
              autoComplete="current-password"
              required
            />
            <TextField
              label="New password"
              type="password"
              value={newPassword}
              onChange={setNewPassword}
              autoComplete="new-password"
              required
              hint={`At least ${MIN_PASSWORD} characters. Every other browser will be signed out.`}
            />
            <button
              className="btn-primary"
              type="submit"
              disabled={changing || newPassword.length < MIN_PASSWORD}
            >
              {changing ? "Saving…" : "Change password"}
            </button>
          </form>
        </Disclosure>

        <Disclosure summary="What this account stores">
          <div className="max-w-prose space-y-3 text-sm text-body">
            <p>
              Your email address, your name if you gave one, and a hash of your password —
              never the password itself. Plus a row per signed-in browser, holding no more
              than the browser description and when it was last used.
            </p>
            <p>
              Running an assessment stores nothing. Values you enter or upload are held in
              the page while you work and sent to the model to be scored; the model writes
              nothing down. An uploaded document is read and discarded — it is never stored,
              and it is not logged.
            </p>
            <p className="text-muted">
              The database has no encryption at rest. That is a deployment property, not
              something this application can claim on its own, and it is stated plainly here
              rather than implied to be otherwise.
            </p>
          </div>
        </Disclosure>

        <Disclosure summary="Delete this account">
          <form className="max-w-sm space-y-5" onSubmit={deleteAccount}>
            <Note tone="caution">
              This deletes your account, every signed-in browser, and everything saved to it.
              It cannot be undone.
            </Note>
            <TextField
              label="Confirm your password"
              type="password"
              value={deletePassword}
              onChange={setDeletePassword}
              autoComplete="current-password"
              required
            />
            <button
              className="btn-secondary text-caution-line"
              type="submit"
              disabled={deleting || deletePassword.length === 0}
            >
              {deleting ? "Deleting…" : "Delete my account"}
            </button>
          </form>
        </Disclosure>
      </div>
    </>
  );
}
