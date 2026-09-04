import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { AuthShell, Note, TextField } from "../components/Chrome";
import * as routes from "../lib/routes";

export default function Forgot() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.auth.requestReset(email.trim());
      setSent(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not send the reset link.");
    } finally {
      setBusy(false);
    }
  }

  if (sent) {
    return (
      <AuthShell
        title="Check your email"
        // Deliberately not "we sent you an email": the server answers identically whether or
        // not an account exists, and this screen must not contradict that by confirming one.
        lead="If that address has an account, a link is on its way. It works once, and only for the next 30 minutes."
        footer={
          <p>
            <Link to={routes.SIGN_IN} className="btn-link">
              Back to sign in
            </Link>
          </p>
        }
      >
        <p className="text-sm text-muted">
          Nothing has changed yet. Your current password keeps working until you use the link.
        </p>
      </AuthShell>
    );
  }

  return (
    <AuthShell
      title="Reset your password"
      lead="We'll email you a link to set a new one."
      footer={
        <p>
          <Link to={routes.SIGN_IN} className="btn-link">
            Back to sign in
          </Link>
        </p>
      }
    >
      <form className="space-y-5" onSubmit={submit}>
        <TextField
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
          autoComplete="email"
          required
          autoFocus
        />
        {error && <Note tone="caution">{error}</Note>}
        <button className="btn-primary w-full" type="submit" disabled={busy}>
          {busy ? "Sending…" : "Send reset link"}
        </button>
      </form>
    </AuthShell>
  );
}
