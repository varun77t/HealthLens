import { useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { AuthShell, Note, TextField } from "../components/Chrome";
import * as routes from "../lib/routes";

const MIN_PASSWORD = 10;

export default function Reset() {
  const { token = "" } = useParams();
  const nav = useNavigate();

  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  const tooShort = password.length > 0 && password.length < MIN_PASSWORD;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.auth.resetPassword(token, password);
      setDone(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not set the new password.");
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <AuthShell
        title="Password updated"
        lead="Sign in with your new password."
      >
        <button className="btn-primary w-full" onClick={() => nav(routes.SIGN_IN)}>
          Go to sign in
        </button>
      </AuthShell>
    );
  }

  return (
    <AuthShell
      title="Set a new password"
      footer={
        <p>
          <Link to={routes.FORGOT} className="btn-link">
            Request a new link
          </Link>
        </p>
      }
    >
      <form className="space-y-5" onSubmit={submit}>
        <TextField
          label="New password"
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete="new-password"
          required
          autoFocus
          hint={`At least ${MIN_PASSWORD} characters.`}
        />
        {error && <Note tone="caution">{error}</Note>}
        <p className="text-sm text-muted">You'll be signed out on any other device.</p>
        <button className="btn-primary w-full" type="submit" disabled={busy || tooShort}>
          {busy ? "Saving…" : "Set new password"}
        </button>
      </form>
    </AuthShell>
  );
}
