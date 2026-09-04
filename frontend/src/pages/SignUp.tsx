import { useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../auth";
import { AuthShell, Note, TextField } from "../components/Chrome";
import * as routes from "../lib/routes";

const MIN_PASSWORD = 10;

export default function SignUp() {
  const { status, signUp } = useAuth();
  const [params] = useSearchParams();
  const nav = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const next = routes.safeNext(params.get("next"));
  if (status === "authenticated") return <Navigate to={next} replace />;

  const tooShort = password.length > 0 && password.length < MIN_PASSWORD;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signUp(email.trim(), password, displayName);
      nav(next, { replace: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create the account.");
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title="Create an account"
      lead="It takes a moment, and keeps your results to yourself."
      footer={
        <p>
          Already have one?{" "}
          <Link to={{ pathname: routes.SIGN_IN, search: location.search }} className="btn-link">
            Sign in
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
        <TextField
          label="Name"
          value={displayName}
          onChange={setDisplayName}
          autoComplete="name"
          placeholder="Optional"
          hint="So we know what to call you."
        />
        <TextField
          label="Password"
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete="new-password"
          required
          hint={`At least ${MIN_PASSWORD} characters.`}
        />
        {tooShort && (
          <Note tone="attention">
            {MIN_PASSWORD - password.length} more character
            {MIN_PASSWORD - password.length === 1 ? "" : "s"} needed.
          </Note>
        )}
        {error && <Note tone="caution">{error}</Note>}

        <p className="text-sm text-muted">
          Nothing is saved unless you ask for it, and you can delete your account any time.
        </p>

        <button className="btn-primary w-full" type="submit" disabled={busy || tooShort}>
          {busy ? "Creating your account…" : "Create account"}
        </button>
      </form>
    </AuthShell>
  );
}
