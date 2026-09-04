import { useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { ApiError } from "../api";
import { useAuth } from "../auth";
import { AuthShell, Note, TextField } from "../components/Chrome";
import * as routes from "../lib/routes";

export default function SignIn() {
  const { status, signIn } = useAuth();
  const [params] = useSearchParams();
  const nav = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Validated, not trusted: `next` arrives in a URL anyone can construct.
  const next = routes.safeNext(params.get("next"));

  if (status === "authenticated") return <Navigate to={next} replace />;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await signIn(email.trim(), password);
      nav(next, { replace: true });
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 429
          ? "Too many attempts on this account. Wait a few minutes and try again."
          : e instanceof Error
            ? e.message
            : "Could not sign in.",
      );
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title="Welcome back"
      lead="Sign in to run an assessment."
      footer={
        <>
          <p>
            Don&apos;t have an account?{" "}
            <Link
              to={{ pathname: routes.SIGN_UP, search: location.search }}
              className="btn-link"
            >
              Create one
            </Link>
          </p>
          <p className="mt-2">
            <Link to={routes.FORGOT} className="btn-link">
              Forgot your password?
            </Link>
          </p>
        </>
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
          label="Password"
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete="current-password"
          required
        />
        {error && <Note tone="caution">{error}</Note>}
        <button className="btn-primary w-full" type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </AuthShell>
  );
}
