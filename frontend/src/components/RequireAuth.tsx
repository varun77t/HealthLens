import type { ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../auth";
import { signInWithNext } from "../lib/routes";

/**
 * The gate on every route that touches health information.
 *
 * Three states, and the middle one is why this component exists. While the first
 * `/auth/me` is still in flight the answer is not "signed out" — it is "not known yet".
 * Rendering the redirect during that window makes reloading any protected page flash the
 * sign-in screen and then bounce back, which looks like the session was lost.
 *
 * The neutral placeholder is deliberately not a spinner. The check usually resolves in a
 * few milliseconds, and a spinner appearing and vanishing on every page load is more
 * distracting than a page that simply arrives.
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const location = useLocation();

  if (status === "checking") {
    return <div className="min-h-[60vh]" aria-busy="true" aria-label="Checking your session" />;
  }

  if (status === "anonymous") {
    // Where they were going is preserved, so signing in finishes the journey they started
    // rather than dropping them at the top of the application.
    return <Navigate to={signInWithNext(location.pathname, location.search)} replace />;
  }

  return <>{children}</>;
}
