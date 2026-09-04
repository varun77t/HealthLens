import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, onUnauthorized } from "./api";
import type { User } from "./types";

/**
 * Who is signed in, for the whole application.
 *
 * The session itself is an httpOnly cookie the browser holds and this code cannot read, so
 * "am I signed in?" is answered by asking the server once on load, not by inspecting
 * storage. Nothing about the user is cached in `localStorage`: a stale cached identity
 * would let the interface render someone's name and history entry points after their
 * session had already been revoked.
 *
 * `status` has three values on purpose. `checking` is the one that is easy to leave out and
 * wrong to: while the first `/auth/me` is in flight the app knows neither that you are
 * signed in nor that you are not, and rendering the signed-out answer during that moment
 * makes a reload of any protected page flash the sign-in screen before bouncing back.
 */
type Status = "checking" | "authenticated" | "anonymous";

interface AuthApi {
  status: Status;
  user: User | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, displayName?: string) => Promise<void>;
  signOut: () => Promise<void>;
  /** Re-read the account from the server, e.g. after a profile change. */
  refresh: () => Promise<void>;
}

const Ctx = createContext<AuthApi | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>("checking");
  const [user, setUser] = useState<User | null>(null);

  const apply = useCallback((next: User | null) => {
    setUser(next);
    setStatus(next ? "authenticated" : "anonymous");
  }, []);

  const refresh = useCallback(async () => {
    // One retry, because a failure here is a service problem and not a signed-out state.
    // The API takes about fifteen seconds to load three pipelines and a KernelExplainer,
    // and a page opened during that window gets a proxy error rather than an answer.
    // Without the retry that reads as "your session ended" and bounces a perfectly
    // signed-in person to the sign-in screen. If it fails twice, anonymous is the honest
    // answer: the sign-in screen at least explains itself, where a signed-in shell whose
    // every request fails does not.
    for (const delay of [0, 600]) {
      if (delay) await new Promise((r) => setTimeout(r, delay));
      try {
        apply(await api.auth.me());
        return;
      } catch {
        /* try once more, then fall through */
      }
    }
    apply(null);
  }, [apply]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    // Any protected request refused for want of a session drops the app back to anonymous,
    // so an expired cookie shows the sign-in screen rather than a wall of failed requests.
    onUnauthorized(() => apply(null));
    return () => onUnauthorized(null);
  }, [apply]);

  const value = useMemo<AuthApi>(
    () => ({
      status,
      user,
      signIn: async (email, password) => {
        apply((await api.auth.signIn(email, password)).user);
      },
      signUp: async (email, password, displayName) => {
        apply((await api.auth.signUp(email, password, displayName)).user);
      },
      signOut: async () => {
        try {
          await api.auth.signOut();
        } finally {
          // Local state clears even if the request failed: the alternative is an interface
          // that says you are signed in after you asked not to be.
          apply(null);
        }
      },
      refresh,
    }),
    [status, user, apply, refresh],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAuth must be used inside <AuthProvider>");
  return v;
}
