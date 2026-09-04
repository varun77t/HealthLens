import type { Disease } from "../types";

/**
 * Route helpers.
 *
 * Everything that touches health information lives under `/app`. The prefix is not
 * decoration: it is what makes "protected" a property of a path rather than a list of
 * pages someone has to keep up to date, and it is what `safeNext` checks against.
 */
export const APP = "/app";

export const home = APP;
export const start = (d: Disease | string) => `${APP}/${d}`;
export const intake = (d: Disease | string) => `${APP}/${d}/enter`;
export const review = (d: Disease | string) => `${APP}/${d}/review`;
export const result = (d: Disease | string) => `${APP}/${d}/result`;
export const about = (d: Disease | string) => `${APP}/${d}/about`;
export const account = `${APP}/account`;
export const history = `${APP}/history`;
export const historyItem = (id: string) => `${APP}/history/${id}`;

export const SIGN_IN = "/signin";
export const SIGN_UP = "/signup";
export const FORGOT = "/forgot";

/**
 * Validate a `?next=` destination before navigating to it.
 *
 * An unvalidated `next` is an open redirect: a link to
 * `/signin?next=https://elsewhere.example` would send someone to another site immediately
 * after they typed their password, with our origin in the referrer. Only same-site paths
 * inside the application are accepted — a scheme, a protocol-relative `//host`, or a
 * backslash (which some browsers normalise to `/`) is refused.
 */
export function safeNext(raw: string | null): string {
  if (!raw) return home;
  let value: string;
  try {
    value = decodeURIComponent(raw);
  } catch {
    return home;
  }
  if (!value.startsWith(APP)) return home;
  if (value.startsWith("//") || value.includes("\\")) return home;
  // `/application-elsewhere` starts with "/app" but is not inside it.
  const next = value.charAt(APP.length);
  if (next && next !== "/" && next !== "?") return home;
  return value;
}

export function signInWithNext(pathname: string, search = ""): string {
  return `${SIGN_IN}?next=${encodeURIComponent(pathname + search)}`;
}
