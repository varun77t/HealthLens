import type {
  DemoReportsResponse,
  Disease,
  ExtractionResult,
  ModelSummary,
  PredictionResponse,
  SamplesResponse,
  SchemaResponse,
  ScenarioResponse,
  SessionSummary,
  SignInResponse,
  User,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

/**
 * Every request carries the session cookie.
 *
 * The cookie is httpOnly, so nothing here can read it — the browser attaches it. That is
 * the point: there is no token in JavaScript for an XSS to steal, and no header for us to
 * forget on one call. It also means a plain `<a href>` download (the demo reports) is
 * authenticated for free, which a bearer token would have made impossible.
 */
const CREDENTIALS: RequestCredentials = "include";

/**
 * Called when a request is refused for want of a session.
 *
 * Registered by `AuthProvider`. Without it, an expired session leaves the app rendering a
 * signed-in shell whose every request fails — the user sees errors instead of a sign-in
 * screen, which reads as the application being broken.
 */
let unauthorizedHandler: (() => void) | null = null;

export function onUnauthorized(handler: (() => void) | null) {
  unauthorizedHandler = handler;
}

function noteUnauthorized(path: string, status: number) {
  // A rejected sign-in is a normal answer on the auth routes, not an expired session.
  if (status === 401 && !path.startsWith("/auth/")) unauthorizedHandler?.();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { credentials: CREDENTIALS });
  if (!res.ok) {
    noteUnauthorized(path, res.status);
    throw new ApiError(res.status, await readError(res));
  }
  return res.json() as Promise<T>;
}

async function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    credentials: CREDENTIALS,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    noteUnauthorized(path, res.status);
    throw new ApiError(res.status, await readError(res));
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body?: unknown) => send<T>("POST", path, body);

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** Turn FastAPI's validation payloads into something a person can act on. */
async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const d = body?.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((e) => (typeof e?.msg === "string" ? e.msg : JSON.stringify(e)))
        .join("; ");
    }
    if (d) return JSON.stringify(d);
    return res.statusText;
  } catch {
    return res.statusText || `Request failed with status ${res.status}`;
  }
}

export const api = {
  models: () => get<ModelSummary[]>("/models"),
  modelCard: (d: Disease) => get<Record<string, unknown>>(`/models/${d}`),
  schema: (d: Disease) => get<SchemaResponse>(`/models/${d}/schema`),
  samples: (d: Disease) => get<SamplesResponse>(`/samples/${d}`),
  predict: (d: Disease, features: Record<string, number | null>) =>
    post<PredictionResponse>(`/predict/${d}`, features),
  scenario: (
    d: Disease,
    features: Record<string, number | null>,
    overrides: Record<string, number | null>,
  ) => post<ScenarioResponse>(`/scenario/${d}`, { features, overrides }),

  /** Upload a report for extraction. Returns candidates for review — never a prediction. */
  extract: async (d: Disease, file: File): Promise<ExtractionResult> => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/extract/${d}`, {
      method: "POST",
      credentials: CREDENTIALS,
      body: form,
    });
    if (!res.ok) {
      noteUnauthorized(`/extract/${d}`, res.status);
      throw new ApiError(res.status, await readError(res));
    }
    return res.json() as Promise<ExtractionResult>;
  },

  demoReports: (d: Disease) => get<DemoReportsResponse>(`/demo-reports/${d}`),
  demoReportUrl: (d: Disease, caseId: string) => `${BASE}/demo-reports/${d}/${caseId}`,

  /** The take-away PDF, rendered server-side from a prediction response. */
  reportPdf: async (d: Disease, prediction: PredictionResponse): Promise<Blob> => {
    const res = await fetch(`${BASE}/report/${d}`, {
      method: "POST",
      credentials: CREDENTIALS,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(prediction),
    });
    if (!res.ok) {
      noteUnauthorized(`/report/${d}`, res.status);
      throw new ApiError(res.status, await readError(res));
    }
    return res.blob();
  },

  auth: {
    /** `null` when not signed in — the endpoint answers 204, which is not an error. */
    me: async (): Promise<User | null> => {
      const res = await fetch(`${BASE}/auth/me`, { credentials: CREDENTIALS });
      if (res.status === 204) return null;
      if (!res.ok) throw new ApiError(res.status, await readError(res));
      return res.json() as Promise<User>;
    },
    signUp: (email: string, password: string, displayName?: string) =>
      post<SignInResponse>("/auth/signup", {
        email,
        password,
        display_name: displayName?.trim() || null,
      }),
    signIn: (email: string, password: string) =>
      post<SignInResponse>("/auth/login", { email, password }),
    signOut: () => post<void>("/auth/logout"),
    signOutEverywhereElse: () => post<{ message: string }>("/auth/logout-all"),
    sessions: () => get<SessionSummary[]>("/auth/sessions"),
    changePassword: (currentPassword: string, newPassword: string) =>
      post<{ message: string }>("/auth/password", {
        current_password: currentPassword,
        new_password: newPassword,
      }),
    requestReset: (email: string) =>
      post<{ message: string }>("/auth/forgot", { email }),
    resetPassword: (token: string, newPassword: string) =>
      post<{ message: string }>("/auth/reset", { token, new_password: newPassword }),
    deleteAccount: (password: string) =>
      send<void>("DELETE", "/auth/account", { password, confirm: "DELETE" }),
  },
};
