import type {
  DemoReportsResponse,
  Disease,
  ExtractionResult,
  ModelSummary,
  PredictionResponse,
  SamplesResponse,
  SchemaResponse,
  ScenarioResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  return res.json() as Promise<T>;
}

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
    const res = await fetch(`${BASE}/extract/${d}`, { method: "POST", body: form });
    if (!res.ok) throw new ApiError(res.status, await readError(res));
    return res.json() as Promise<ExtractionResult>;
  },

  demoReports: (d: Disease) => get<DemoReportsResponse>(`/demo-reports/${d}`),
  demoReportUrl: (d: Disease, caseId: string) => `${BASE}/demo-reports/${d}/${caseId}`,

  /** The take-away PDF, rendered server-side from a prediction response. */
  reportPdf: async (d: Disease, prediction: PredictionResponse): Promise<Blob> => {
    const res = await fetch(`${BASE}/report/${d}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(prediction),
    });
    if (!res.ok) throw new ApiError(res.status, await readError(res));
    return res.blob();
  },
};
