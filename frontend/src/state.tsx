import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { Disease, IntakeValues, PredictionResponse, SchemaResponse } from "./types";

/**
 * The working set for one consultation, shared by intake, review and result.
 *
 * Held in memory only. Nothing entered here is written to localStorage or sent anywhere
 * except the prediction request — health details should not outlive the tab by accident.
 */
interface Session {
  disease: Disease | null;
  schema: SchemaResponse | null;
  values: IntakeValues;
  prediction: PredictionResponse | null;
  /** True once the user has been through review and pressed the confirm button. */
  confirmed: boolean;
  sampleId: string | null;
}

interface SessionApi extends Session {
  start: (disease: Disease, schema: SchemaResponse, values: IntakeValues) => void;
  setValues: (values: IntakeValues) => void;
  setPrediction: (p: PredictionResponse | null) => void;
  setConfirmed: (v: boolean) => void;
  setSampleId: (id: string | null) => void;
  reset: () => void;
}

const empty: Session = {
  disease: null,
  schema: null,
  values: {},
  prediction: null,
  confirmed: false,
  sampleId: null,
};

const Ctx = createContext<SessionApi | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<Session>(empty);

  const api = useMemo<SessionApi>(
    () => ({
      ...s,
      start: (disease, schema, values) =>
        setS({ ...empty, disease, schema, values }),
      setValues: (values) =>
        // Any edit invalidates a prediction made from the previous values, so the result
        // screen can never show a number computed from inputs the user has since changed.
        setS((p) => ({ ...p, values, prediction: null, confirmed: false })),
      setPrediction: (prediction) => setS((p) => ({ ...p, prediction })),
      setConfirmed: (confirmed) => setS((p) => ({ ...p, confirmed })),
      setSampleId: (sampleId) => setS((p) => ({ ...p, sampleId })),
      reset: () => setS(empty),
    }),
    [s],
  );

  return <Ctx.Provider value={api}>{children}</Ctx.Provider>;
}

export function useSession(): SessionApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSession must be used inside <SessionProvider>");
  return v;
}
