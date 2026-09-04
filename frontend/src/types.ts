/** Shapes returned by the FastAPI backend. Mirrors backend/schemas/responses.py. */

export type Disease = "heart" | "kidney" | "diabetes";

export interface FeatureSpec {
  name: string;
  kind: "numeric" | "categorical" | "binary";
  description: string;
  label: string;
  /** Training unit ("g/dL", "mm Hg"), or "" for coded and dimensionless fields. */
  unit: string;
  group: string;
  tier: "core" | "optional";
  shap_rank: number;
  shap_mean_abs: number;
  /** This field's share of the model's total attribution mass. Coverage sums these. */
  shap_mass_share: number;
  options: { value: number; label: string }[] | null;
  categories?: number[];
  observed_min?: number;
  observed_max?: number;
  observed_median?: number;
  hard_min?: number;
  hard_max?: number;
  n_missing_in_training: number;
}

export interface GroupSpec {
  id: string;
  label: string;
  help: string;
  order: number;
}

export interface SchemaResponse {
  disease: Disease;
  module: string;
  positive_class_meaning: string;
  feature_order: string[];
  groups: GroupSpec[];
  range_note: string;
  tier_note: string;
  value_label_source: string;
  dataset_notes: string;
  features: FeatureSpec[];
  disclaimer: string;
}

export interface ModelSummary {
  disease: Disease;
  module: string;
  positive_class_meaning: string;
  model_name: string;
  calibration: string;
  version: string;
  n_train: number;
  n_test: number;
  roc_auc_test: number;
  operating_threshold: number;
  external_validation: { status: string; summary: string; dataset?: string };
  explainer: { kind: string; explain_one_median_ms: number };
  limitations: string[];
  disclaimer: string;
}

export interface Contribution {
  feature: string;
  value: number | null;
  contribution: number;
  description: string;
}

export interface Explanation {
  method: string;
  explainer: string;
  base_value: number;
  uncalibrated_probability: number;
  note: string;
  contributions: Contribution[];
}

export interface PredictionResponse {
  disease: Disease;
  module: string;
  positive_class_meaning: string;
  probability: number;
  flagged: boolean;
  threshold: number;
  threshold_rule: string;
  risk_band: { label: string; lower: number; upper: number; note: string };
  model_name: string;
  calibration: string;
  model_version: string;
  n_features_expected: number;
  n_features_provided: number;
  imputed_features: string[];
  extrapolated_features: {
    feature: string;
    value: number;
    observed_min: number;
    observed_max: number;
    note: string;
  }[];
  warnings: string[];
  explanation: Explanation | null;
  disclaimer: string;
}

export interface SampleCase {
  id: string;
  recorded_label: number;
  recorded_label_meaning: string;
  features: Record<string, number | null>;
}

export interface SamplesResponse {
  disease: Disease;
  note: string;
  cases: SampleCase[];
  disclaimer: string;
}

export interface ScenarioResponse {
  disease: Disease;
  module: string;
  label: string;
  baseline: PredictionResponse;
  modified: PredictionResponse;
  changed_features: { feature: string; from: number | null; to: number | null; description: string }[];
  probability_delta: number;
  band_changed: boolean;
  flag_changed: boolean;
  warnings: string[];
  disclaimer: string;
}

// --- document extraction ---------------------------------------------------------------

export type ExtractionStatus = "found" | "needs_review" | "missing";

export interface ExtractedField {
  name: string;
  label: string;
  unit: string;
  value: number | null;
  /** Human-readable rendering of the value ("Poor", "9.6"), or "—" if absent. */
  display: string;
  /** The line the value was read from, for provenance. */
  raw_text: string;
  page: number | null;
  status: ExtractionStatus;
  confidence: "high" | "medium" | "low";
  note: string;
}

export interface ExtractionResult {
  disease: Disease;
  module: string;
  n_expected: number;
  n_found: number;
  n_needs_review: number;
  n_missing: number;
  pages: number;
  note: string;
  warnings: string[];
  fields: ExtractedField[];
  disclaimer: string;
}

export interface DemoReportCase {
  id: string;
  recorded_label: number;
  recorded_label_meaning: string;
  filename: string;
  available: boolean;
}

export interface DemoReportsResponse {
  disease: Disease;
  cases: DemoReportCase[];
  note: string;
  disclaimer: string;
}

/**
 * How a field came to have (or not have) its value.
 *
 * `unavailable` and `blank` both reach the model as missing, but they are kept apart
 * on purpose: "I had this test and it was not recorded" is a different statement from
 * "I have not answered yet", and only the first is a considered answer. The review
 * screen refuses to run while anything is still `blank`.
 *
 * `extracted` and `flagged` both came from a document: the first was read cleanly, the
 * second could not be read confidently and must be looked at before anything runs.
 */
export type FieldStatus =
  | "entered"
  | "sample"
  | "extracted"
  | "flagged"
  | "unavailable"
  | "blank";

export interface FieldState {
  value: number | null;
  status: FieldStatus;
}

export type IntakeValues = Record<string, FieldState>;

// --- accounts ----------------------------------------------------------------------------

export interface User {
  id: string;
  email: string;
  display_name: string | null;
  /** What the greeting shows. The local part of the address when no name was given. */
  greeting_name: string;
  created_at: string;
}

export interface SignInResponse {
  user: User;
  expires_at: string;
}

export interface SessionSummary {
  id: string;
  created_at: string;
  last_seen_at: string;
  user_agent: string | null;
  current: boolean;
}
