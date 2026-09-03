/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Origin the API is served from in a build. Dev uses the vite proxy at /api. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
