/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_RTMPOSE_MODEL_S?: string;
  readonly VITE_RTMPOSE_MODEL_M?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
