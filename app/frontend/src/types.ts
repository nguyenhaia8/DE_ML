export type TraitId = "cEXT" | "cNEU" | "cAGR" | "cCON" | "cOPN";

export type TraitScore = {
  id: TraitId;
  label: string;
  score: number;
  validation?: number;
};

export type ModelInfo = {
  id: string;
  name: string;
  encoder?: string;
  version?: string;
  loadedAt?: string;
};

export type PredictionResponse = {
  model?: ModelInfo;
  source?: "text" | "pdf";
  wordCount?: number;
  truncated?: boolean;
  warnings?: string[];
  scores: TraitScore[];
};

export type RawPredictionResponse = {
  model?: Partial<ModelInfo> | string;
  source?: string;
  word_count?: number;
  wordCount?: number;
  truncated?: boolean;
  warnings?: string[];
  scores?:
    | Array<Partial<TraitScore> & { trait?: string; name?: string; probability?: number }>
    | Record<string, number>;
  predictions?: Record<string, number>;
};

export type PdfExtractionResponse = {
  filename?: string | null;
  text: string;
  word_count?: number;
  wordCount?: number;
};

export type InputMode = "text" | "pdf";
