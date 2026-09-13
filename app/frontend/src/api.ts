import type { PdfExtractionResponse, PredictionResponse, RawPredictionResponse, TraitId, TraitScore } from "./types";

export class BackendRequestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BackendRequestError";
  }
}

export const TRAITS: Array<{ id: TraitId; label: string; description: string }> = [
  {
    id: "cEXT",
    label: "Extraversion",
    description: "Social energy and outward engagement",
  },
  {
    id: "cNEU",
    label: "Neuroticism",
    description: "Emotional volatility signal",
  },
  {
    id: "cAGR",
    label: "Agreeableness",
    description: "Cooperative and empathetic language",
  },
  {
    id: "cCON",
    label: "Conscientiousness",
    description: "Planning, reliability, and structure",
  },
  {
    id: "cOPN",
    label: "Openness",
    description: "Curiosity and breadth of interests",
  },
];

export function countWords(value: string) {
  return value.trim().split(/\s+/).filter(Boolean).length;
}

export async function extractTextFromPdf(apiBaseUrl: string, file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${normalizeBaseUrl(apiBaseUrl)}/api/v1/extract/pdf`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new BackendRequestError(await readErrorMessage(response));
  }

  const payload = (await response.json()) as PdfExtractionResponse;
  return {
    filename: payload.filename ?? file.name,
    text: payload.text,
    wordCount: payload.wordCount ?? payload.word_count ?? countWords(payload.text),
  };
}

export async function predictFromText(apiBaseUrl: string, text: string, modelId: string) {
  const response = await fetch(`${normalizeBaseUrl(apiBaseUrl)}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model_id: modelId }),
  });

  return parsePredictionResponse(response);
}

export async function predictFromPdf(apiBaseUrl: string, file: File, modelId: string) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_id", modelId);

  const response = await fetch(`${normalizeBaseUrl(apiBaseUrl)}/api/v1/predict/pdf`, {
    method: "POST",
    body: formData,
  });

  return parsePredictionResponse(response);
}

async function parsePredictionResponse(response: Response): Promise<PredictionResponse> {
  if (!response.ok) {
    throw new BackendRequestError(await readErrorMessage(response));
  }

  const payload = (await response.json()) as RawPredictionResponse;
  return normalizePrediction(payload);
}

async function readErrorMessage(response: Response) {
  const text = await response.text();
  if (!text) {
    return `Prediction failed with status ${response.status}`;
  }

  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    return text;
  }

  return text;
}

function normalizePrediction(payload: RawPredictionResponse): PredictionResponse {
  const model =
    typeof payload.model === "string"
      ? { id: payload.model, name: payload.model }
      : {
          id: payload.model?.id ?? "unknown",
          name: payload.model?.name ?? payload.model?.id ?? "Prediction model",
          encoder: payload.model?.encoder,
          version: payload.model?.version,
          loadedAt: payload.model?.loadedAt,
        };

  return {
    model,
    source: payload.source === "pdf" ? "pdf" : "text",
    wordCount: payload.wordCount ?? payload.word_count,
    truncated: payload.truncated,
    warnings: payload.warnings ?? [],
    scores: normalizeScores(payload),
  };
}

function normalizeScores(payload: RawPredictionResponse): TraitScore[] {
  const source = payload.scores ?? payload.predictions ?? {};

  if (Array.isArray(source)) {
    return TRAITS.map((trait) => {
      const match = source.find((item) => item.id === trait.id || item.trait === trait.id || item.name === trait.label);
      return {
        id: trait.id,
        label: trait.label,
        score: clampScore(match?.score ?? match?.probability ?? 0),
        validation: match?.validation,
      };
    });
  }

  return TRAITS.map((trait) => ({
    id: trait.id,
    label: trait.label,
    score: clampScore(source[trait.id] ?? 0),
  }));
}

function clampScore(value: number) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.min(1, Math.max(0, value));
}

function normalizeBaseUrl(value: string) {
  return value.trim().replace(/\/+$/, "");
}
