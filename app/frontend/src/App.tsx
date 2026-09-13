import { AlertCircle, ArrowUpRight, CheckCircle2, FileText, Loader2, RefreshCw, Settings, Upload, X } from "lucide-react";
import { ChangeEvent, FormEvent, useMemo, useRef, useState } from "react";
import {
  TRAITS,
  BackendRequestError,
  countWords,
  extractTextFromPdf,
  predictFromPdf,
  predictFromText,
} from "./api";
import type { InputMode, PredictionResponse, TraitScore } from "./types";

const API_BASE_STORAGE_KEY = "cv-personality-api-base";
const MODEL_STORAGE_KEY = "cv-personality-model";
const SAMPLE_TEXT = `Senior business development manager with experience building partner pipelines, launching customer programs, and coordinating sales, product, and operations teams. Led forecasting rituals, built CRM reporting habits, mentored account managers, and improved renewal workflows across healthcare and technology accounts.`;

function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(() => localStorage.getItem(API_BASE_STORAGE_KEY) ?? "http://127.0.0.1:8000");
  const [modelId, setModelId] = useState(() => localStorage.getItem(MODEL_STORAGE_KEY) ?? "baseline");
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [pdfText, setPdfText] = useState("");
  const [isExtracting, setIsExtracting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const extractionRequestRef = useRef(0);

  const wordCount = useMemo(() => countWords(text), [text]);
  const pdfWordCount = useMemo(() => countWords(pdfText), [pdfText]);
  const activeWordCount = mode === "pdf" ? pdfWordCount : wordCount;
  const canSubmit = mode === "pdf" ? Boolean(selectedFile && pdfText && pdfWordCount >= 20 && !isExtracting) : wordCount >= 20;

  function updateApiBaseUrl(value: string) {
    setApiBaseUrl(value);
    localStorage.setItem(API_BASE_STORAGE_KEY, value);
  }

  function updateModelId(value: string) {
    setModelId(value);
    localStorage.setItem(MODEL_STORAGE_KEY, value);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setPrediction(null);

    if (mode === "text" && wordCount < 20) {
      setError("CV text must contain at least 20 words.");
      return;
    }

    if (mode === "pdf" && !selectedFile) {
      setError("Choose a PDF resume first.");
      return;
    }

    if (mode === "pdf" && isExtracting) {
      setError("PDF extraction is still running.");
      return;
    }

    if (mode === "pdf" && pdfWordCount < 20) {
      setError("Extracted CV text must contain at least 20 words.");
      return;
    }

    setIsLoading(true);
    try {
      const result =
        mode === "pdf" && selectedFile
          ? await predictFromPdf(apiBaseUrl, selectedFile, modelId)
          : await predictFromText(apiBaseUrl, text, modelId);
      setPrediction(result);
    } catch (caught) {
      if (caught instanceof BackendRequestError) {
        setError(caught.message);
        return;
      }

      setError(caught instanceof Error ? caught.message : "Could not reach prediction service.");
    } finally {
      setIsLoading(false);
    }
  }

  function loadSample() {
    setMode("text");
    setText(SAMPLE_TEXT);
    setSelectedFile(null);
    setPdfText("");
    setPrediction(null);
    setError("");
    extractionRequestRef.current += 1;
  }

  function clearInput() {
    setText("");
    setSelectedFile(null);
    setPdfText("");
    setPrediction(null);
    setError("");
    setIsExtracting(false);
    extractionRequestRef.current += 1;
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    const requestId = extractionRequestRef.current + 1;
    extractionRequestRef.current = requestId;
    setSelectedFile(file);
    setPdfText("");
    setPrediction(null);
    setError("");

    if (!file) {
      setIsExtracting(false);
      return;
    }

    setIsExtracting(true);
    try {
      const extracted = await extractTextFromPdf(apiBaseUrl, file);
      if (requestId !== extractionRequestRef.current) {
        return;
      }
      setPdfText(extracted.text);
      if (extracted.wordCount < 20) {
        setError("Extracted CV text must contain at least 20 words.");
      }
    } catch (caught) {
      if (requestId !== extractionRequestRef.current) {
        return;
      }
      setError(caught instanceof Error ? caught.message : "Could not extract text from PDF.");
    } finally {
      if (requestId === extractionRequestRef.current) {
        setIsExtracting(false);
      }
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">ML CV Analyzer</p>
          <h1>Personality prediction workspace</h1>
        </div>
        <div className="topbar-actions">
          <div className="status-pill" aria-live="polite">
            <span className="status-dot" />
            API only
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="Open settings"
            title="Settings"
            onClick={() => setIsSettingsOpen((current) => !current)}
          >
            <Settings size={20} />
          </button>

          {isSettingsOpen ? (
            <div className="settings-popover" role="dialog" aria-label="Backend settings">
              <div className="popover-header">
                <h2>Settings</h2>
                <button
                  type="button"
                  className="icon-button subtle"
                  aria-label="Close settings"
                  title="Close"
                  onClick={() => setIsSettingsOpen(false)}
                >
                  <X size={18} />
                </button>
              </div>
              <label>
                <span>Backend URL</span>
                <input value={apiBaseUrl} onChange={(event) => updateApiBaseUrl(event.target.value)} />
              </label>
            </div>
          ) : null}
        </div>
      </header>

      <section className="workspace-grid">
        <form className="input-panel" onSubmit={handleSubmit}>
          <div className="model-row">
            <label>
              <span>Model</span>
              <select value={modelId} onChange={(event) => updateModelId(event.target.value)}>
                <option value="improved">Improved</option>
                <option value="baseline">Baseline</option>
              </select>
            </label>
          </div>

          <div className="tabs" role="tablist" aria-label="Input mode">
            <button
              type="button"
              className={mode === "text" ? "tab active" : "tab"}
              onClick={() => setMode("text")}
              role="tab"
              aria-selected={mode === "text"}
            >
              <FileText size={18} />
              Text
            </button>
            <button
              type="button"
              className={mode === "pdf" ? "tab active" : "tab"}
              onClick={() => setMode("pdf")}
              role="tab"
              aria-selected={mode === "pdf"}
            >
              <Upload size={18} />
              PDF
            </button>
          </div>

          {mode === "text" ? (
            <label className="text-entry">
              <span>CV text</span>
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Paste resume text here..."
              />
              <span className={modelId === "improved" && wordCount > 256 ? "input-meter warning" : "input-meter"}>
                {wordCount} words {modelId === "improved" && wordCount > 256 ? " - likely truncated by the current encoder" : ""}
              </span>
            </label>
          ) : (
            <div className="drop-zone">
              <input ref={fileInputRef} id="cv-pdf" type="file" accept="application/pdf" onChange={handleFileChange} />
              <label htmlFor="cv-pdf">
                {isExtracting ? <Loader2 className="spin" size={26} /> : <Upload size={26} />}
                <span>{selectedFile ? selectedFile.name : "Choose PDF resume"}</span>
              </label>
              {selectedFile ? <p>{formatBytes(selectedFile.size)}</p> : null}
            </div>
          )}

          {mode === "pdf" && pdfText ? (
            <label className="text-entry pdf-preview">
              <span>Extracted content</span>
              <textarea value={pdfText} readOnly />
              <span className={modelId === "improved" && activeWordCount > 256 ? "input-meter warning" : "input-meter"}>
                {activeWordCount} words {modelId === "improved" && activeWordCount > 256 ? " - likely truncated by the current encoder" : ""}
              </span>
            </label>
          ) : null}

          {error ? (
            <div className="alert" role="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          ) : null}

          <div className="action-row">
            <button type="submit" className="primary-button" disabled={!canSubmit || isLoading}>
              {isLoading ? <Loader2 className="spin" size={18} /> : <ArrowUpRight size={18} />}
              Predict
            </button>
            <button type="button" className="ghost-button" onClick={loadSample}>
              <FileText size={18} />
              Sample
            </button>
            <button type="button" className="ghost-button" onClick={clearInput}>
              <RefreshCw size={18} />
              Reset
            </button>
          </div>
        </form>

        <section className="result-panel" aria-live="polite">
          {prediction ? <ResultsView prediction={prediction} /> : <EmptyResults />}
        </section>
      </section>
    </main>
  );
}

function ResultsView({ prediction }: { prediction: PredictionResponse }) {
  const topTrait = prediction.scores.reduce((best, item) => (item.score > best.score ? item : best), prediction.scores[0]);

  return (
    <>
      <div className="result-header">
        <div>
          <p className="eyebrow">{prediction.model?.name ?? "Model response"}</p>
          <h2>{topTrait.label} is the strongest signal</h2>
        </div>
        <CheckCircle2 size={24} />
      </div>

      <div className="summary-grid">
        <Radar scores={prediction.scores} />
        <div className="meta-list">
          <Metric label="Source" value={prediction.source ?? "text"} />
          <Metric label="Words" value={prediction.wordCount?.toLocaleString() ?? "Backend"} />
          <Metric label="Encoder limit" value={prediction.truncated ? "Likely hit" : "Clear"} />
        </div>
      </div>

      <div className="trait-list">
        {prediction.scores.map((score) => (
          <TraitRow key={score.id} score={score} />
        ))}
      </div>

      {prediction.warnings?.length ? (
        <div className="warning-list">
          {prediction.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
    </>
  );
}

function EmptyResults() {
  return (
    <div className="empty-state">
      <div className="empty-mark">
        <FileText size={30} />
      </div>
      <h2>Awaiting prediction</h2>
      <p>Results appear here after the frontend receives the model response.</p>
    </div>
  );
}

function TraitRow({ score }: { score: TraitScore }) {
  const trait = TRAITS.find((item) => item.id === score.id);
  const percent = Math.round(score.score * 100);

  return (
    <article className="trait-card">
      <div>
        <h3>{score.label}</h3>
        <p>{trait?.description}</p>
      </div>
      <div className="score-block">
        <strong>{percent}%</strong>
        <div className="score-track" aria-hidden="true">
          <span style={{ width: `${percent}%` }} />
        </div>
      </div>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Radar({ scores }: { scores: TraitScore[] }) {
  const size = 260;
  const center = size / 2;
  const maxRadius = 96;
  const points = scores.map((score, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / scores.length;
    const radius = maxRadius * score.score;
    return {
      x: center + Math.cos(angle) * radius,
      y: center + Math.sin(angle) * radius,
      labelX: center + Math.cos(angle) * (maxRadius + 26),
      labelY: center + Math.sin(angle) * (maxRadius + 26),
      score,
    };
  });

  const polygon = points.map((point) => `${point.x},${point.y}`).join(" ");

  return (
    <svg className="radar" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Big Five personality radar chart">
      {[0.35, 0.7, 1].map((scale) => {
        const ring = scores
          .map((_, index) => {
            const angle = -Math.PI / 2 + (index * Math.PI * 2) / scores.length;
            const radius = maxRadius * scale;
            return `${center + Math.cos(angle) * radius},${center + Math.sin(angle) * radius}`;
          })
          .join(" ");
        return <polygon key={scale} points={ring} className="radar-ring" />;
      })}
      {points.map((point) => (
        <line key={point.score.id} x1={center} y1={center} x2={point.labelX} y2={point.labelY} className="radar-axis" />
      ))}
      <polygon points={polygon} className="radar-fill" />
      {points.map((point) => (
        <g key={point.score.id}>
          <circle cx={point.x} cy={point.y} r="4" className="radar-point" />
          <text x={point.labelX} y={point.labelY} textAnchor="middle" dominantBaseline="middle">
            {point.score.label.slice(0, 3)}
          </text>
        </g>
      ))}
    </svg>
  );
}

function formatBytes(value: number) {
  if (value < 1024) {
    return `${value} B`;
  }

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }

  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export default App;
