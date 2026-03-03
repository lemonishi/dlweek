import {
  AlertOutlined,
  CloseCircleOutlined,
  FileTextOutlined,
  LoadingOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { Alert, Button, Card, Col, Input, Progress, Row, Select, Upload } from "antd";
import type { UploadFile, UploadProps } from "antd";
import type { Dispatch, SetStateAction } from "react";
import { useEffect, useState } from "react";
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from "react-router-dom";

const { Dragger } = Upload;
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

type QuizQuestion = {
  question: string;
  choices: string[];
  answerIndex: number;
};

type QuizBlock = {
  skill: string;
  questions: QuizQuestion[];
  error?: string;
};

type AnalysisResult = {
  studentId: string;
  processedFiles: string[];
  docTag: string;
  weakSkills: string[];
  quiz: QuizBlock[];
};

type SkillSummaryBar = {
  skillId: string;
  skillName: string;
  learnt: number;
  toLearn: number;
  gap: number;
  status: string;
};

type SkillSummaryResponse = {
  studentId: string;
  graph: {
    type: string;
    title: string;
    xKey: string;
    maxBars: number;
    bars: SkillSummaryBar[];
  };
};

type DashboardProps = {
  files: UploadFile[];
  result: AnalysisResult | null;
};

function Dashboard({ files, result }: DashboardProps) {
  const navigate = useNavigate();
  const [summary, setSummary] = useState<SkillSummaryResponse | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const quizCount = (result?.quiz ?? []).reduce(
    (sum, block) => sum + (block.questions?.length ?? 0),
    0,
  );

  useEffect(() => {
    if (!result?.studentId) {
      setSummary(null);
      return;
    }

    let cancelled = false;
    const fetchSummary = async () => {
      setSummaryLoading(true);
      setSummaryError(null);
      try {
        const params = new URLSearchParams({
          student_id: result.studentId,
          max_bars: "5",
        });
        const response = await fetch(`${API_BASE}/api/student-skill-summary?${params.toString()}`);
        if (!response.ok) {
          throw new Error(`Skill summary request failed with ${response.status}`);
        }
        const data = (await response.json()) as SkillSummaryResponse;
        if (!cancelled) {
          setSummary(data);
        }
      } catch (e) {
        if (!cancelled) {
          setSummaryError(e instanceof Error ? e.message : "Failed to fetch skill summary.");
        }
      } finally {
        if (!cancelled) {
          setSummaryLoading(false);
        }
      }
    };

    void fetchSummary();
    return () => {
      cancelled = true;
    };
  }, [result?.studentId]);

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <div>
          <p className="dashboard-kicker">Learning Pulse Dashboard</p>
          <h1>Learning analysis completed.</h1>
          <p className="dashboard-subtext">
            Files were processed with converter + weakness detection + quiz
            generation.
          </p>
        </div>
        <Button onClick={() => navigate("/")}>Upload More Files</Button>
      </header>

      {!result && (
        <Alert
          type="warning"
          showIcon
          message="No analysis result found."
          description="Upload files from the landing page to run the full pipeline."
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} md={12} lg={8}>
          <Card title="Files Uploaded" bordered={false}>
            <h2 className="dashboard-value">{files.length}</h2>
            <p className="dashboard-muted">Processed by converter agent</p>
          </Card>
        </Col>
        <Col xs={24} md={12} lg={8}>
          <Card title="Weak Skills Found" bordered={false}>
            <h2 className="dashboard-value">{result?.weakSkills.length ?? 0}</h2>
            <Progress
              percent={Math.min(100, ((result?.weakSkills.length ?? 0) / 5) * 100)}
              showInfo={false}
              strokeColor="#14b8a6"
            />
          </Card>
        </Col>
        <Col xs={24} md={24} lg={8}>
          <Card title="Quiz Questions" bordered={false}>
            <h2 className="dashboard-value">{quizCount}</h2>
            <p className="dashboard-muted">
              Generated via `models/teacher.py`
            </p>
          </Card>
        </Col>
      </Row>

      <Card
        title="Uploaded Documents"
        bordered={false}
        className="uploaded-card"
      >
        <ul className="uploaded-list">
          {files.map((file) => (
            <li key={file.uid}>
              <FileTextOutlined />
              <span>{file.name}</span>
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Weak Skills" bordered={false} className="uploaded-card">
        <ul className="uploaded-list">
          {(result?.weakSkills ?? []).map((skill) => (
            <li key={skill}>
              <AlertOutlined />
              <span>{skill}</span>
            </li>
          ))}
          {!result?.weakSkills?.length && <li>No weak skills found.</li>}
        </ul>
      </Card>

      <Card title="Generated Quiz" bordered={false} className="uploaded-card">
        {(result?.quiz ?? []).map((block) => (
          <div key={block.skill} className="quiz-block">
            <h3>{block.skill}</h3>
            {block.error && (
              <Alert
                type="error"
                showIcon
                message={`Quiz generation failed: ${block.error}`}
                style={{ marginBottom: 8 }}
              />
            )}
            <ol className="quiz-list">
              {block.questions.map((q, index) => (
                <li key={`${block.skill}-${index}`}>
                  <p>{q.question}</p>
                  <ul>
                    {q.choices.map((choice, cIndex) => (
                      <li
                        key={`${block.skill}-${index}-${cIndex}`}
                        className={cIndex === q.answerIndex ? "correct-choice" : ""}
                      >
                        {choice}
                      </li>
                    ))}
                  </ul>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </Card>

      <Card
        title="Current vs To Learn Skills"
        bordered={false}
        className="uploaded-card"
      >
        {summaryLoading && <p className="dashboard-muted">Loading skill summary...</p>}
        {summaryError && (
          <Alert
            type="warning"
            showIcon
            message={summaryError}
            style={{ marginBottom: 8 }}
          />
        )}
        {!summaryLoading && !summaryError && !summary?.graph?.bars?.length && (
          <p className="dashboard-muted">No skill summary data found for this student.</p>
        )}
        {!summaryLoading && !summaryError && (summary?.graph?.bars?.length ?? 0) > 0 && (
          <div className="skill-summary-graph">
            {summary!.graph.bars.map((bar) => (
              <div key={bar.skillId} className="skill-summary-row">
                <div className="skill-summary-label">{bar.skillName}</div>
                <div className="skill-summary-bars">
                  <div className="skill-track">
                    <div
                      className="skill-fill skill-fill-current"
                      style={{ width: `${Math.max(0, Math.min(100, bar.learnt * 100))}%` }}
                    />
                  </div>
                  <div className="skill-track">
                    <div
                      className="skill-fill skill-fill-target"
                      style={{ width: `${Math.max(0, Math.min(100, bar.toLearn * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="skill-summary-values">
                  <span>C: {bar.learnt.toFixed(2)}</span>
                  <span>T: {bar.toLearn.toFixed(2)}</span>
                </div>
              </div>
            ))}
            <div className="skill-summary-legend">
              <span><i className="legend-current" /> Current</span>
              <span><i className="legend-target" /> To Learn</span>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

type LandingPageProps = {
  files: UploadFile[];
  setFiles: Dispatch<SetStateAction<UploadFile[]>>;
  setResult: Dispatch<SetStateAction<AnalysisResult | null>>;
};

function LandingPage({ files, setFiles, setResult }: LandingPageProps) {
  const navigate = useNavigate();
  const [studentId, setStudentId] = useState("student_123");
  const [docTag, setDocTag] = useState<"lecture" | "homework">("lecture");
  const [topK, setTopK] = useState(5);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uploadProps: UploadProps = {
    multiple: true,
    fileList: files,
    showUploadList: false,
    beforeUpload: () => false,
    onChange: ({ fileList }) => {
      setFiles(fileList);
    },
    onRemove: (removedFile) => {
      setFiles((currentFiles) =>
        currentFiles.filter((file) => file.uid !== removedFile.uid),
      );
    },
  };

  const submitForAnalysis = async () => {
    if (files.length === 0 || !studentId.trim()) {
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("student_id", studentId.trim());
      formData.append("doc_tag", docTag);
      formData.append("top_k", String(topK));

      files.forEach((file) => {
        if (file.originFileObj) {
          formData.append("files", file.originFileObj, file.name);
        }
      });

      const response = await fetch(`${API_BASE}/api/upload-analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Request failed with ${response.status}`);
      }

      const data = (await response.json()) as AnalysisResult;
      setResult(data);
      navigate("/dashboard");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="landing-page">
      <header className="landing-nav">
        <div className="brand">Learning Pulse</div>
        <Button type="primary" ghost>
          Login
        </Button>
      </header>

      <main className="landing-hero">
        <section className="hero-left">
          <h1>Documents that improve learning outcomes.</h1>
          <p>
            Upload your study files to generate personalized, explainable
            guidance. Start with multiple documents and continue to your
            dashboard.
          </p>

          <Card className="upload-card" bordered={false}>
            <div className="form-grid">
              <Input
                placeholder="Student ID"
                value={studentId}
                onChange={(event) => setStudentId(event.target.value)}
              />
              <Select
                value={docTag}
                options={[
                  { label: "Lecture", value: "lecture" },
                  { label: "Homework", value: "homework" },
                ]}
                onChange={(value) => setDocTag(value)}
              />
              <Input
                type="number"
                min={1}
                max={10}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value) || 5)}
              />
            </div>

            <Dragger {...uploadProps} className="upload-dragger">
              <p className="ant-upload-drag-icon">
                <UploadOutlined />
              </p>
              <p className="ant-upload-text">Click or drag files to upload</p>
              <p className="ant-upload-hint">
                Supports multiple files (PDF, DOCX, PPTX, TXT, CSV)
              </p>
            </Dragger>

            <div className="selected-files">
              <div className="selected-files-header">
                <h3>Selected files</h3>
                <span>{files.length} file(s)</span>
              </div>

              {files.length === 0 ? (
                <p className="selected-files-empty">No files selected yet.</p>
              ) : (
                <ul className="selected-files-list">
                  {files.map((file) => (
                    <li key={file.uid}>
                      <div>
                        <FileTextOutlined />
                        <span>{file.name}</span>
                      </div>
                      <Button
                        type="text"
                        icon={<CloseCircleOutlined />}
                        onClick={() => {
                          setFiles((currentFiles) =>
                            currentFiles.filter(
                              (currentFile) => currentFile.uid !== file.uid,
                            ),
                          );
                        }}
                        aria-label={`Remove ${file.name}`}
                      />
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {error && <Alert type="error" showIcon message={error} />}

            <Button
              type="primary"
              size="large"
              className="continue-button"
              disabled={files.length === 0 || isSubmitting}
              onClick={submitForAnalysis}
              icon={isSubmitting ? <LoadingOutlined /> : undefined}
            >
              {isSubmitting ? "Processing..." : "Upload and Analyze"}
            </Button>
          </Card>
        </section>

        <section className="hero-right">
          <div className="accent-circle" />
          <div className="mock-doc doc-back">
            <div className="doc-header" />
            <div className="doc-line" />
            <div className="doc-line short" />
            <div className="doc-block" />
          </div>
          <div className="mock-doc doc-middle">
            <div className="doc-header" />
            <div className="doc-line" />
            <div className="doc-line" />
            <div className="doc-line short" />
          </div>
          <div className="mock-doc doc-front">
            <div className="doc-header" />
            <div className="doc-line" />
            <div className="doc-line short" />
            <div className="doc-block" />
          </div>
        </section>
      </main>
    </div>
  );
}

function App() {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <LandingPage files={files} setFiles={setFiles} setResult={setResult} />
          }
        />
        <Route
          path="/dashboard"
          element={<Dashboard files={files} result={result} />}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
