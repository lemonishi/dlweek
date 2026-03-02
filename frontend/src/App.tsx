import {
  CloseCircleOutlined,
  FileTextOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { Button, Card, Col, Progress, Row, Upload } from "antd";
import type { UploadFile, UploadProps } from "antd";
import type { Dispatch, SetStateAction } from "react";
import { useState } from "react";
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from "react-router-dom";

const { Dragger } = Upload;

type DashboardProps = {
  files: UploadFile[];
};

function Dashboard({ files }: DashboardProps) {
  const navigate = useNavigate();

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <div>
          <p className="dashboard-kicker">Learning Pulse Dashboard</p>
          <h1>Welcome back — documents uploaded successfully.</h1>
          <p className="dashboard-subtext">
            Backend is not connected yet, so this dashboard is a frontend
            placeholder for your next phase.
          </p>
        </div>
        <Button onClick={() => navigate("/")}>Upload More Files</Button>
      </header>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={12} lg={8}>
          <Card title="Files Uploaded" bordered={false}>
            <h2 className="dashboard-value">{files.length}</h2>
            <p className="dashboard-muted">Ready for analysis pipeline</p>
          </Card>
        </Col>
        <Col xs={24} md={12} lg={8}>
          <Card title="Model Readiness" bordered={false}>
            <h2 className="dashboard-value">70%</h2>
            <Progress percent={70} showInfo={false} strokeColor="#14b8a6" />
          </Card>
        </Col>
        <Col xs={24} md={24} lg={8}>
          <Card title="Next Action" bordered={false}>
            <p className="dashboard-muted">
              Connect backend endpoint to parse and index uploaded files.
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
    </div>
  );
}

type LandingPageProps = {
  files: UploadFile[];
  setFiles: Dispatch<SetStateAction<UploadFile[]>>;
};

function LandingPage({ files, setFiles }: LandingPageProps) {
  const navigate = useNavigate();

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

            <Button
              type="primary"
              size="large"
              className="continue-button"
              disabled={files.length === 0}
              onClick={() => navigate("/dashboard")}
            >
              Continue to Dashboard
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

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={<LandingPage files={files} setFiles={setFiles} />}
        />
        <Route path="/dashboard" element={<Dashboard files={files} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
