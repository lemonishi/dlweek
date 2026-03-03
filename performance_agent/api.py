from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from azure.cosmos.exceptions import CosmosResourceNotFoundError


# Ensure both project .env files are loaded.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "performance_agent" / ".env")
load_dotenv(ROOT / "converter_agent" / ".env")

# Make local packages importable regardless of launch directory.
import sys

sys.path.append(str(ROOT / "converter_agent"))
sys.path.append(str(ROOT / "performance_agent"))
sys.path.append(str(ROOT))

from process_upload import process_file  # type: ignore
from agent.pipeline import get_weak_skill_names_for_student  # type: ignore
from models.teacher import generate_quiz  # type: ignore


app = FastAPI(title="DLWeek API", version="0.1.0")

extra_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *extra_origins,
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_filename(name: str) -> str:
    # Keep only a basic basename to avoid path traversal.
    return Path(name).name or "upload.bin"


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload-analyze")
async def upload_analyze(
    student_id: str = Form(...),
    doc_tag: str = Form("lecture"),
    top_k: int = Form(5),
    quiz_questions: int = Form(5),
    quiz_difficulty: int = Form(5),
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    """
    End-to-end flow:
      Part 1: process uploaded files via converter_agent and upsert into Cosmos.
      Part 2: compute weak skill names via performance_agent pipeline.
      Part 3: generate quiz blocks per weak skill using models/teacher.py.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files received.")

    processed_files: List[str] = []
    try:
        for uploaded in files:
            suffix = Path(uploaded.filename or "").suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(uploaded.file, tmp)
                tmp_path = Path(tmp.name)

            try:
                process_file(tmp_path, student_id=student_id, user_tag=doc_tag)
                processed_files.append(_safe_filename(uploaded.filename or tmp_path.name))
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        weak_skills = get_weak_skill_names_for_student(
            student_id,
            top_k=top_k,
            model_path=str(ROOT / "performance_agent" / "dkvmn_model.pt"),
            skill_map_path=str(ROOT / "performance_agent" / "skill2idx.json"),
        )
    except CosmosResourceNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cosmos resource not found. Check COSMOS_DB_NAME/COSMOS_DATABASE and "
                "required containers (e.g. skill_library, objective, student_profiles). "
                f"Original error: {exc}"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    quiz_blocks: List[Dict[str, Any]] = []
    for skill_name in weak_skills:
        try:
            questions = generate_quiz(
                skill_name,
                num_questions=quiz_questions,
                difficulty=quiz_difficulty,
            )
            quiz_blocks.append({"skill": skill_name, "questions": questions})
        except Exception as exc:
            quiz_blocks.append({"skill": skill_name, "questions": [], "error": str(exc)})

    return {
        "studentId": student_id,
        "processedFiles": processed_files,
        "docTag": doc_tag,
        "weakSkills": weak_skills,
        "quiz": quiz_blocks,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=True)
