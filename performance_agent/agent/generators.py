import json
import uuid
from typing import Any, Dict, List, Literal, Tuple
from agent.llm import get_client  # your llm client getter

def pick_next_skills(
    dkvmn_scores: dict,
    weak_skills: list,
    k: int,
    target: float = 0.55
):
    """
    Choose skills whose predicted P(correct) is near target.
    This selects 'optimal difficulty' questions.

    DKVMN advantage:
    adaptive difficulty instead of random weak-skill selection.
    """

    scored = []

    for skill_node, score in weak_skills:
        sid = skill_node.get("skillId") or skill_node.get("id")

        if sid in dkvmn_scores:
            p = dkvmn_scores[sid]
            scored.append((skill_node, abs(p - target)))

    scored.sort(key=lambda x: x[1])

    return [s for s, _ in scored[:k]]


def _parse_json_output(text: str) -> Any:
    """
    Parses JSON output with a minimal fallback when the model wraps JSON in prose.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            return json.loads(text[start:end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and start < end:
            return json.loads(text[start:end + 1])
        raise


def generate_yt_videos(domain: str, *, model: str = "gpt-5.2") -> List[str]:
    client = get_client()
    format_hint = """[
  "<YOUTUBE_LINK>",
  "..."
]"""
    resp = client.responses.create(
        model=model,
        instructions="You are an expert curator of educational videos on YouTube.",
        input=f"Find a few high-quality YouTube video links for '{domain}'. Return only valid JSON using this format: {format_hint}",
    )
    data = _parse_json_output(resp.output_text)
    return [x for x in data if isinstance(x, str)] if isinstance(data, list) else []


def generate_quiz(
    domain: str,
    *,
    num_questions: int = 10,
    difficulty: int = 5,
    model: str = "gpt-5.2",
) -> List[Dict[str, Any]]:
    client = get_client()
    format_hint = """[
  {
    "question": "...",
    "choices": ["...", "...", "...", "..."],
    "answerIndex": 0
  }
]"""
    resp = client.responses.create(
        model=model,
        instructions=f"You are an expert instructor in {domain}.",
        input=(
            f"Generate {num_questions} multiple-choice questions for '{domain}' at difficulty {difficulty}/10. "
            f"Return only JSON in this format: {format_hint}"
        ),
    )
    data = _parse_json_output(resp.output_text)
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def generate_flash_cards(
    domain: str,
    *,
    num_cards: int = 5,
    model: str = "gpt-5.2",
) -> List[Dict[str, str]]:
    client = get_client()
    format_hint = """[
  {
    "front": "...",
    "back": "..."
  }
]"""
    resp = client.responses.create(
        model=model,
        instructions=f"You are an expert instructor in {domain}.",
        input=f"Generate {num_cards} flash cards for '{domain}'. Return only JSON in this format: {format_hint}",
    )
    data = _parse_json_output(resp.output_text)
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def generate_learning_resources(
    weak_skill_names: List[str],
    *,
    resource_type: Literal["video", "flashcards", "quiz"],
    quiz_questions: int = 10,
    quiz_difficulty: int = 5,
    flashcard_count: int = 5,
    model: str = "gpt-5.2",
) -> List[Dict[str, Any]]:
    """
    Generates resources per weak skill, based on the student's selected mode.
    """
    out: List[Dict[str, Any]] = []
    for skill_name in weak_skill_names:
        if resource_type == "video":
            content = generate_yt_videos(skill_name, model=model)
        elif resource_type == "flashcards":
            content = generate_flash_cards(skill_name, num_cards=flashcard_count, model=model)
        else:
            content = generate_quiz(
                skill_name,
                num_questions=quiz_questions,
                difficulty=quiz_difficulty,
                model=model,
            )
        out.append({
            "skill": skill_name,
            "type": resource_type,
            "content": content,
        })
    return out

def generate_recommendations_and_quiz(
    weak_skills: List[Tuple[Dict[str, Any], float]],
    student_context: Dict[str, Any],
    make_quiz: bool,
    num_questions_per_skill: int,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    client = get_client()

    system = (
        "You are an educational 'Quizzer' agent.\n"
        "You receive weak skills (skill nodes) and a DKVMN mastery score (0-1).\n"
        "Produce:\n"
        "1) Actionable recommendations per weak skill.\n"
        "2) If make_quiz=true, generate MCQ quizzes for those skills.\n\n"
        "Output MUST be valid JSON.\n"
        "Quiz rules: 4 options A-D, exactly 1 correct, include explanation.\n"
    )

    # DKVMN adaptive selection
    dkvmn_scores = {
        (s.get("skillId") or s.get("id")): score
        for s, score in weak_skills
    }

    selected_skills = pick_next_skills(
        dkvmn_scores,
        weak_skills,
        k=len(weak_skills)
    )

    payload = {
        "weak_skills": [
            {
                "skillId": s.get("skillId") or s.get("id"),
                "name": s.get("name"),
                "domain": s.get("domain"),
                "prerequisites": s.get("prerequisites", []),
                "vector": s.get("vector"),
                "dkvmn_score": dkvmn_scores[s.get("skillId") or s.get("id")]
            }
            for s in selected_skills
        ],
        "student_context": student_context,
        "make_quiz": make_quiz,
        "num_questions_per_skill": num_questions_per_skill,
        "required_output_schema": {
            "recommendations": [
                {
                    "skillId": "string",
                    "dkvmn_score": "float",
                    "priority": "high|medium|low",
                    "actions": ["string", "string"],
                    "quick_check": "string",
                    "common_mistakes": ["string"]
                }
            ],
            "quiz": [
                {
                    "skillId": "string",
                    "questions": [
                        {
                            "id": "string",
                            "prompt": "string",
                            "choices": [{"id": "A|B|C|D", "text": "string"}],
                            "answer": {"correct_choice_id": "A|B|C|D"},
                            "explanation": "string"
                        }
                    ]
                }
            ]
        }
    }

    # IMPORTANT: use chat.completions if your SDK doesn't support response_format on responses
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)}
        ],
        response_format={"type": "json_object"},
        temperature=0.5,
    )

    data = json.loads(resp.choices[0].message.content)

    # ensure IDs
    if isinstance(data.get("quiz"), list):
        for block in data["quiz"]:
            for q in block.get("questions", []):
                if not q.get("id"):
                    q["id"] = str(uuid.uuid4())

    return data
