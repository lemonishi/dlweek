import json, uuid
from typing import Any, Dict, List, Tuple
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