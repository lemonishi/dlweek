import json
import random
from agent import agent
from dkvmn.infer import mastery_scores_from_history

def load_random_student(path="data/interactions.jsonl", seed=0):
    rng = random.Random(seed)
    with open(path, "r") as f:
        lines = f.readlines()
    row = json.loads(rng.choice(lines))
    return row  # {"student_id": ..., "seq": [...]}

def main():
    # Load skill metadata (100 synthetic ALG.* skills)
    skills = json.load(open("data/skills.json"))

    # Load one student sequence from synthetic logs
    row = load_random_student("data/interactions.jsonl", seed=42)

    # Build student JSON in the format your agent expects
    student = {
        "student_id": row.get("student_id", "demo"),
        "skills": skills,
        "interactions": row["seq"],   # convert seq -> interactions
    }

    # Load the map your DKVMN was trained with (should be ~100 skills)
    skill_map = json.load(open("skill2idx.json"))

    # DKVMN inference (scores for ALL skills in the map)
    student["dkvmn"] = mastery_scores_from_history(
        student["interactions"],
        skill_map,
        model_path="dkvmn_model.pt",
    )

    out = agent(
        student,
        make_quiz=True,
        num_questions_per_skill=2,
        weak_threshold=0.6,
        top_k=3,
    )

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()