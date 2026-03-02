import json
import random
import math
from pathlib import Path

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# -------------------------------------------------
# Skill ID Generator (ALG.LINEAR.EQ.001 format)
# -------------------------------------------------

TOPICS = [
    ("LINEAR", "EQ"),
    ("FRACTIONS", "ADD"),
    ("FRACTIONS", "SUB"),
    ("FRACTIONS", "MULT"),
    ("FRACTIONS", "DIV"),
    ("QUADRATIC", "SOLVE"),
    ("QUADRATIC", "FACTOR"),
    ("INEQUAL", "SOLVE"),
    ("POLY", "SIMPLIFY"),
    ("EXP", "RULES"),
]

def make_skill_id(topic, subtopic, number):
    return f"ALG.{topic}.{subtopic}.{number:03d}"


def make_skills(n_skills=100):

    skills = []

    for i in range(n_skills):

        topic, subtopic = TOPICS[i % len(TOPICS)]

        sid = make_skill_id(topic, subtopic, i+1)

        skills.append({
            "id": sid,
            "type": "skill_node",
            "domain": "math",
            "skillId": sid,
            "name": f"{topic} {subtopic} skill {i+1}",
            "vector": [
                random.random(),
                random.random(),
                random.random()
            ],
            "prerequisites": [],
            "version": 1
        })

    return skills


# -------------------------------------------------
# Synthetic Sequence Generator (DKVMN-friendly)
# -------------------------------------------------

def generate_sequences(
    skills,
    n_students=800,
    steps_per_student=80,
    n_clusters=5,
    seed=42
):

    rng = random.Random(seed)

    n_skills = len(skills)

    skill_ids = [s["skillId"] for s in skills]

    # cluster assignment
    skill_cluster = {
        skill_ids[i]: i % n_clusters
        for i in range(n_skills)
    }

    # difficulty per skill
    difficulty = {
        sid: rng.uniform(-1.0, 1.0)
        for sid in skill_ids
    }

    sequences = []

    for s in range(n_students):

        cluster_ability = [
            rng.uniform(-0.5, 0.5)
            for _ in range(n_clusters)
        ]

        student_bias = rng.uniform(-0.4, 0.4)

        seq = []

        for t in range(steps_per_student):

            # choose skill

            if t > 0 and rng.random() < 0.6:

                prev_skill = seq[-1]["skillId"]

                c = skill_cluster[prev_skill]

                candidates = [
                    sid for sid in skill_ids
                    if skill_cluster[sid] == c
                ]

                skillId = rng.choice(candidates)

            else:

                skillId = rng.choice(skill_ids)

            c = skill_cluster[skillId]

            logit = (
                1.2 * cluster_ability[c]
                + student_bias
                - 0.9 * difficulty[skillId]
            )

            p = sigmoid(logit)

            correct = 1 if rng.random() < p else 0

            seq.append({
                "skillId": skillId,
                "correct": correct
            })

            # learning update (creates transfer effect)

            learn_rate = 0.05 if correct else 0.09

            cluster_ability[c] += learn_rate

        sequences.append({
            "student_id": f"student_{s:04d}",
            "seq": seq
        })

    return sequences


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    n_skills = 100
    n_students = 800
    steps_per_student = 80

    skills = make_skills(n_skills)

    with open(out_dir / "skills.json", "w") as f:
        json.dump(skills, f, indent=2)

    sequences = generate_sequences(
        skills,
        n_students=n_students,
        steps_per_student=steps_per_student
    )

    with open(out_dir / "interactions.jsonl", "w") as f:
        for row in sequences:
            f.write(json.dumps(row) + "\n")

    print("Generated:")
    print("skills:", len(skills))
    print("students:", len(sequences))
    print("steps/student:", steps_per_student)


if __name__ == "__main__":
    main()