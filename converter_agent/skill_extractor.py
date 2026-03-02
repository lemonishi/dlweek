"""Convert document text to skill JSON using OpenAI GPT.

This module encapsulates prompt generation and response parsing.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# load .env
load_dotenv()

from openai import OpenAI

logger = logging.getLogger(__name__)

# Load skill vocabulary from database dynamically if needed
# For now we'll fetch when required via a simple query function.

def fetch_skill_vocab() -> str:
    """Return a string representation of all DKVMN skills in the database."""
    from cosmos_conn import container
    c = container("skill_library")
    items = list(c.read_all_items())
    lines = []
    for s in items:
        sid = s.get("skillId") or s.get("id")
        name = s.get("name", "")
        lines.append(f"{sid}: {name}")
    return "\n".join(lines)


class SkillExtractor:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        # allow model to be specified via env or argument
        self.model = model or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.skill_vocab = fetch_skill_vocab()

    PROMPT_TEMPLATE = """
You are an educational AI that maps document content to a standardized skill
schema.  The available DKVMN skills are listed below:

{skill_vocab}

Document Type: {doc_type}

Document Text:
{doc_text}

Return EXACTLY one JSON object with the keys "skills_learnt" and
"skills_to_learn".  Each must be an array of objects matching this schema:

{{
  id: string,
  studentId: string,
  name: string,
  description: string,               # short description or empty string
  tags: [string],                    # zero or more tags
  vector: {{                         # numeric sub-object
      mastery: number,
      edge_case: number,
      implementation: number
  }},
  prerequisites: [string],
  difficulty: number,                # provide a numeric rating (float)
  version: integer,
  timestamp: string                 # ISO date or empty string
}}

Every field must appear exactly as above; if you do not have information
for description, tags, vector values, prerequisites, or timestamp, use an
empty string, an empty list, or zero for numbers.  The `difficulty` field
should be a numeric value (e.g. a float between 0.0 and 10.0).  If you
feel compelled to use words like "beginner", "intermediate" or
"advanced", you may, but they will be mapped to numbers later.

Rules:
- Use the skill vocabulary when naming skills and setting `id`/`name`.
- If the document appears to be homework (handwritten or explicitly tagged),
  treat identified skills as "skills_learnt"; otherwise place them in
  "skills_to_learn".
- Do NOT include extra keys or explanatory text, and do not wrap the JSON in
  markdown fences.
- If studentId is known it should be set; otherwise leave blank.
"""

    def extract(self, doc_text: str, doc_type: str, student_id: str = "") -> Dict[str, Any]:
        prompt = self.PROMPT_TEMPLATE.format(
            skill_vocab=self.skill_vocab,
            doc_type=doc_type,
            doc_text=doc_text
        )
        logger.debug("Calling OpenAI with prompt length %d", len(prompt))
        # log prompt in debug for troubleshooting
        logger.debug("Prompt:\n%s", prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an educational AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from model response: %s", text)
            raise
        # ensure studentId applied
        for arr in ("skills_learnt", "skills_to_learn"):
            for obj in result.get(arr, []):
                if student_id:
                    obj["studentId"] = student_id
        return result
