from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()

def generate_yt_videos(domain: str) -> list[str]:
    '''
    Note that the links provided may not be video links, they might sometimes be search queries on youtube. If they links to videos,
    the videos are not guaranteed to be valid.
    '''

    FORMAT = '''[
        "<LINK>",
        ...
    ]'''

    response = client.responses.create(
        model="gpt-5.2",
        instructions="You are an expert curator of educational videos on Youtube.",
        input=f"Find me a few Youtube videos links on {domain} in JSON list format like so {FORMAT}."
    )
    return json.loads(response.output_text)

def generate_quiz(domain: str, num_questions: int = 10, difficulty: int = 5) -> list[dict[str, str | list[str]]]:
    FORMAT = '''[
        {
            "question": "...",
            "choices": ["...", "..."],
            "answerIndex": "..."
        },
        {
            ...
        },
        ...
    ]'''

    response = client.responses.create(
        model="gpt-5.2",
        instructions=f"You are an expert in {domain}.",
        input=f"Generate me {num_questions} multiple choice quiz questions related to {domain} with difficulty {difficulty}/10, in JSON list format like so {FORMAT}."
    )
    return json.loads(response.output_text)

def generate_flash_cards(domain: str, num_cards: int = 5) -> list[dict[str, str]]:
    FORMAT = '''[
        {
            "front": "...",
            "back": "...",
        },
        {
            ...
        },
        ...
    ]'''

    response = client.responses.create(
        model="gpt-5.2",
        instructions=f"You are an expert in {domain}.",
        input=f"Generate {num_cards} flash cards on {domain}, in JSON list format like so {FORMAT}."
    )
    return json.loads(response.output_text)