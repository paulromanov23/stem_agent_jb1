from openai import OpenAI
import json
from config import (OPENAI_API_KEY, MODEL_NAME, REASONING_MODEL,
                    TEMPERATURE, TOP_P,
                    MAX_TOKENS_SOLVE, MAX_TOKENS_GENOME)

client = OpenAI(api_key=OPENAI_API_KEY)

def call(messages: list[dict], system: str = None,
         max_tokens: int = 3000, thinking: bool = False) -> str:
    """
    Returns the text content of the response.
    `thinking=True` swaps to o4-mini with internal reasoning.
    System prompt is prepended as a system message if provided.
    """
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    if thinking:
        # o4-mini does chain-of-thought internally
        # max_completion_tokens replaces max_tokens for reasoning models
        # For the thinking mode, temperature is set to a default value
        response = client.chat.completions.create(
            model= REASONING_MODEL,
            max_completion_tokens=max_tokens,
            messages=all_messages,
        )
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            # Lower temperature for code generation to reduce randomness
            temperature=TEMPERATURE,  
            top_p = TOP_P, 
            messages=all_messages,
        )

    return response.choices[0].message.content


def parse_json(text: str) -> dict:
    """Strip markdown fences if present, then parse."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())