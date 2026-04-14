import os
import re
import requests

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def clean_ocr_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\S\n]*[\*\|\~`]+[^\S\n]*", " ", text)

    return text.strip()


def preserve_sensitive_tokens(text: str):
    if not isinstance(text, str):
        return "", {}

    pattern = r"\b(?:\d[\d,./:-]*|[A-Z]{2,}\d+|DOC\d+|NEW\S*)\b"
    matches = re.findall(pattern, text)

    placeholders = {}
    masked = text

    for i, token in enumerate(matches):
        placeholder = f"__TOKEN_{i}__"
        placeholders[placeholder] = token
        masked = masked.replace(token, placeholder, 1)

    return masked, placeholders


def restore_sensitive_tokens(text: str, placeholders: dict):
    if not isinstance(text, str):
        return ""

    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)

    return text


def strip_llm_boilerplate(text: str) -> str:
    if not isinstance(text, str):
        return ""

    replacements = [
        "Here is the corrected text:",
        "Corrected Text:",
        "Here is the cleaned text:",
        "Here is the corrected OCR text:",
    ]

    for r in replacements:
        text = text.replace(r, "").strip()

    cutoff_markers = [
        "\nNote:",
        "\nExplanation:",
        "\nI only corrected",
        "\nI corrected",
        "\nThis text",
        "\nLet me know",
    ]

    for marker in cutoff_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    return text.strip()


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    print(f"[LLM_CORRECTION] Calling Ollama URL: {url}", flush=True)
    print(f"[LLM_CORRECTION] Model: {OLLAMA_MODEL}", flush=True)

    try:
        response = requests.post(
            url,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=600,
        )
        print(f"[LLM_CORRECTION] Response status: {response.status_code}", flush=True)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError as e:
        raise Exception(
            f"Could not connect to Ollama at {OLLAMA_HOST}. "
            f"Please start Ollama and run the model first. Error: {e}"
        )
    except requests.exceptions.HTTPError as e:
        raise Exception(f"Ollama HTTP error: {e}. Response: {response.text}")
    except requests.exceptions.Timeout:
        raise Exception("Ollama correction request timed out.")


def llm_refine_text(raw_text: str) -> str:
    print("[LLM_CORRECTION] Starting OCR text refinement...", flush=True)

    cleaned = clean_ocr_text(raw_text)
    if not cleaned:
        print("[LLM_CORRECTION] Cleaned text is empty", flush=True)
        return ""

    masked, placeholders = preserve_sensitive_tokens(cleaned)

    prompt = f"""
You are correcting OCR text from Sinhala-English financial documents.

STRICT RULES:
- Correct only OCR spelling mistakes
- Preserve Sinhala text in Sinhala
- Do NOT translate Sinhala to English
- Do NOT rewrite item names
- Do NOT change numbers, prices, totals, dates, IDs
- Return same structure
- Return ONLY corrected text

OCR text:
{masked}
""".strip()

    corrected = call_ollama(prompt)

    corrected = strip_llm_boilerplate(corrected)
    corrected = restore_sensitive_tokens(corrected, placeholders)

    print("[LLM_CORRECTION] OCR text refinement completed", flush=True)
    return corrected