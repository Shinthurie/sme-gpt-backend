import re
import ollama

OLLAMA_MODEL = "llama3"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def clean_ocr_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove HTML tags like <b>...</b>
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize line endings and spaces
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove junk standalone symbols but preserve useful punctuation
    text = re.sub(r"[^\S\n]*[\*\|\~`]+[^\S\n]*", " ", text)

    return text.strip()


def preserve_sensitive_tokens(text: str):
    if not isinstance(text, str):
        return "", {}

    # Preserve numbers, dates, ids, codes, money-like values
    pattern = r'\b(?:\d[\d,./:-]*|[A-Z]{2,}\d+|DOC\d+|NEW\S*)\b'
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


def llm_refine_text(ocr_text: str) -> str:
    cleaned_text = clean_ocr_text(ocr_text)
    masked_text, placeholders = preserve_sensitive_tokens(cleaned_text)

    prompt = f"""
You are correcting OCR text from a financial document written in Sinhala and English.

Rules:
- Correct OCR spelling mistakes in Sinhala and English
- Preserve the original financial meaning
- DO NOT change placeholders like __TOKEN_0__
- DO NOT invent missing content
- Remove meaningless OCR junk symbols
- Keep the text readable and structured
- Return only the corrected text
- Do not add explanations
- Do not add notes
- Do not say "Here is the corrected text"

Text:
{masked_text}

Corrected Text:
""".strip()

    print("\n[LLM] Starting OCR correction...")

    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": 0,
        }
    )

    text = response["response"].strip()

    print("[LLM] OCR correction completed.")
    print("[LLM] Corrected text preview:")
    print(text[:500])

    final_text = restore_sensitive_tokens(text, placeholders)
    final_text = strip_llm_boilerplate(final_text)
    return clean_ocr_text(final_text)