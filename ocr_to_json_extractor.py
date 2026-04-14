import json
import os
import re
import requests
from llm_correction import clean_ocr_text

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    print(f"[JSON_EXTRACTOR] Calling Ollama URL: {url}", flush=True)
    print(f"[JSON_EXTRACTOR] Model: {OLLAMA_MODEL}", flush=True)

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
            timeout=600
        )
        print(f"[JSON_EXTRACTOR] Response status: {response.status_code}", flush=True)
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
        raise Exception("Ollama extraction request timed out.")


def extract_json_block(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("LLM response is not a string.")

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found:\n{text}")

    return text[start:end + 1]


def clean_json_string(json_text: str) -> str:
    if not isinstance(json_text, str):
        return ""

    fixed = json_text.strip()
    fixed = fixed.replace("```json", "").replace("```", "").strip()
    fixed = fixed.replace("“", '"').replace("”", '"')
    fixed = fixed.replace("‘", "'").replace("’", "'")
    fixed = re.sub(r"\bNone\b", "null", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    return fixed


def normalize_number(value):
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value

    text = str(value).replace(",", "").replace("Rs", "").replace("LKR", "").strip()

    if text == "":
        return 0

    try:
        return float(text) if "." in text else int(text)
    except Exception:
        return value


def normalize_items(items):
    if not isinstance(items, list):
        return []

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue

        result.append({
            "description": str(item.get("description", "")).strip(),
            "quantity": normalize_number(item.get("quantity", 0)),
            "unit_price": normalize_number(item.get("unit_price", 0)),
        })

    return result


def normalize_root_fields(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        return {}

    return {
        "document_id": str(parsed.get("document_id", "")).strip(),
        "document_type": str(parsed.get("document_type", "unknown")).strip().lower() or "unknown",
        "order_id": str(parsed.get("order_id", "")).strip(),
        "flow_type": str(parsed.get("flow_type", "unknown")).strip().lower() or "unknown",
        "company_name": str(parsed.get("company_name", "")).strip(),
        "supplier_name": str(parsed.get("supplier_name", "")).strip(),
        "date": str(parsed.get("date", "")).strip(),
        "currency": str(parsed.get("currency", "")).strip(),
        "raw_total_amount": normalize_number(parsed.get("raw_total_amount", 0)),
        "final_total_amount": normalize_number(parsed.get("final_total_amount", 0)),
        "payable_amount": normalize_number(parsed.get("payable_amount", 0)),
        "cash_return": normalize_number(parsed.get("cash_return", 0)),
        "received_status": str(parsed.get("received_status", "")).strip(),
        "paid_status": str(parsed.get("paid_status", "")).strip(),
        "items": normalize_items(parsed.get("items", [])),
    }


def extract_structured_json_from_text(raw_text: str) -> dict:
    print("[JSON_EXTRACTOR] Starting structured JSON extraction...", flush=True)

    cleaned_text = clean_ocr_text(raw_text)

    prompt = f"""
Extract structured data from this financial document OCR.

Return ONLY valid JSON.

STRICT RULES:
- Copy values exactly from the text
- Do NOT calculate anything
- Do NOT infer missing numbers
- Do NOT modify prices, quantities, totals, dates, phone numbers, IDs, or currency
- Keep Sinhala text in Sinhala
- Do NOT translate Sinhala to English
- Use double quotes for all JSON keys and string values
- Do not use trailing commas
- If missing values -> "" or 0

Return JSON like:

{{
  "document_id": "",
  "document_type": "unknown",
  "order_id": "",
  "flow_type": "unknown",
  "company_name": "",
  "supplier_name": "",
  "date": "",
  "currency": "",
  "raw_total_amount": 0,
  "final_total_amount": 0,
  "payable_amount": 0,
  "cash_return": 0,
  "received_status": "",
  "paid_status": "",
  "items": [
    {{
      "description": "",
      "quantity": 0,
      "unit_price": 0
    }}
  ]
}}

Text:
{cleaned_text}
""".strip()

    llm_response = call_ollama(prompt)
    print(f"[JSON_EXTRACTOR] Raw LLM response preview:\n{llm_response[:1200]}", flush=True)

    json_block = extract_json_block(llm_response)
    print(f"[JSON_EXTRACTOR] Extracted JSON block preview:\n{json_block[:1200]}", flush=True)

    cleaned_json_block = clean_json_string(json_block)
    print(f"[JSON_EXTRACTOR] Cleaned JSON block preview:\n{cleaned_json_block[:1200]}", flush=True)

    try:
        parsed = json.loads(cleaned_json_block)
    except json.JSONDecodeError as e:
        print("[JSON_EXTRACTOR] JSON parsing failed.", flush=True)
        print(f"[JSON_EXTRACTOR] Error: {e}", flush=True)
        raise Exception(
            "LLM returned invalid JSON during extraction.\n"
            f"JSON error: {e}\n"
            f"Problematic JSON preview:\n{cleaned_json_block[:1500]}"
        )

    normalized = normalize_root_fields(parsed)

    print("[JSON_EXTRACTOR] Structured JSON extraction completed", flush=True)
    return normalized