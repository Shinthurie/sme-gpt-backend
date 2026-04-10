import json
import re
import os
import requests
from llm_correction import clean_ocr_text

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"

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

    response.raise_for_status()
    data = response.json()

    return data.get("response", "").strip()


def extract_json_block(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("LLM response is not a string.")

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found:\n{text}")

    return text[start:end + 1]


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
    except:
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


def extract_structured_json_from_text(raw_text: str) -> dict:
    cleaned_text = clean_ocr_text(raw_text)

    prompt = f"""
Extract structured data from this financial document OCR.

Return ONLY JSON.

Rules:
- Do NOT explain anything
- Do NOT add text outside JSON
- Keep numbers unchanged
- If missing values → "" or 0
- Identify document_type: receipt / invoice / po / dn / unknown
- Identify flow_type: receivable / payable / unknown

Return JSON like:

{{
  "document_id": "",
  "document_type": "unknown",
  "order_id": "",
  "flow_type": "unknown",
  "company_name": "",
  "supplier_name": "",
  "date": "",
  "raw_total_amount": "",
  "final_total_amount": "",
  "payable_amount": "",
  "cash_return": "",
  "currency": "",
  "received_status": "",
  "paid_status": "",
  "status": "draft",
  "language": "",
  "corrected_text": "",
  "items": []
}}

OCR:
{cleaned_text}
""".strip()

    print("\n[LLM] Starting structured extraction...")

    text = call_ollama(prompt)

    print("[LLM] Extraction done")
    print(text[:500])

    json_text = extract_json_block(text)
    data = json.loads(json_text)

    data["raw_text"] = cleaned_text
    data["corrected_text"] = clean_ocr_text(
        data.get("corrected_text", cleaned_text)
    )

    data["items"] = normalize_items(data.get("items", []))

    data["raw_total_amount"] = normalize_number(data.get("raw_total_amount", 0))
    data["final_total_amount"] = normalize_number(data.get("final_total_amount", 0))
    data["payable_amount"] = normalize_number(data.get("payable_amount", 0))
    data["cash_return"] = normalize_number(data.get("cash_return", 0))

    return data