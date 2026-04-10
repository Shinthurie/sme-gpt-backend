import json
import re
import ollama
from llm_correction import clean_ocr_text

OLLAMA_MODEL = "llama3"


def extract_json_block(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("LLM response is not a string.")

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found in response:\n{text}")

    return text[start:end + 1]


def normalize_number(value):
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace("Rs", "")
    text = text.replace("LKR", "")
    text = text.strip()

    if text == "":
        return 0

    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return value


def normalize_items(items):
    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue

        normalized.append({
            "description": str(item.get("description", "")).strip(),
            "quantity": normalize_number(item.get("quantity", 0)),
            "unit_price": normalize_number(item.get("unit_price", 0)),
        })

    return normalized


def extract_order_id_rule_based(text: str) -> str:
    if not isinstance(text, str):
        return ""

    patterns = [
        r"Order\s*ID[:\s]+([A-Za-z0-9\-\/]+)",
        r"Order[:\s]+([A-Za-z0-9\-\/]+)",
        r"Invoice\s*No[:\s]+([A-Za-z0-9\-\/]+)",
        r"Receipt\s*No[:\s]+([A-Za-z0-9\-\/]+)",
        r"PO\s*No[:\s]+([A-Za-z0-9\-\/]+)",
        r"DN\s*No[:\s]+([A-Za-z0-9\-\/]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""


def extract_date_rule_based(text: str) -> str:
    if not isinstance(text, str):
        return ""

    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM))?",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return ""


def extract_total_rule_based(text: str) -> str:
    if not isinstance(text, str):
        return ""

    patterns = [
        r"Total[:\s]*([\d,]+(?:\.\d+)?)",
        r"Total[:\s]*\n\s*([\d,]+(?:\.\d+)?)",
        r"මුළු[\s:]*([\d,]+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value.startswith("."):
                value = "0" + value
            return value

    return ""


def extract_cash_return_rule_based(text: str) -> str:
    if not isinstance(text, str):
        return ""

    patterns = [
        r"Cash\s*return[:\s]*([\d,]+(?:\.\d+)?)",
        r"Change[:\s]*([\d,]+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value.startswith("."):
                value = "0" + value
            return value

    return ""


def fallback_extract_items_from_text(text: str):
    items = []
    if not isinstance(text, str):
        return items

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        # Examples:
        # "කොණ්ඩය කැපීමට (වැඩිහිටි) 600"
        # "2 කොණ්ඩය කැපීමට (පොඩිහිට්) 400"
        # "3 රැවල කැපීමට 300"
        m = re.match(r"^(?:(\d+)\s+)?(.+?)\s+(\d+(?:,\d+)?(?:\.\d+)?)$", line)
        if not m:
            continue

        qty_raw, desc, price_raw = m.groups()

        desc = desc.strip()
        price = normalize_number(price_raw)
        qty = normalize_number(qty_raw) if qty_raw else 1

        desc_low = desc.lower()

        if desc_low in ["date", "order id", "cash return", "total", "rs", "no.", "මුදල"]:
            continue
        if "date" in desc_low or "order id" in desc_low or "cash return" in desc_low:
            continue
        if desc_low.startswith("total"):
            continue

        items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": price
        })

    return items


def fix_flow_type_and_flags(data: dict):
    flow_type = str(data.get("flow_type", "unknown")).strip().lower()

    if flow_type == "receivable":
        data["received_status"] = data.get("received_status", "") or "not_received"
        data["paid_status"] = ""
    elif flow_type == "payable":
        data["paid_status"] = data.get("paid_status", "") or "not_paid"
        data["received_status"] = ""
    else:
        data["received_status"] = data.get("received_status", "") or ""
        data["paid_status"] = data.get("paid_status", "") or ""

    return data


def extract_structured_json_from_text(raw_text: str) -> dict:
    cleaned_text = clean_ocr_text(raw_text)

    prompt = f"""
You are extracting structured information from OCR text of a financial document.

Document type must be one of:
- receipt
- invoice
- po
- dn
- unknown

flow_type must be one of:
- receivable
- payable
- unknown

Return ONLY valid JSON.
Do not add markdown.
Do not explain anything.
Do not write anything before or after the JSON.

Rules:
- Extract order_id if available
- Preserve numbers where possible
- If a value is missing, use empty string for text fields and 0 for numeric fields
- Try to extract line items if clearly available
- If line items are not clearly available, use an empty list
- For salon bills, simple payment slips, and customer bills, classify as "receipt"
- If flow_type is receivable:
  - received_status = "not_received"
  - paid_status = ""
- If flow_type is payable:
  - paid_status = "not_paid"
  - received_status = ""
- If flow_type is unknown:
  - keep both blank
- If there is "Cash return", extract it as cash_return

Return JSON exactly in this structure:

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
  "items": [
    {{
      "description": "",
      "quantity": 0,
      "unit_price": 0
    }}
  ]
}}

OCR text:
{cleaned_text}

JSON:
""".strip()

    print("\n[LLM] Starting structured extraction...")

    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"temperature": 0}
    )

    text = response["response"].strip()

    print("[LLM] Structured extraction completed.")
    print("[LLM] Raw response preview:")
    print(text[:700])

    json_text = extract_json_block(text)
    data = json.loads(json_text)

    data["raw_text"] = cleaned_text
    data["corrected_text"] = clean_ocr_text(data.get("corrected_text", cleaned_text))

    llm_items = normalize_items(data.get("items", []))
    fallback_items = fallback_extract_items_from_text(cleaned_text)

    data["items"] = llm_items if len(llm_items) >= len(fallback_items) else fallback_items

    # Rule-based fallback for fields
    if not str(data.get("order_id", "")).strip():
        data["order_id"] = extract_order_id_rule_based(cleaned_text)

    rule_date = extract_date_rule_based(cleaned_text)
    if rule_date:
        data["date"] = rule_date

    rule_total = extract_total_rule_based(cleaned_text)
    if rule_total:
        data["raw_total_amount"] = rule_total
        if not str(data.get("final_total_amount", "")).strip():
            data["final_total_amount"] = rule_total

    rule_cash_return = extract_cash_return_rule_based(cleaned_text)
    if rule_cash_return:
        data["cash_return"] = rule_cash_return
        if not str(data.get("payable_amount", "")).strip():
            data["payable_amount"] = rule_cash_return

    # Clean numeric fields
    data["raw_total_amount"] = normalize_number(data.get("raw_total_amount", 0))
    data["final_total_amount"] = normalize_number(data.get("final_total_amount", 0))
    data["payable_amount"] = normalize_number(data.get("payable_amount", 0))
    data["cash_return"] = normalize_number(data.get("cash_return", 0))

    data = fix_flow_type_and_flags(data)

    return data