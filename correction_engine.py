import re
import json
from copy import deepcopy
from llm_correction import llm_refine_text

SINHALA_CORRECTIONS = {
    "ඇනවුම": "ඇණවුම",
    "කදාසි": "කඩදාසි",
    "මුලු": "මුළු",
    "සිනි": "සීනි",
    "පත‍්‍රය": "පත්‍රය",
    "ගාන": "ගණන",
    "වටිනාකම්": "වටිනාකම",
    "ඉන්වොය්ස්": "ඉන්වොයිස්",
    "රිසිට්": "රිසිට්පත"
}

ENGLISH_CORRECTIONS = {
    "invioce": "invoice",
    "reciept": "receipt",
    "prnter": "printer",
    "papre": "paper",
    "payble": "payable",
    "toatl": "total",
    "amunt": "amount",
    "devces": "devices",
    "keybaord": "keyboard",
    "quatity": "quantity",
    "suplier": "supplier"
}


def preserve_numbers(text: str):
    if not isinstance(text, str):
        return text, {}

    numbers = re.findall(r'\b[\d.,/-]+\b', text)
    placeholders = {}
    masked = text

    for i, num in enumerate(numbers):
        placeholder = f"__NUM_{i}__"
        placeholders[placeholder] = num
        masked = masked.replace(num, placeholder, 1)

    return masked, placeholders


def restore_numbers(text: str, placeholders: dict):
    for placeholder, value in placeholders.items():
        text = text.replace(placeholder, value)
    return text


def _replace_english_word(word: str):
    prefix = ""
    suffix = ""

    while word and not word[0].isalnum():
        prefix += word[0]
        word = word[1:]

    while word and not word[-1].isalnum():
        suffix = word[-1] + suffix
        word = word[:-1]

    lower_word = word.lower()
    corrected = ENGLISH_CORRECTIONS.get(lower_word, lower_word)

    if word.istitle():
        corrected = corrected.title()
    elif word.isupper():
        corrected = corrected.upper()

    return prefix + corrected + suffix


def dictionary_correct_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    masked_text, placeholders = preserve_numbers(text)

    words = masked_text.split()
    corrected_words = []

    for word in words:
        new_word = _replace_english_word(word)

        for wrong, correct in SINHALA_CORRECTIONS.items():
            if wrong in new_word:
                new_word = new_word.replace(wrong, correct)

        corrected_words.append(new_word)

    corrected = " ".join(corrected_words)
    corrected = restore_numbers(corrected, placeholders)
    return corrected


def calculate_confidence(original_text: str, dictionary_text: str, llm_text: str) -> float:
    if not isinstance(original_text, str) or not isinstance(dictionary_text, str) or not isinstance(llm_text, str):
        return 0.0

    score = 1.0

    len_dict = max(len(dictionary_text), 1)
    length_ratio = abs(len(llm_text) - len(dictionary_text)) / len_dict
    if length_ratio > 0.35:
        score -= 0.25

    len_orig = max(len(original_text), 1)
    orig_ratio = abs(len(llm_text) - len(original_text)) / len_orig
    if orig_ratio > 0.5:
        score -= 0.15

    if dictionary_text == llm_text:
        score = min(1.0, score + 0.05)

    return round(max(0.0, min(score, 1.0)), 2)


def hybrid_correct_text(text: str):
    dictionary_text = dictionary_correct_text(text)

    try:
        llm_text = llm_refine_text(dictionary_text)
    except Exception:
        llm_text = dictionary_text

    confidence = calculate_confidence(text, dictionary_text, llm_text)

    return {
        "original_text": text,
        "dictionary_text": dictionary_text,
        "llm_text": llm_text,
        "final_text": llm_text,
        "confidence_score": confidence
    }


def normalize_items(items):
    """
    Converts item list into clean normalized items with line totals.
    Expected input:
    [
      {"description": "...", "quantity": 2, "unit_price": 100},
      ...
    ]
    """
    if not isinstance(items, list):
        return []

    normalized = []

    for item in items:
        description = item.get("description", "")
        quantity = float(item.get("quantity", 0))
        unit_price = float(item.get("unit_price", 0))

        corrected_desc_result = hybrid_correct_text(description)
        corrected_description = corrected_desc_result["final_text"]

        normalized.append({
            "description": corrected_description,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": quantity * unit_price,
            "correction_confidence": corrected_desc_result["confidence_score"]
        })

    return normalized


def summarize_items_for_storage(items):
    """
    Flatten multi-line items into a text summary for CSV storage.
    """
    if not items:
        return ""

    parts = []
    for item in items:
        parts.append(
            f"{item['description']} (qty={item['quantity']}, unit_price={item['unit_price']}, line_total={item['line_total']})"
        )
    return " | ".join(parts)


def correct_record_fields(record: dict) -> tuple[dict, dict]:
    corrected = deepcopy(record)
    correction_log = {}

    text_fields = ["company_name", "supplier_name", "item_description", "raw_text"]

    for field in text_fields:
        old_value = corrected.get(field, "")
        if isinstance(old_value, str):
            result = hybrid_correct_text(old_value)
            new_value = result["final_text"]
            corrected[field] = new_value

            if new_value != old_value:
                correction_log[field] = {
                    "old": old_value,
                    "dictionary_text": result["dictionary_text"],
                    "llm_text": result["llm_text"],
                    "new": new_value,
                    "confidence_score": result["confidence_score"]
                }

    return corrected, correction_log


def validate_totals(record: dict) -> tuple[dict, dict]:
    corrected = deepcopy(record)

    items = corrected.get("items", [])
    raw_total = float(corrected.get("raw_total_amount", 0))

    if isinstance(items, list) and len(items) > 0:
        calculated_total = sum(float(item.get("line_total", 0)) for item in items)
    else:
        quantity = float(corrected.get("quantity", 0))
        unit_price = float(corrected.get("unit_price", 0))
        calculated_total = quantity * unit_price

    total_log = {
        "raw_total_amount": raw_total,
        "calculated_total_amount": calculated_total
    }

    if abs(calculated_total - raw_total) > 0.0001:
        corrected["final_total_amount"] = calculated_total
        corrected["total_status"] = "corrected"
        total_log["status"] = "corrected"
    else:
        corrected["final_total_amount"] = raw_total
        corrected["total_status"] = "valid"
        total_log["status"] = "valid"

    if str(corrected.get("document_type", "")).lower() == "receipt" and str(corrected.get("status", "")).lower() == "paid":
        corrected["payable_amount"] = 0
    else:
        corrected["payable_amount"] = corrected["final_total_amount"]

    total_log["final_total_amount"] = corrected["final_total_amount"]
    total_log["payable_amount"] = corrected["payable_amount"]

    return corrected, total_log


def json_to_record(json_data: dict) -> dict:
    items = normalize_items(json_data.get("items", []))
    item_summary = summarize_items_for_storage(items)

    if items:
        fallback_quantity = sum(item["quantity"] for item in items)
        fallback_unit_price = 0
        fallback_item_description = item_summary
    else:
        fallback_quantity = json_data.get("quantity", 0)
        fallback_unit_price = json_data.get("unit_price", 0)
        fallback_item_description = json_data.get("item_description", "")

    record = {
        "document_id": json_data.get("document_id", ""),
        "document_type": json_data.get("document_type", ""),
        "company_name": json_data.get("company_name", ""),
        "supplier_name": json_data.get("supplier_name", ""),
        "date": json_data.get("date", ""),
        "item_description": fallback_item_description,
        "quantity": fallback_quantity,
        "unit_price": fallback_unit_price,
        "raw_total_amount": json_data.get("raw_total_amount", 0),
        "final_total_amount": json_data.get("raw_total_amount", 0),
        "total_status": "unchecked",
        "payable_amount": json_data.get("raw_total_amount", 0),
        "currency": json_data.get("currency", "LKR"),
        "status": json_data.get("status", "unpaid"),
        "language": json_data.get("language", "english"),
        "raw_text": json_data.get("raw_text", ""),
        "corrected_text": "",
        "source_json": json.dumps(json_data, ensure_ascii=False),
        "correction_confidence": 0.0,
        "correction_log": "",
        "items_json": json.dumps(items, ensure_ascii=False)
    }

    corrected_record, correction_log = correct_record_fields(record)

    raw_text_result = hybrid_correct_text(record.get("raw_text", ""))
    corrected_record["corrected_text"] = raw_text_result["final_text"]
    corrected_record["correction_confidence"] = raw_text_result["confidence_score"]

    corrected_record["items"] = items
    corrected_record, total_log = validate_totals(corrected_record)

    corrected_record["correction_log"] = json.dumps({
        "field_corrections": correction_log,
        "total_check": total_log,
        "items_count": len(items)
    }, ensure_ascii=False)

    corrected_record.pop("items", None)

    return corrected_record