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
    print("[CORRECTION_ENGINE] Running hybrid text correction...", flush=True)
    dictionary_text = dictionary_correct_text(text)

    try:
        llm_text = llm_refine_text(dictionary_text)
    except Exception as e:
        print(f"[CORRECTION_ENGINE] LLM correction failed, using dictionary text. Error: {e}", flush=True)
        llm_text = dictionary_text

    confidence = calculate_confidence(text, dictionary_text, llm_text)

    print(f"[CORRECTION_ENGINE] Hybrid correction complete. Confidence={confidence}", flush=True)
    return {
        "original_text": text,
        "dictionary_text": dictionary_text,
        "llm_text": llm_text,
        "final_text": llm_text,
        "confidence_score": confidence
    }


def normalize_items(items):
    print("[CORRECTION_ENGINE] Normalizing items...", flush=True)

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

    print(f"[CORRECTION_ENGINE] Normalized {len(normalized)} items", flush=True)
    return normalized


def summarize_items_for_storage(items):
    if not isinstance(items, list) or not items:
        return "NULL"

    lines = []
    for item in items:
        description = item.get("description", "")
        quantity = item.get("quantity", 0)
        unit_price = item.get("unit_price", 0)
        line_total = item.get("line_total", 0)

        lines.append(
            f"{description} | qty={quantity} | unit_price={unit_price} | line_total={line_total}"
        )

    return " ; ".join(lines)


def recalculate_totals(items):
    if not isinstance(items, list):
        return 0.0

    total = 0.0
    for item in items:
        try:
            total += float(item.get("line_total", 0))
        except Exception:
            pass

    return round(total, 2)


def correct_extracted_fields(extracted_json: dict):
    print("[CORRECTION_ENGINE] Correcting extracted fields...", flush=True)

    data = deepcopy(extracted_json)

    items = data.get("items", [])
    if not isinstance(items, list):
        items = []

    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue

        normalized_items.append({
            "description": str(item.get("description", "")).strip(),
            "quantity": item.get("quantity", 0),
            "unit_price": item.get("unit_price", 0),
        })

    data["items"] = normalized_items
    data["correction_status"] = "text_only_preserved"
    data["total_status"] = "original"
    data["correction_confidence"] = 1.0

    print("[CORRECTION_ENGINE] Field correction completed with numeric preservation", flush=True)
    return data