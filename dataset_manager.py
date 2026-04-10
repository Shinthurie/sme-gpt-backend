import os
import json
import pandas as pd

DATASET_PATH = "financial_documents_clean.csv"
INCOMING_JSON_DIR = "incoming_json"

DATASET_COLUMNS = [
    "document_id",
    "document_type",
    "company_name",
    "supplier_name",
    "date",
    "raw_total_amount",
    "final_total_amount",
    "total_status",
    "payable_amount",
    "currency",
    "status",
    "language",
    "raw_text",
    "corrected_text",
    "structured_json",
    "correction_json"
]


# =========================
# FILE HANDLING
# =========================
def ensure_dataset_exists():
    if not os.path.exists(DATASET_PATH):
        df = pd.DataFrame(columns=DATASET_COLUMNS)
        df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")


def load_main_dataset():
    ensure_dataset_exists()
    return pd.read_csv(DATASET_PATH, keep_default_na=False)


def save_main_dataset(df):
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")


def save_input_json(json_data: dict, filename: str = "last_confirmed.json"):
    os.makedirs(INCOMING_JSON_DIR, exist_ok=True)
    path = os.path.join(INCOMING_JSON_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    return path


# =========================
# CLEANING HELPERS
# =========================
def flatten_text(text):
    if text is None:
        return "NULL"

    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\n", " ")
    text = " ".join(text.split())
    text = text.strip()

    return text if text else "NULL"


def nullify_text(value):
    if value is None:
        return "NULL"

    text = str(value).strip()
    return text if text else "NULL"


def safe_to_float_or_null(value):
    if value is None:
        return "NULL"

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.upper() == "NULL":
        return "NULL"

    text = text.replace(",", "")
    text = text.replace("Rs", "")
    text = text.replace("LKR", "")
    text = text.strip()

    if not text:
        return "NULL"

    try:
        return float(text)
    except Exception:
        return "NULL"


def normalize_compare_text(value):
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = " ".join(text.split())
    return text


def normalize_compare_number(value):
    parsed = safe_to_float_or_null(value)
    if parsed == "NULL":
        return "NULL"
    return float(parsed)


# =========================
# DOCUMENT ID GENERATION
# =========================
def get_prefix_for_document_type(document_type: str) -> str:
    if not isinstance(document_type, str):
        return "DOC"

    doc_type = document_type.lower().strip()

    prefix_map = {
        "receipt": "R",
        "invoice": "IN",
        "po": "PO",
        "dn": "DN",
    }

    return prefix_map.get(doc_type, "DOC")


def generate_document_id(document_type: str) -> str:
    df = load_main_dataset()
    prefix = get_prefix_for_document_type(document_type)

    if df.empty or "document_id" not in df.columns:
        return f"{prefix}1"

    existing_ids = df["document_id"].astype(str).tolist()
    numbers = []

    for doc_id in existing_ids:
        doc_id = str(doc_id).strip()
        if doc_id.startswith(prefix):
            num_part = doc_id[len(prefix):]
            if num_part.isdigit():
                numbers.append(int(num_part))

    next_number = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_number}"


# =========================
# NORMALIZATION
# =========================
def normalize_record(data: dict, force_generate_document_id: bool = True) -> dict:
    document_type = nullify_text(data.get("document_type", None))
    if document_type == "NULL":
        document_type = "unknown"

    if force_generate_document_id:
        document_id = generate_document_id(document_type)
    else:
        document_id = nullify_text(data.get("document_id", None))

    raw_text = flatten_text(data.get("raw_text", None))
    corrected_text = flatten_text(data.get("corrected_text", None))

    raw_total = safe_to_float_or_null(data.get("raw_total_amount", None))
    final_total = safe_to_float_or_null(data.get("final_total_amount", None))
    payable_amount = safe_to_float_or_null(data.get("payable_amount", None))

    if raw_total == "NULL" or final_total == "NULL":
        total_status = "NULL"
    else:
        total_status = "corrected" if final_total != raw_total else "original"

    structured_json = json.dumps(data, ensure_ascii=False).replace("\n", " ")
    correction_json = json.dumps({
        "total_check": {
            "raw_total_amount": raw_total,
            "final_total_amount": final_total,
            "status": total_status,
            "payable_amount": payable_amount
        }
    }, ensure_ascii=False).replace("\n", " ")

    return {
        "document_id": document_id,
        "document_type": document_type,
        "company_name": nullify_text(data.get("company_name", None)),
        "supplier_name": nullify_text(data.get("supplier_name", None)),
        "date": nullify_text(data.get("date", None)),
        "raw_total_amount": raw_total,
        "final_total_amount": final_total,
        "total_status": total_status,
        "payable_amount": payable_amount,
        "currency": nullify_text(data.get("currency", None)),
        "status": nullify_text(data.get("status", None)),
        "language": nullify_text(data.get("language", None)),
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "structured_json": structured_json if structured_json.strip() else "NULL",
        "correction_json": correction_json if correction_json.strip() else "NULL",
    }


# =========================
# DUPLICATE CHECK
# =========================
def is_exact_duplicate(existing_row: dict, new_record: dict) -> bool:
    compare_text_fields = [
        "document_type",
        "company_name",
        "supplier_name",
        "date",
        "currency",
        "raw_text",
    ]

    compare_number_fields = [
        "raw_total_amount",
        "final_total_amount",
        "payable_amount",
    ]

    for field in compare_text_fields:
        if normalize_compare_text(existing_row.get(field, "")) != normalize_compare_text(new_record.get(field, "")):
            return False

    for field in compare_number_fields:
        if normalize_compare_number(existing_row.get(field, "NULL")) != normalize_compare_number(new_record.get(field, "NULL")):
            return False

    return True


def find_duplicate_record(record_dict: dict):
    df = load_main_dataset()
    new_record = normalize_record(record_dict, force_generate_document_id=False)

    if df.empty:
        return None

    rows = df.to_dict(orient="records")
    for row in rows:
        if is_exact_duplicate(row, new_record):
            return row

    return None


# =========================
# UPSERT / SAVE
# =========================
def upsert_confirmed_record(record_dict: dict):
    ensure_dataset_exists()
    df = load_main_dataset()

    record = normalize_record(record_dict, force_generate_document_id=True)

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save_main_dataset(df)

    return {
        "action": "inserted",
        "record": record
    }


# =========================
# READ HELPERS FOR FRONTEND
# =========================
def _format_amount(value):
    parsed = safe_to_float_or_null(value)
    if parsed == "NULL":
        return "NULL"
    return f"{float(parsed):.2f}"


def load_all_records():
    df = load_main_dataset()
    if df.empty:
        return []

    records = df.to_dict(orient="records")
    cleaned = []

    for row in records:
        structured_json = row.get("structured_json", "NULL")
        parsed_structured = {}

        if structured_json and structured_json != "NULL":
            try:
                parsed_structured = json.loads(structured_json)
            except Exception:
                parsed_structured = {}

        cleaned.append({
            "document_id": row.get("document_id", "NULL"),
            "document_type": row.get("document_type", "unknown"),
            "company_name": row.get("company_name", "NULL"),
            "supplier_name": row.get("supplier_name", "NULL"),
            "date": row.get("date", "NULL"),
            "raw_total_amount": _format_amount(row.get("raw_total_amount", "NULL")),
            "final_total_amount": _format_amount(row.get("final_total_amount", "NULL")),
            "payable_amount": _format_amount(row.get("payable_amount", "NULL")),
            "currency": row.get("currency", "NULL"),
            "status": row.get("status", "NULL"),
            "language": row.get("language", "NULL"),
            "items": parsed_structured.get("items", []),
            "order_id": parsed_structured.get("order_id", "NULL"),
            "flow_type": parsed_structured.get("flow_type", "NULL"),
            "received_status": parsed_structured.get("received_status", "NULL"),
            "paid_status": parsed_structured.get("paid_status", "NULL"),
        })

    cleaned.reverse()
    return cleaned