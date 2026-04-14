import os
import json
import pandas as pd

DATASET_PATH = "financial_documents_clean.csv"


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found.")

    df = pd.read_csv(DATASET_PATH, keep_default_na=False)
    return enrich_dataset(df)


def safe_json_load(value):
    if not value or str(value).strip() in ["", "NULL", "null", "None"]:
        return {}

    try:
        return json.loads(value)
    except Exception:
        return {}


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def to_float(value):
    if value is None:
        return 0.0

    text = str(value).strip()
    if text == "" or text.upper() == "NULL":
        return 0.0

    text = text.replace(",", "").replace("Rs", "").replace("LKR", "").strip()

    try:
        return float(text)
    except Exception:
        return 0.0


def enrich_dataset(df: pd.DataFrame):
    if df.empty:
        for col in [
            "order_id", "flow_type", "received_status", "paid_status", "items"
        ]:
            df[col] = []
        return df

    structured = df["structured_json"].apply(safe_json_load) if "structured_json" in df.columns else pd.Series([{}] * len(df))

    df = df.copy()
    df["order_id"] = structured.apply(lambda x: x.get("order_id", "NULL"))
    df["flow_type"] = structured.apply(lambda x: x.get("flow_type", "unknown"))
    df["received_status"] = structured.apply(lambda x: x.get("received_status", "NULL"))
    df["paid_status"] = structured.apply(lambda x: x.get("paid_status", "NULL"))
    df["items"] = structured.apply(lambda x: x.get("items", []))

    numeric_cols = ["raw_total_amount", "final_total_amount", "payable_amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_float)

    return df


def filter_company_context(df: pd.DataFrame, company_name: str):
    target = normalize_text(company_name)
    if not target:
        return df.iloc[0:0].copy()

    return df[df["company_name"].apply(normalize_text) == target].copy()


def route_question(question: str):
    q = normalize_text(question)

    if any(term in q for term in ["receivable", "receive", "receiver", "received amount", "අයකරගත", "ලැබිය යුතු"]):
        return "receivable"

    if any(term in q for term in ["payable", "pay", "amount due", "ගෙවිය යුතු"]):
        return "payable"

    if any(term in q for term in ["invoice", "invoices", "show invoices"]):
        return "invoice_list"

    if any(term in q for term in ["receipt", "receipts", "show receipts"]):
        return "receipt_list"

    if any(term in q for term in ["po", "purchase order"]):
        return "po_list"

    if any(term in q for term in ["dn", "delivery note"]):
        return "dn_list"

    return "summary"


def build_evidence(records: pd.DataFrame, reason: str):
    evidence = []

    for _, row in records.iterrows():
        evidence.append({
            "document_id": row.get("document_id", "NULL"),
            "document_type": row.get("document_type", "NULL"),
            "date": row.get("date", "NULL"),
            "company_name": row.get("company_name", "NULL"),
            "supplier_name": row.get("supplier_name", "NULL"),
            "order_id": row.get("order_id", "NULL"),
            "flow_type": row.get("flow_type", "unknown"),
            "received_status": row.get("received_status", "NULL"),
            "paid_status": row.get("paid_status", "NULL"),
            "final_total_amount": float(row.get("final_total_amount", 0) or 0),
            "payable_amount": float(row.get("payable_amount", 0) or 0),
            "reason_used": reason,
        })

    return evidence

def normalize_flow(value):
    if not value:
        return "unknown"

    v = str(value).strip().lower()

    if v in ["receivable", "receive"]:
        return "receivable"
    if v in ["payable", "pay"]:
        return "payable"
    if v in ["income"]:
        return "income"
    if v in ["expense", "expence"]:
        return "expense"

    return "unknown"
def analyze_financial_query(question: str, company_name: str):
    df = load_dataset()
    company_df = filter_company_context(df, company_name)

    if company_df.empty:
        return {
            "success": False,
            "question_type": "none",
            "explanation": f"No records found in financial_documents_clean.csv for company '{company_name}'.",
            "evidence": [],
            "metrics": {},
            "source_file": DATASET_PATH,
        }

    question_type = route_question(question)

    if question_type == "receivable":
        receivable_df = company_df[company_df["flow_type"].apply(normalize_text) == "receivable"].copy()

        if "received_status" in receivable_df.columns:
            outstanding_df = receivable_df[
                receivable_df["received_status"].apply(normalize_text) != "completed"
            ].copy()
        else:
            outstanding_df = receivable_df.copy()

        amount_col = "payable_amount" if "payable_amount" in outstanding_df.columns else "final_total_amount"
        total_amount = float(outstanding_df[amount_col].sum()) if not outstanding_df.empty else 0.0

        return {
            "success": True,
            "question_type": "receivable",
            "explanation": f"Computed receivable amount for company '{company_name}' using receivable documents from financial_documents_clean.csv.",
            "evidence": build_evidence(
                outstanding_df,
                "Included because company_name matched the selected company context and flow_type is receivable."
            ),
            "metrics": {
                "company_name": company_name,
                "matching_records": int(len(company_df)),
                "receivable_documents": int(len(outstanding_df)),
                "total_receivable_amount": total_amount,
            },
            "source_file": DATASET_PATH,
        }

    if question_type == "payable":
        payable_df = company_df[company_df["flow_type"].apply(normalize_text) == "payable"].copy()

        if "paid_status" in payable_df.columns:
            outstanding_df = payable_df[
                payable_df["paid_status"].apply(normalize_text) != "completed"
            ].copy()
        else:
            outstanding_df = payable_df.copy()

        amount_col = "payable_amount" if "payable_amount" in outstanding_df.columns else "final_total_amount"
        total_amount = float(outstanding_df[amount_col].sum()) if not outstanding_df.empty else 0.0

        return {
            "success": True,
            "question_type": "payable",
            "explanation": f"Computed payable amount for company '{company_name}' using payable documents from financial_documents_clean.csv.",
            "evidence": build_evidence(
                outstanding_df,
                "Included because company_name matched the selected company context and flow_type is payable."
            ),
            "metrics": {
                "company_name": company_name,
                "matching_records": int(len(company_df)),
                "payable_documents": int(len(outstanding_df)),
                "total_payable_amount": total_amount,
            },
            "source_file": DATASET_PATH,
        }

    if question_type == "invoice_list":
        result_df = company_df[company_df["document_type"].apply(normalize_text) == "invoice"].copy()

        return {
            "success": True,
            "question_type": "invoice_list",
            "explanation": f"Listed invoices for company '{company_name}' from financial_documents_clean.csv.",
            "evidence": build_evidence(
                result_df,
                "Included because company_name matched the selected company context and document_type is invoice."
            ),
            "metrics": {
                "company_name": company_name,
                "invoice_count": int(len(result_df)),
            },
            "source_file": DATASET_PATH,
        }

    if question_type == "receipt_list":
        result_df = company_df[company_df["document_type"].apply(normalize_text) == "receipt"].copy()

        return {
            "success": True,
            "question_type": "receipt_list",
            "explanation": f"Listed receipts for company '{company_name}' from financial_documents_clean.csv.",
            "evidence": build_evidence(
                result_df,
                "Included because company_name matched the selected company context and document_type is receipt."
            ),
            "metrics": {
                "company_name": company_name,
                "receipt_count": int(len(result_df)),
            },
            "source_file": DATASET_PATH,
        }

    if question_type == "po_list":
        result_df = company_df[company_df["document_type"].apply(normalize_text) == "po"].copy()

        return {
            "success": True,
            "question_type": "po_list",
            "explanation": f"Listed purchase orders for company '{company_name}' from financial_documents_clean.csv.",
            "evidence": build_evidence(
                result_df,
                "Included because company_name matched the selected company context and document_type is po."
            ),
            "metrics": {
                "company_name": company_name,
                "po_count": int(len(result_df)),
            },
            "source_file": DATASET_PATH,
        }

    if question_type == "dn_list":
        result_df = company_df[company_df["document_type"].apply(normalize_text) == "dn"].copy()

        return {
            "success": True,
            "question_type": "dn_list",
            "explanation": f"Listed delivery notes for company '{company_name}' from financial_documents_clean.csv.",
            "evidence": build_evidence(
                result_df,
                "Included because company_name matched the selected company context and document_type is dn."
            ),
            "metrics": {
                "company_name": company_name,
                "dn_count": int(len(result_df)),
            },
            "source_file": DATASET_PATH,
        }

    return {
        "success": True,
        "question_type": "summary",
        "explanation": f"Generated summary for company '{company_name}' from financial_documents_clean.csv.",
        "evidence": build_evidence(
            company_df.head(10),
            "Included because company_name matched the selected company context."
        ),
        "metrics": {
            "company_name": company_name,
            "matching_records": int(len(company_df)),
            "invoice_count": int((company_df["document_type"].apply(normalize_text) == "invoice").sum()),
            "receipt_count": int((company_df["document_type"].apply(normalize_text) == "receipt").sum()),
            "po_count": int((company_df["document_type"].apply(normalize_text) == "po").sum()),
            "dn_count": int((company_df["document_type"].apply(normalize_text) == "dn").sum()),
            "total_final_amount": float(company_df["final_total_amount"].sum()) if "final_total_amount" in company_df.columns else 0.0,
        },
        "source_file": DATASET_PATH,
    }