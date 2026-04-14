import os
import json
import requests

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
        timeout=600,
    )

    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def build_fallback_answer(question: str, company_name: str, result: dict) -> str:
    if not result.get("success"):
        return result.get("explanation", "No answer could be generated.")

    qtype = result.get("question_type", "summary")
    metrics = result.get("metrics", {})
    evidence = result.get("evidence", [])

    if qtype == "receivable":
        total = metrics.get("total_receivable_amount", 0.0)
        doc_count = metrics.get("receivable_documents", 0)
        doc_ids = ", ".join([e["document_id"] for e in evidence]) if evidence else "none"
        return (
            f"For company {company_name}, the current receivable amount is {total:.2f}. "
            f"This was calculated using {doc_count} receivable document(s) from financial_documents_clean.csv. "
            f"Evidence documents: {doc_ids}."
        )

    if qtype == "payable":
        total = metrics.get("total_payable_amount", 0.0)
        doc_count = metrics.get("payable_documents", 0)
        doc_ids = ", ".join([e["document_id"] for e in evidence]) if evidence else "none"
        return (
            f"For company {company_name}, the current payable amount is {total:.2f}. "
            f"This was calculated using {doc_count} payable document(s) from financial_documents_clean.csv. "
            f"Evidence documents: {doc_ids}."
        )

    if qtype in ["invoice_list", "receipt_list", "po_list", "dn_list"]:
        doc_ids = ", ".join([e["document_id"] for e in evidence]) if evidence else "none"
        return (
            f"I found these matching documents for company {company_name}: {doc_ids}. "
            f"All results were taken only from financial_documents_clean.csv."
        )

    return (
        f"I generated this answer for company {company_name} using only financial_documents_clean.csv. "
        f"{len(evidence)} matching record(s) were used as evidence."
    )


def generate_explainable_answer(question: str, company_name: str, result: dict) -> str:
    if not result.get("success"):
        return result.get("explanation", "No answer available.")

    prompt = f"""
You are a financial assistant.

Rules:
- Use ONLY the provided analysis result
- Do NOT invent numbers
- Do NOT invent document IDs
- Mention that the answer is based only on financial_documents_clean.csv
- Mention why the evidence documents were included
- If a total is aggregated, explicitly say it is an aggregated total
- Keep the answer clear and business-friendly
- Do not mention any hidden source or external memory

Company Context:
{company_name}

User Question:
{question}

Analysis Result:
{json.dumps(result, ensure_ascii=False, indent=2)}

Answer:
""".strip()

    try:
        response = call_ollama(prompt)
        return response if response else build_fallback_answer(question, company_name, result)
    except Exception:
        return build_fallback_answer(question, company_name, result)