import os
import shutil
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path

import jwt
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from document_pipeline import process_uploaded_document
from dataset_manager import (
    upsert_confirmed_record,
    save_input_json,
    find_duplicate_record,
    load_all_records,
    get_record_by_id_for_user,
)
from ai_helper import generate_explainable_answer
from data_tools import analyze_financial_query

JWT_SECRET = "your_super_secret_key_123"
JWT_ALGORITHM = "HS256"

app = FastAPI(title="SME-GPT Financial Document Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSING_SESSIONS = {}

SAVED_DOCS_DIR = Path("saved_documents")
SAVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/saved-documents", StaticFiles(directory=str(SAVED_DOCS_DIR)), name="saved-documents")


def get_current_user_id(authorization: str = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format.")

    token = authorization.replace("Bearer ", "").strip()

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("userId")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload.")

        return str(user_id)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


def to_preview_data(fields: dict) -> dict:
    return {
        "document_type": fields.get("document_type", "NULL") or "NULL",
        "order_id": fields.get("order_id", "NULL") or "NULL",
        "flow_type": fields.get("flow_type", "NULL") or "NULL",
        "company_name": fields.get("company_name", "NULL") or "NULL",
        "supplier_name": fields.get("supplier_name", "NULL") or "NULL",
        "date": fields.get("date", "NULL") or "NULL",
        "currency": fields.get("currency", "NULL") or "NULL",
        "raw_total_amount": fields.get("raw_total_amount", "NULL"),
        "final_total_amount": fields.get("final_total_amount", "NULL"),
        "payable_amount": fields.get("payable_amount", "NULL"),
        "cash_return": fields.get("cash_return", "NULL"),
        "received_status": fields.get("received_status", "NULL") or "NULL",
        "paid_status": fields.get("paid_status", "NULL") or "NULL",
        "items": fields.get("items", []),
    }


def merge_edited_preview_into_fields(original_fields: dict, edited_preview: dict) -> dict:
    merged = deepcopy(original_fields)

    editable_keys = [
        "document_type",
        "order_id",
        "flow_type",
        "company_name",
        "supplier_name",
        "date",
        "currency",
        "raw_total_amount",
        "final_total_amount",
        "payable_amount",
        "cash_return",
        "received_status",
        "paid_status",
        "items",
    ]

    for key in editable_keys:
        if key in edited_preview:
            merged[key] = edited_preview[key]

    merged["status"] = "confirmed"
    return merged


def get_saved_image_url(document_id: str):
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = SAVED_DOCS_DIR / f"{document_id}{ext}"
        if path.exists():
            return f"/saved-documents/{path.name}"
    return None


def save_document_image_from_session(session_meta: dict, document_id: str):
    src = session_meta.get("standard_image")
    if not src:
        return None

    src_path = Path(src)
    if not src_path.exists():
        return None

    ext = src_path.suffix.lower() if src_path.suffix else ".png"
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        ext = ".png"

    dst = SAVED_DOCS_DIR / f"{document_id}{ext}"
    shutil.copy(src_path, dst)
    return f"/saved-documents/{dst.name}"


def build_document_detail(user_id: str, document_id: str):
    record = get_record_by_id_for_user(user_id=user_id, document_id=document_id)

    if not record:
        return None

    return {
        **record,
        "image_url": get_saved_image_url(document_id),
    }


class ConfirmSaveRequest(BaseModel):
    session_id: str
    edited_preview: dict
    force_save: bool = False


class QueryRequest(BaseModel):
    company_name: str
    question: str


@app.get("/health")
def health():
    print("[HEALTH] /health checked", flush=True)
    return {
        "success": True,
        "message": "Backend is running."
    }


@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    authorization: str = Header(default=None),
):
    print("\n==============================", flush=True)
    print("[API] /process-document START", flush=True)

    _ = get_current_user_id(authorization)
    temp_dir = tempfile.mkdtemp(prefix="smegpt_")

    try:
        if not file.filename:
            print("[API] No file uploaded", flush=True)
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No file uploaded."}
            )

        print(f"[API] Received file: {file.filename}", flush=True)

        ext = Path(file.filename).suffix.lower()
        allowed_exts = {".pdf", ".png", ".jpg", ".jpeg"}

        if ext not in allowed_exts:
            print(f"[API] Unsupported file type: {ext}", flush=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Unsupported file type. Use PDF, PNG, JPG, or JPEG."
                }
            )

        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[API] Temp file saved: {temp_file_path}", flush=True)
        print("[API] Launching document pipeline...", flush=True)

        result = process_uploaded_document(temp_file_path)

        print("[API] Document pipeline completed successfully", flush=True)

        fields = result["extracted_fields"]
        preview = to_preview_data(fields)

        session_id = str(uuid.uuid4())
        PROCESSING_SESSIONS[session_id] = {
            "fields": fields,
            "preview": preview,
            "meta": {
                "uploaded_file": result.get("uploaded_file"),
                "standard_image": result.get("standard_image"),
                "selected_ocr_version": result.get("selected_ocr_version"),
            }
        }

        print(f"[API] Session created: {session_id}", flush=True)
        print("[API] /process-document END", flush=True)
        print("==============================\n", flush=True)

        return {
            "success": True,
            "message": "Document processed successfully.",
            "session_id": session_id,
            "preview": preview,
            "meta": {
                "uploaded_file": result.get("uploaded_file"),
                "standard_image": result.get("standard_image"),
                "selected_ocr_version": result.get("selected_ocr_version"),
            }
        }

    except Exception as e:
        import traceback
        print("[API] ERROR in /process-document", flush=True)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error while processing document: {str(e)}"
            }
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[API] Temp folder cleaned: {temp_dir}", flush=True)


@app.post("/confirm-save")
def confirm_save(
    payload: ConfirmSaveRequest,
    authorization: str = Header(default=None),
):
    print("\n==============================", flush=True)
    print("[API] /confirm-save START", flush=True)

    try:
        user_id = get_current_user_id(authorization)
        session = PROCESSING_SESSIONS.get(payload.session_id)

        if not session:
            print("[API] Session missing or expired", flush=True)
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Session expired or not found. Please process the document again."
                }
            )

        print(f"[API] Saving session: {payload.session_id}", flush=True)
        original_fields = session["fields"]
        final_data = merge_edited_preview_into_fields(original_fields, payload.edited_preview)

        print("[API] Checking duplicates...", flush=True)
        duplicate = find_duplicate_record(final_data, user_id=user_id)

        if duplicate and not payload.force_save:
            print(f"[API] Duplicate found: {duplicate.get('document_id', 'NULL')}", flush=True)
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "duplicate_found": True,
                    "message": "Already we have this document.",
                    "existing_document_id": duplicate.get("document_id", "NULL")
                }
            )

        print("[API] Saving input JSON...", flush=True)
        save_input_json(final_data, "last_confirmed.json")

        print("[API] Writing record to dataset...", flush=True)
        save_result = upsert_confirmed_record(final_data, user_id=user_id)

        document_id = save_result["record"]["document_id"]
        image_url = save_document_image_from_session(session["meta"], document_id)

        PROCESSING_SESSIONS.pop(payload.session_id, None)

        print(f"[API] Saved document: {document_id}", flush=True)
        print("[API] /confirm-save END", flush=True)
        print("==============================\n", flush=True)

        return {
            "success": True,
            "duplicate_found": bool(duplicate),
            "message": "Document saved successfully.",
            "document_id": document_id,
            "image_url": image_url,
            "action": save_result["action"],
            "record": save_result["record"]
        }

    except HTTPException as http_err:
        print(f"[API] Auth/HTTP error in /confirm-save: {http_err.detail}", flush=True)
        return JSONResponse(
            status_code=http_err.status_code,
            content={
                "success": False,
                "message": http_err.detail,
            }
        )
    except Exception as e:
        import traceback
        print("[API] ERROR in /confirm-save", flush=True)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error while saving document: {str(e)}"
            }
        )


@app.get("/documents")
def get_documents(authorization: str = Header(default=None)):
    print("[API] /documents requested", flush=True)
    try:
        user_id = get_current_user_id(authorization)
        records = load_all_records(user_id=user_id)

        print(f"[API] Returned {len(records)} documents", flush=True)
        return {
            "success": True,
            "documents": records
        }
    except HTTPException as http_err:
        print(f"[API] Auth/HTTP error in /documents: {http_err.detail}", flush=True)
        return JSONResponse(
            status_code=http_err.status_code,
            content={"success": False, "message": http_err.detail}
        )


@app.get("/documents/{document_id}")
def get_document_by_id(document_id: str, authorization: str = Header(default=None)):
    print(f"[API] /documents/{document_id} requested", flush=True)
    try:
        user_id = get_current_user_id(authorization)
        document = build_document_detail(user_id=user_id, document_id=document_id)

        if not document:
            print(f"[API] Document not found for user: {document_id}", flush=True)
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Document not found."
                }
            )

        print(f"[API] Document returned: {document_id}", flush=True)
        return {
            "success": True,
            "document": document
        }
    except HTTPException as http_err:
        print(f"[API] Auth/HTTP error in /documents/{{id}}: {http_err.detail}", flush=True)
        return JSONResponse(
            status_code=http_err.status_code,
            content={"success": False, "message": http_err.detail}
        )


@app.get("/dashboard-summary")
def dashboard_summary(authorization: str = Header(default=None)):
    print("[API] /dashboard-summary requested", flush=True)
    try:
        user_id = get_current_user_id(authorization)
        records = load_all_records(user_id=user_id)

        total = len(records)
        invoice = sum(1 for r in records if str(r.get("document_type", "")).lower() == "invoice")
        receipt = sum(1 for r in records if str(r.get("document_type", "")).lower() == "receipt")
        po = sum(1 for r in records if str(r.get("document_type", "")).lower() == "po")
        dn = sum(1 for r in records if str(r.get("document_type", "")).lower() == "dn")

        recent_documents = list(reversed(records[-4:]))

        print(
            f"[API] Summary -> total={total}, invoice={invoice}, receipt={receipt}, po={po}, dn={dn}",
            flush=True
        )

        return {
            "success": True,
            "total": total,
            "invoice": invoice,
            "receipt": receipt,
            "po": po,
            "dn": dn,
            "recent_documents": recent_documents
        }
    except HTTPException as http_err:
        print(f"[API] Auth/HTTP error in /dashboard-summary: {http_err.detail}", flush=True)
        return JSONResponse(
            status_code=http_err.status_code,
            content={"success": False, "message": http_err.detail}
        )


@app.post("/ask-query")
def ask_query(payload: QueryRequest):
    print("[API] /ask-query START", flush=True)
    try:
        company_name = (payload.company_name or "").strip()
        question = (payload.question or "").strip()

        if not company_name:
            print("[API] Query rejected: missing company name", flush=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Company name is required before asking a question."
                }
            )

        if not question:
            print("[API] Query rejected: missing question", flush=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Question is required."
                }
            )

        print(f"[API] Company: {company_name}", flush=True)
        print(f"[API] Question: {question}", flush=True)

        analysis_result = analyze_financial_query(question, company_name)
        final_answer = generate_explainable_answer(question, company_name, analysis_result)

        print("[API] /ask-query END", flush=True)

        return {
            "success": analysis_result.get("success", False),
            "company_name": company_name,
            "question": question,
            "answer": final_answer,
            "explanation": analysis_result.get("explanation", ""),
            "evidence": analysis_result.get("evidence", []),
            "metrics": analysis_result.get("metrics", {}),
            "source_file": analysis_result.get("source_file", "financial_documents_clean.csv"),
        }

    except Exception as e:
        import traceback
        print("[API] ERROR in /ask-query", flush=True)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error while answering query: {str(e)}"
            }
        )


if __name__ == "__main__":
    import uvicorn
    print("[BOOT] Starting backend with python app.py", flush=True)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)