import os
import shutil
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
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
)

app = FastAPI(title="SME-GPT Financial Document Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSING_SESSIONS = {}

SAVED_DOCS_DIR = Path("saved_documents")
SAVED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/saved-documents", StaticFiles(directory=str(SAVED_DOCS_DIR)), name="saved-documents")


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


def build_document_detail(document_id: str):
    records = load_all_records()

    for record in records:
        if record.get("document_id") == document_id:
            detail = {
                **record,
                "image_url": get_saved_image_url(document_id),
            }
            return detail

    return None


class ConfirmSaveRequest(BaseModel):
    session_id: str
    edited_preview: dict
    force_save: bool = False


@app.get("/health")
def health():
    return {
        "success": True,
        "message": "Backend is running."
    }


@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp(prefix="smegpt_")

    try:
        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No file uploaded."}
            )

        ext = Path(file.filename).suffix.lower()
        allowed_exts = {".pdf", ".png", ".jpg", ".jpeg"}

        if ext not in allowed_exts:
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

        result = process_uploaded_document(temp_file_path)
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
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error while processing document: {str(e)}"
            }
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/confirm-save")
def confirm_save(payload: ConfirmSaveRequest):
    try:
        session = PROCESSING_SESSIONS.get(payload.session_id)

        if not session:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Session expired or not found. Please process the document again."
                }
            )

        original_fields = session["fields"]
        final_data = merge_edited_preview_into_fields(original_fields, payload.edited_preview)

        duplicate = find_duplicate_record(final_data)

        if duplicate and not payload.force_save:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "duplicate_found": True,
                    "message": "Already we have this document.",
                    "existing_document_id": duplicate.get("document_id", "NULL")
                }
            )

        save_input_json(final_data, "last_confirmed.json")
        save_result = upsert_confirmed_record(final_data)

        document_id = save_result["record"]["document_id"]
        image_url = save_document_image_from_session(session["meta"], document_id)

        PROCESSING_SESSIONS.pop(payload.session_id, None)

        return {
            "success": True,
            "duplicate_found": bool(duplicate),
            "message": "Document saved successfully.",
            "document_id": document_id,
            "image_url": image_url,
            "action": save_result["action"],
            "record": save_result["record"]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error while saving document: {str(e)}"
            }
        )


@app.get("/documents")
def get_documents():
    records = load_all_records()
    return {
        "success": True,
        "documents": records
    }


@app.get("/documents/{document_id}")
def get_document_by_id(document_id: str):
    document = build_document_detail(document_id)

    if not document:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Document not found."
            }
        )

    return {
        "success": True,
        "document": document
    }


@app.get("/dashboard-summary")
def dashboard_summary():
    records = load_all_records()

    total = len(records)
    invoice = sum(1 for r in records if r.get("document_type") == "invoice")
    receipt = sum(1 for r in records if r.get("document_type") == "receipt")
    po = sum(1 for r in records if r.get("document_type") == "po")
    dn = sum(1 for r in records if r.get("document_type") == "dn")

    recent_documents = records[:4]

    return {
        "success": True,
        "total": total,
        "invoice": invoice,
        "receipt": receipt,
        "po": po,
        "dn": dn,
        "recent_documents": recent_documents
    }