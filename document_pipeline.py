import os
import time
import shutil
from pathlib import Path

import cv2
from pdf2image import convert_from_path

from colab_ocr_client import send_images_to_colab_ocr
from ocr_selector import select_best_ocr_version
from llm_correction import llm_refine_text, clean_ocr_text
from ocr_to_json_extractor import extract_structured_json_from_text

# =========================
# CONFIG
# =========================
COLAB_OCR_URL ="https://catechizable-uncongruously-armani.ngrok-free.dev/"

POPPLER_PATH = r"C:\Users\ASUS\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

TEMP_BASE = Path("temp_processing")
RAW_DIR = TEMP_BASE / "raw"
ORIG_DIR = TEMP_BASE / "pages" / "orig"
P_DIR = TEMP_BASE / "pages" / "P"
M_DIR = TEMP_BASE / "pages" / "M"


# =========================
# DIRECTORY HELPERS
# =========================
def ensure_dirs():
    for d in [RAW_DIR, ORIG_DIR, P_DIR, M_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _safe_remove_file(path: Path, retries: int = 5, delay: float = 0.3):
    for i in range(retries):
        try:
            if path.exists():
                path.unlink()
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(delay)


def _safe_remove_dir_contents(folder: Path):
    if not folder.exists():
        return

    for child in folder.iterdir():
        if child.is_file():
            _safe_remove_file(child)
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def clean_temp_files():
    print("[PIPELINE] Cleaning temp files...", flush=True)
    ensure_dirs()
    for folder in [RAW_DIR, ORIG_DIR, P_DIR, M_DIR]:
        _safe_remove_dir_contents(folder)
    print("[PIPELINE] Temp folders ready", flush=True)


# =========================
# UPLOAD + STANDARDIZATION
# =========================
def save_uploaded_file(upload_path: str) -> Path:
    print("[PIPELINE] Step 1: Saving uploaded file", flush=True)
    clean_temp_files()

    src = Path(upload_path)
    if not src.exists():
        raise FileNotFoundError(f"Uploaded file not found: {upload_path}")

    dst = RAW_DIR / src.name
    shutil.copy(src, dst)

    print(f"[PIPELINE] Raw file saved to: {dst}", flush=True)
    return dst


def standardize_to_image(raw_file_path: Path) -> Path:
    print("[PIPELINE] Step 2: Standardizing input to image", flush=True)
    output_path = ORIG_DIR / "page_001.png"

    if raw_file_path.suffix.lower() == ".pdf":
        print("[PIPELINE] Detected PDF. Converting first page to image...", flush=True)
        if POPPLER_PATH:
            images = convert_from_path(str(raw_file_path), dpi=300, poppler_path=POPPLER_PATH)
        else:
            images = convert_from_path(str(raw_file_path), dpi=300)

        if not images:
            raise ValueError("No pages found in uploaded PDF.")

        images[0].save(output_path, "PNG")
        print(f"[PIPELINE] PDF converted to image: {output_path}", flush=True)
    else:
        shutil.copy(raw_file_path, output_path)
        print(f"[PIPELINE] Image copied as standardized image: {output_path}", flush=True)

    return output_path


# =========================
# PREPROCESSING
# =========================
def preprocess_image(orig_path: Path):
    print("[PIPELINE] Step 3: Creating OCR image variants (orig, P, M)", flush=True)
    p_path = P_DIR / "page_001.png"
    m_path = M_DIR / "page_001.png"

    img = cv2.imread(str(orig_path))
    if img is None:
        raise ValueError("Failed to read standardized image.")

    h, w = img.shape[:2]
    print(f"[PIPELINE] Original image size: {w}x{h}", flush=True)

    target_w = 1200
    if w < target_w:
        scale = target_w / w
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )
        print(f"[PIPELINE] Image upscaled to width {target_w}", flush=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    p_img = cv2.adaptiveThreshold(
        den,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11
    )
    cv2.imwrite(str(p_path), p_img)

    m_img = cv2.bilateralFilter(gray, 7, 50, 50)
    m_img = cv2.normalize(m_img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(m_path), m_img)

    print(f"[PIPELINE] OCR versions saved:", flush=True)
    print(f"           orig -> {orig_path}", flush=True)
    print(f"           P    -> {p_path}", flush=True)
    print(f"           M    -> {m_path}", flush=True)

    return {
        "orig": str(orig_path),
        "P": str(p_path),
        "M": str(m_path),
    }


# =========================
# OCR + LLM PREVIEW
# =========================
def build_preview_from_versions(version_paths: dict) -> dict:
    print("[PIPELINE] Step 4: Sending versions to OCR service", flush=True)
    final_colab_url = COLAB_OCR_URL or os.getenv("COLAB_OCR_URL", "").strip()

    if not final_colab_url:
        raise ValueError("COLAB_OCR_URL is not set. Please set it before running the backend.")

    ocr_result = send_images_to_colab_ocr(
        orig_path=version_paths["orig"],
        p_path=version_paths["P"],
        m_path=version_paths["M"],
        colab_url=final_colab_url,
    )

    print("[PIPELINE] Step 5: OCR response received", flush=True)

    versions = ocr_result.get("versions", {})
    failures = ocr_result.get("failures", {})

    if not versions:
        raise ValueError(f"No OCR versions returned from Colab OCR API. Failures: {failures}")

    print("\n[PIPELINE] Returned OCR versions:", flush=True)
    for k, v in versions.items():
        preview = v.get("text", "")[:250] if isinstance(v, dict) else str(v)[:250]
        print(f"  - {k} preview: {preview}", flush=True)

    print("[PIPELINE] Step 6: Selecting best OCR version", flush=True)
    selection = select_best_ocr_version(versions)

    selected_version = selection["selected_version"]
    selected_text = selection["selected_text"]

    if not selected_text.strip():
        raise ValueError("Selected OCR text is empty.")

    selected_text = clean_ocr_text(selected_text)

    print("\n[PIPELINE] Local OCR selection complete.", flush=True)
    print(f"[PIPELINE] Selected version: {selected_version}", flush=True)
    print(f"[PIPELINE] Scores: {selection['scores']}", flush=True)
    print("[PIPELINE] OCR text preview:", flush=True)
    print(selected_text[:500], flush=True)

    print("\n[PIPELINE] Step 7: Sending selected OCR text to correction LLM...", flush=True)
    corrected_text = llm_refine_text(selected_text)

    print("\n[PIPELINE] Step 8: Sending corrected text to extraction LLM...", flush=True)
    extracted_json = extract_structured_json_from_text(corrected_text)

    extracted_json["document_type"] = str(
        extracted_json.get("document_type", "unknown")
    ).strip().lower() or "unknown"

    extracted_json["corrected_text"] = corrected_text
    extracted_json["raw_text"] = selected_text
    extracted_json["ocr_selected_version"] = selected_version
    extracted_json["ocr_scores"] = selection["scores"]
    extracted_json["ocr_failures"] = failures

    print("[PIPELINE] Step 9: Preview build complete", flush=True)

    return {
        "selected_ocr_version": selected_version,
        "selected_ocr_text": selected_text,
        "corrected_text": corrected_text,
        "extracted_fields": extracted_json
    }


# =========================
# FULL PIPELINE (PREVIEW ONLY)
# =========================
def process_uploaded_document(upload_path: str):
    print("\n----------------------------------------", flush=True)
    print("[PIPELINE] FULL DOCUMENT PIPELINE START", flush=True)
    print(f"[PIPELINE] Input path: {upload_path}", flush=True)

    raw_file = save_uploaded_file(upload_path)
    orig_img = standardize_to_image(raw_file)
    versions = preprocess_image(orig_img)
    preview = build_preview_from_versions(versions)

    print("[PIPELINE] FULL DOCUMENT PIPELINE END", flush=True)
    print("----------------------------------------\n", flush=True)

    return {
        "uploaded_file": str(raw_file),
        "standard_image": str(orig_img),
        "preprocessed_versions": versions,
        "selected_ocr_version": preview["selected_ocr_version"],
        "selected_ocr_text": preview["selected_ocr_text"],
        "corrected_text": preview["corrected_text"],
        "extracted_fields": preview["extracted_fields"],
    }