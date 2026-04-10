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
COLAB_OCR_URL = None  
POPPLER_PATH = None

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
    ensure_dirs()
    for folder in [RAW_DIR, ORIG_DIR, P_DIR, M_DIR]:
        _safe_remove_dir_contents(folder)


# =========================
# UPLOAD + STANDARDIZATION
# =========================
def save_uploaded_file(upload_path: str) -> Path:
    clean_temp_files()

    src = Path(upload_path)
    if not src.exists():
        raise FileNotFoundError(f"Uploaded file not found: {upload_path}")

    dst = RAW_DIR / src.name
    shutil.copy(src, dst)
    return dst


def standardize_to_image(raw_file_path: Path) -> Path:
    output_path = ORIG_DIR / "page_001.png"

    if raw_file_path.suffix.lower() == ".pdf":
        if POPPLER_PATH:
            images = convert_from_path(str(raw_file_path), dpi=300, poppler_path=POPPLER_PATH)
        else:
            images = convert_from_path(str(raw_file_path), dpi=300)

        if not images:
            raise ValueError("No pages found in uploaded PDF.")

        images[0].save(output_path, "PNG")
    else:
        shutil.copy(raw_file_path, output_path)

    return output_path


# =========================
# PREPROCESSING
# =========================
def preprocess_image(orig_path: Path):
    p_path = P_DIR / "page_001.png"
    m_path = M_DIR / "page_001.png"

    img = cv2.imread(str(orig_path))
    if img is None:
        raise ValueError("Failed to read standardized image.")

    h, w = img.shape[:2]
    target_w = 1200
    if w < target_w:
        scale = target_w / w
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

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

    return {
        "orig": str(orig_path),
        "P": str(p_path),
        "M": str(m_path),
    }


# =========================
# OCR + LLM PREVIEW
# =========================
def build_preview_from_versions(version_paths: dict) -> dict:

    ocr_result = send_images_to_colab_ocr(
        orig_path=version_paths["orig"],
        p_path=version_paths["P"],
        m_path=version_paths["M"],
        colab_url=None
    )

    versions = ocr_result.get("versions", {})
    failures = ocr_result.get("failures", {})

    if not versions:
        raise ValueError(f"No OCR versions returned from Colab OCR API. Failures: {failures}")

    print("\nReturned OCR versions:")
    for k, v in versions.items():
        preview = v.get("text", "")[:250] if isinstance(v, dict) else str(v)[:250]
        print(f"{k} preview: {preview}")

    selection = select_best_ocr_version(versions)

    selected_version = selection["selected_version"]
    selected_text = selection["selected_text"]

    if not selected_text.strip():
        raise ValueError("Selected OCR text is empty.")

    selected_text = clean_ocr_text(selected_text)

    print("\nLocal OCR selection complete.")
    print("Selected version:", selected_version)
    print("Scores:", selection["scores"])
    print("OCR text preview:")
    print(selected_text[:500])

    print("\n[PIPELINE] Sending selected OCR text to correction LLM...")
    corrected_text = llm_refine_text(selected_text)

    print("\n[PIPELINE] Sending corrected text to extraction LLM...")
    extracted_json = extract_structured_json_from_text(corrected_text)

    extracted_json["document_type"] = str(
        extracted_json.get("document_type", "unknown")
    ).strip().lower() or "unknown"

    extracted_json["corrected_text"] = corrected_text
    extracted_json["raw_text"] = selected_text
    extracted_json["ocr_selected_version"] = selected_version
    extracted_json["ocr_scores"] = selection["scores"]
    extracted_json["ocr_failures"] = failures

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
    raw_file = save_uploaded_file(upload_path)
    orig_img = standardize_to_image(raw_file)
    versions = preprocess_image(orig_img)

    preview = build_preview_from_versions(versions)

    return {
        "uploaded_file": str(raw_file),
        "standard_image": str(orig_img),
        "preprocessed_versions": versions,
        "selected_ocr_version": preview["selected_ocr_version"],
        "selected_ocr_text": preview["selected_ocr_text"],
        "corrected_text": preview["corrected_text"],
        "extracted_fields": preview["extracted_fields"],
    }