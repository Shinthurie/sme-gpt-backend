import re


def score_ocr_text(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0

    score = 0.0
    clean = text.strip()

    score += min(len(clean) / 50.0, 10.0)

    keywords = [
        "date", "order", "invoice", "receipt", "total",
        "cash", "amount", "rs", "qty", "quantity", "po", "dn"
    ]
    lower_text = clean.lower()
    for kw in keywords:
        if kw in lower_text:
            score += 1.5

    number_matches = re.findall(r'\d+', clean)
    score += min(len(number_matches) * 0.5, 5.0)

    html_tags = re.findall(r"<[^>]+>", clean)
    score -= len(html_tags) * 1.2

    strange_chars = re.findall(r'[^\w\s\.,:/()\-\u0D80-\u0DFF]', clean, flags=re.UNICODE)
    score -= min(len(strange_chars) * 0.1, 3.0)

    return round(score, 2)


def select_best_ocr_version(versions: dict) -> dict:
    print("[OCR_SELECTOR] Selecting best OCR version...", flush=True)

    if not versions:
        raise ValueError("No OCR versions provided.")

    scored_versions = {}

    for version_name, version_data in versions.items():
        if isinstance(version_data, dict):
            text = version_data.get("text", "") or ""
            pages = version_data.get("pages", []) or []
        else:
            text = str(version_data or "")
            pages = []

        score = score_ocr_text(text)
        line_count = len([line for line in text.splitlines() if line.strip()])
        text_length = len(text)

        scored_versions[version_name] = {
            "score": score,
            "text_length": text_length,
            "line_count": line_count,
            "full_text": text,
            "pages": pages,
        }

        print(
            f"[OCR_SELECTOR] {version_name}: score={score}, lines={line_count}, chars={text_length}",
            flush=True
        )

    best_version = max(scored_versions.items(), key=lambda x: x[1]["score"])[0]
    best_data = scored_versions[best_version]

    print(f"[OCR_SELECTOR] Best version selected: {best_version}", flush=True)

    return {
        "selected_version": best_version,
        "selected_text": best_data["full_text"],
        "selected_pages": best_data["pages"],
        "scores": {
            key: {
                "score": value["score"],
                "text_length": value["text_length"],
                "line_count": value["line_count"],
            }
            for key, value in scored_versions.items()
        },
    }