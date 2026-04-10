import os
import requests


def send_images_to_colab_ocr(orig_path: str, p_path: str, m_path: str, colab_url: str = None) -> dict:
    """
    Sends orig, P, and M image versions to the Modal OCR API.
    Keeps the old function name so the rest of the backend still works.
    Returns parsed JSON response.
    """

    modal_url = colab_url or os.getenv("MODAL_OCR_URL")

    if not modal_url:
        raise ValueError("MODAL_OCR_URL is empty.")

    url = modal_url.rstrip("/") + "/ocr"

    files = {}
    try:
        files["orig"] = open(orig_path, "rb")
        files["p_img"] = open(p_path, "rb")
        files["m_img"] = open(m_path, "rb")

        print("\n=== SENDING TO MODAL OCR API ===")
        print("URL:", url)
        print("orig_path:", orig_path)
        print("p_path:", p_path)
        print("m_path:", m_path)

        response = requests.post(
            url,
            files={
                "orig": (os.path.basename(orig_path), files["orig"], "image/png"),
                "p_img": (os.path.basename(p_path), files["p_img"], "image/png"),
                "m_img": (os.path.basename(m_path), files["m_img"], "image/png"),
            },
            timeout=600
        )

        print("Response status:", response.status_code)
        print("Response preview:", response.text[:1500])

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        raise Exception("Modal OCR API timed out.")

    except requests.exceptions.ConnectionError as e:
        raise Exception(f"Could not connect to Modal OCR API: {e}")

    except requests.exceptions.HTTPError as e:
        raise Exception(
            f"HTTP error from Modal OCR API: {e}\n"
            f"Response body: {response.text if 'response' in locals() else 'No response'}"
        )

    finally:
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass