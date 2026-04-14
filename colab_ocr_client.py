import os
import requests


def send_images_to_colab_ocr(orig_path: str, p_path: str, m_path: str, colab_url: str = None) -> dict:
    """
    Sends orig, P, and M image versions to the Google Colab OCR API.
    Returns parsed JSON response.
    """

    final_colab_url = "https://catechizable-uncongruously-armani.ngrok-free.dev/"

    if not final_colab_url:
        raise ValueError("COLAB_OCR_URL is empty. Please set it before running the backend.")

    url = final_colab_url.rstrip("/") + "/ocr"

    files = {}
    try:
        files["orig"] = open(orig_path, "rb")
        files["p_img"] = open(p_path, "rb")
        files["m_img"] = open(m_path, "rb")

        print("\n=== OCR REQUEST START ===", flush=True)
        print(f"[OCR] URL: {url}", flush=True)
        print(f"[OCR] orig_path: {orig_path}", flush=True)
        print(f"[OCR] p_path: {p_path}", flush=True)
        print(f"[OCR] m_path: {m_path}", flush=True)

        response = requests.post(
            url,
            files={
                "orig": (os.path.basename(orig_path), files["orig"], "image/png"),
                "p_img": (os.path.basename(p_path), files["p_img"], "image/png"),
                "m_img": (os.path.basename(m_path), files["m_img"], "image/png"),
            },
            timeout=600,
        )

        print(f"[OCR] Response status: {response.status_code}", flush=True)
        print(f"[OCR] Response preview: {response.text[:1500]}", flush=True)

        response.raise_for_status()

        print("=== OCR REQUEST END ===\n", flush=True)
        return response.json()

    except requests.exceptions.Timeout:
        raise Exception("Colab OCR API timed out.")

    except requests.exceptions.ConnectionError as e:
        raise Exception(f"Could not connect to Colab OCR API: {e}")

    except requests.exceptions.HTTPError as e:
        raise Exception(
            f"HTTP error from Colab OCR API: {e}\n"
            f"Response body: {response.text if 'response' in locals() else 'No response'}"
        )

    finally:
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass