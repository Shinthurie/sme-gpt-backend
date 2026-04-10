import requests


def send_images_to_colab_ocr(orig_path: str, p_path: str, m_path: str, colab_url: str) -> dict:
    """
    Sends orig, P, and M image versions to the Colab OCR API.
    Returns parsed JSON response.
    """
    if not colab_url:
        raise ValueError("colab_url is empty.")

    url = colab_url.rstrip("/") + "/ocr"

    files = {}
    try:
        files["orig"] = open(orig_path, "rb")
        files["p_img"] = open(p_path, "rb")
        files["m_img"] = open(m_path, "rb")

        print("\n=== SENDING TO COLAB OCR API ===")
        print("URL:", url)
        print("orig_path:", orig_path)
        print("p_path:", p_path)
        print("m_path:", m_path)

        response = requests.post(
            url,
            files=files,
            timeout=300
        )

        print("Response status:", response.status_code)
        print("Response preview:", response.text[:1000])

        response.raise_for_status()
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