import fitz  # PyMuPDF
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pytesseract
import imghdr
from bs4 import BeautifulSoup  # pip install beautifulsoup4

def _extract_text(page):
    text = page.get_text()
    return text.strip() if text and text.strip() else None

def is_image(content):
    return imghdr.what(None, h=content) in ["jpeg", "png", "bmp", "gif", "tiff", "webp"]

def extract_text_from_image_bytes(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()

def parse_pdf_from_url_multithreaded(url, max_workers=2, chunk_size=1):
    """
    Download document (PDF, Image, or Webpage) from URL, extract text accordingly.
    Gracefully return fallback message if unsupported or failed.
    """
    try:
        res = requests.get(url)
        content = res.content
        content_type = res.headers.get("content-type", "").lower()
    except Exception as e:
        print(f"❌ Failed to download: {str(e)}")
        return [f"No data found in this document (download error)"]

    # Handle HTML webpages
    if "text/html" in content_type or url.endswith(".html"):
        print("🌐 Detected HTML page. Extracting text...")
        try:
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n")
            lines = [t.strip() for t in text.splitlines() if t.strip()]
            return lines if lines else ["No data found in this document (empty HTML)"]
        except Exception as e:
            print(f"❌ HTML parse failed: {str(e)}")
            return [f"No data found in this document (HTML error)"]

    # Check for unsupported content
    if "zip" in content_type or url.endswith(".zip"):
        return ["No data found in this document (zip)"]
    if "octet-stream" in content_type or url.endswith(".bin"):
        return ["No data found in this document (bin)"]

    # OCR for image files
    if "image" in content_type or is_image(content):
        print("📷 Detected image file. Using OCR...")
        try:
            text = extract_text_from_image_bytes(content)
            return [text] if text else ["No data found in this document (image empty)"]
        except Exception as e:
            print(f"❌ OCR failed: {str(e)}")
            return [f"No data found in this document (image/OCR error)"]

    # Try PDF parsing
    try:
        with fitz.open(stream=BytesIO(content), filetype="pdf") as doc:
            pages = list(doc)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                texts = list(executor.map(_extract_text, pages))
            if chunk_size > 1:
                chunks = []
                for i in range(0, len(texts), chunk_size):
                    chunk = ' '.join([t for t in texts[i:i+chunk_size] if t])
                    if chunk:
                        chunks.append(chunk)
                return chunks if chunks else ["No data found in this document (empty PDF)"]
            return [t for t in texts if t] or ["No data found in this document (empty PDF)"]
    except Exception as e:
        print(f"❌ Failed to parse as PDF: {str(e)}")
        return [f"No data found in this document (not PDF or corrupted)"]

def parse_pdf_from_file_multithreaded(file_path, max_workers=2, chunk_size=1):
    """
    Parse a local PDF file, extract text in parallel, optionally chunk pages.
    """
    try:
        with fitz.open(file_path) as doc:
            pages = list(doc)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                texts = list(executor.map(_extract_text, pages))
            if chunk_size > 1:
                chunks = []
                for i in range(0, len(texts), chunk_size):
                    chunk = ' '.join([t for t in texts[i:i+chunk_size] if t])
                    if chunk:
                        chunks.append(chunk)
                return chunks if chunks else ["No data found in this document (local PDF empty)"]
            return [t for t in texts if t] or ["No data found in this document (local PDF empty)"]
    except Exception as e:
        print(f"❌ Failed to open local file: {str(e)}")
        return [f"No data found in this document (local file error)"]
