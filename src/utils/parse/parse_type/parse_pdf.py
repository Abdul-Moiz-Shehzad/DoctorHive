from typing import List
import fitz
from PIL import Image
import easyocr
import io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ocr_reader = easyocr.Reader(['en'], gpu=False)

def parse_pdf_files(file_path: str, use_ocr_if_empty: bool = True) -> str:
    """
    Parse a PDF file and return combined text.
    Handles both selectable text and scanned PDFs with OCR using EasyOCR.

    Args:
        file_path (str): Path to the PDF file.
        use_ocr_if_empty (bool): If True, run OCR if no selectable text is found.

    Returns:
        str: Combined text from the PDF.
    """
    combined_text = []
    logger.info(f"Parsing PDF file: {file_path}")
    try:
        doc = fitz.open(file_path)
        pdf_text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            
            if not text and use_ocr_if_empty:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_result = ocr_reader.readtext(np.array(img), detail=0)
                pdf_text += "\n".join(ocr_result) + "\n"
            else:
                pdf_text += text + "\n"

        combined_text.append(pdf_text.strip())
        doc.close()

    except Exception as e:
        logger.error(f"Failed to parse PDF {file_path}: {e}")

    return "\n".join(combined_text)
