from PIL import Image
import easyocr
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

ocr_reader = easyocr.Reader(['en'], gpu=False)

def parse_image_files(file_path: str) -> str:
    """
    Parse an image file (jpg, jpeg, png) using EasyOCR and return text.

    Args:
        file_path (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return ""

    combined_text = []
    try:
        logger.info(f"Parsing image file with OCR: {file_path}")
        img = Image.open(file_path)
        ocr_result = ocr_reader.readtext(np.array(img), detail=0)
        combined_text.append("\n".join(ocr_result))

    except Exception as e:
        logger.error(f"Failed to parse image {file_path}: {e}")

    return "\n".join(combined_text)