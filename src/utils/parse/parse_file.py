import logging
from typing import List
from src.utils.parse.parse_type.parse_pdf import parse_pdf_files
from src.utils.parse.parse_type.parse_pic import parse_image_files
from fastapi import FastAPI, APIRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Report files parser",
    description="Parses the pdf or jpg files",
    version="1.0"
    )
router = APIRouter(prefix="/utils/parse", tags=["File Parser"])

@app.post("/parse_pdf")
async def parse_endpoint(file_paths: List[str]):
    """
    Endpoint to parse PDF files and extract text.
    """
    document_text=[]
    for file_path, ext in file_paths:
        logger.info(f"Processing file: {file_path} with extension: {ext}")
        if ext =="pdf":
            pdf_text = parse_pdf_files(file_path)
            document_text.append(pdf_text)
        else:
            img_text=parse_image_files(file_path)
            document_text.append(img_text)
    return document_text

app.include_router(router)