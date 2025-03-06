import os
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
import io
import tempfile
from logging import getLogger

logger = getLogger(__name__)


class DocumentProcessor:
    def __init__(self, config):
        self.config = config

        os.makedirs(config.TEMP_STORAGE_DIR, exist_ok=True)
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

    def process_document(self, file_path):
        """Process a document and extract text using OCR if needed"""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            return self._process_pdf(file_path)
        elif file_extension in [".docx", ".doc"]:
            return self._process_word(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _process_pdf(self, file_path):
        """Extract text from PDF, using OCR for image-based content"""
        try:
            document = fitz.open(file_path)
            text_content = ""

            for page_num, page in enumerate(document):
                # Try to extract text directly
                text = page.get_text()

                # If little or no text is extracted, try OCR
                if len(text.strip()) < 50:  # Arbitrary threshold
                    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    ocr_text = pytesseract.image_to_string(img)
                    text_content += ocr_text
                else:
                    text_content += text

            document.close()
            return text_content
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise

    def _process_word(self, file_path):
        """Extract text from Word documents"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {str(e)}")
            raise
