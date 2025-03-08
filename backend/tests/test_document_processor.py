import os
import tempfile
from PIL import Image
import io
import pytest
from unittest.mock import patch, MagicMock
from app.services.document_processor import DocumentProcessor

class TestConfig:
    TESSERACT_PATH = "C:\Program Files\Tesseract-OCR\/tesseract.exe"
    TEMP_STORAGE_DIR="data/temp"

@pytest.fixture
def document_processor():
    return DocumentProcessor(TestConfig())

def test_process_pdf_with_text():
    """Test processing a PDF that has extractable text"""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        with patch('fitz.open') as mock_open:
            # Mock a page with extractable text
            mock_page = MagicMock()
            mock_page.get_text.return_value = ""
            
            # Mock the get_pixmap function to return actual image bytes
            mock_pixmap = MagicMock()
            image_bytes = io.BytesIO()
            Image.new("RGB", (100, 100)).save(image_bytes, format="PNG")  # Create a sample image
            mock_pixmap.tobytes.return_value = image_bytes.getvalue()  # Ensure it returns real bytes
            mock_page.get_pixmap.return_value = mock_pixmap

            # Mock document with a single page
            mock_doc = MagicMock()
            mock_doc.__iter__.return_value = iter([mock_page])
            mock_open.return_value = mock_doc

            # Create processor and process document
            processor = DocumentProcessor(TestConfig())
            result = processor._process_pdf(temp_file.name)

            # Verify the result
            assert result.strip() == ""
            mock_page.get_text.assert_called_once()

def test_process_pdf_with_ocr():
    """Test processing a PDF that requires OCR"""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        # Create a small valid image
        dummy_image = Image.new('RGB', (1, 1))
        dummy_buffer = io.BytesIO()
        dummy_image.save(dummy_buffer, format='PNG')
        dummy_image_data = dummy_buffer.getvalue()

        # Set up mocks
        with patch('fitz.open') as mock_open, \
             patch('pytesseract.image_to_string') as mock_ocr, \
             patch('fitz.Matrix') as mock_matrix:
            
            # Configure mock page with little text
            mock_page = MagicMock()
            mock_page.get_text.return_value = ""  # Empty text, will trigger OCR
            
            # Configure mock pixmap
            mock_pixmap = MagicMock()
            mock_pixmap.tobytes.return_value = dummy_image_data
            mock_page.get_pixmap.return_value = mock_pixmap
            
            # Configure mock matrix
            mock_matrix.return_value = MagicMock()
            
            # Configure OCR result
            mock_ocr.return_value = "OCR extracted text"
            
            # Configure mock document
            mock_doc = MagicMock()
            mock_doc.__iter__.return_value = [mock_page]
            mock_open.return_value = mock_doc
            
            # Create processor and process document
            processor = DocumentProcessor(TestConfig())
            result = processor._process_pdf(temp_file.name)
            
            # Verify the result
            assert result == "OCR extracted text"
            mock_ocr.assert_called_once()

def test_process_word():
    """Test processing a Word document"""
    with tempfile.NamedTemporaryFile(suffix=".docx") as temp_file:
        # Set up mock for python-docx
        with patch('docx.Document') as mock_document:
            # Configure mock paragraphs
            mock_para1 = MagicMock()
            mock_para1.text = "This is paragraph 1"
            mock_para2 = MagicMock()
            mock_para2.text = "This is paragraph 2"
            
            # Configure mock document
            mock_doc = MagicMock()
            mock_doc.paragraphs = [mock_para1, mock_para2]
            mock_document.return_value = mock_doc
            
            # Create processor and process document
            processor = DocumentProcessor(TestConfig())
            result = processor._process_word(temp_file.name)
            
            # Verify the result
            assert result == "This is paragraph 1\nThis is paragraph 2"