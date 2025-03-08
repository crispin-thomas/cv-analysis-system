# CV Analysis System

A complete system for processing, analyzing, and querying multiple CV documents with natural language capabilities.

## System Architecture

```
cv-analysis-system/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   └── security.py        # API security
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Document handling and OCR
│   │   ├── information_extractor.py  # CV data extraction
│   │   ├── llm_service.py     # LLM integration
│   │   └── query_engine.py    # Query processing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cv_schema.py       # Data models
│   │   └── query_schema.py    # Query models
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Utility functions
├── data/
│   ├── processed/             # Processed CV data
│   └── sample_cvs/            # Sample CVs
├── tests/
│   ├── __init__.py
│   ├── test_document_processor.py
│   ├── test_information_extractor.py
│   ├── test_llm_service.py
│   └── test_query_engine.py
├── requirements.txt           # Dependencies
├── .env.example               # Environment variables example
├── Dockerfile                 # Container configuration
└── README.md                  # Setup instructions
```

## Key Components Implementation

### 1. Document Processing Module

```python
# app/services/document_processor.py
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
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
        
    def process_document(self, file_path):
        """Process a document and extract text using OCR if needed"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
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
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
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
```

### 2. Information Extractor Module

```python
# app/services/information_extractor.py
import re
import json
from logging import getLogger

logger = getLogger(__name__)

class InformationExtractor:
    def __init__(self, llm_service):
        self.llm_service = llm_service
        
    def extract_cv_information(self, text_content):
        """Extract structured information from CV text"""
        # Use LLM to extract structured information
        prompt = self._create_extraction_prompt(text_content)
        
        try:
            response = self.llm_service.query(prompt)
            structured_data = self._parse_llm_response(response)
            return structured_data
        except Exception as e:
            logger.error(f"Error during CV information extraction: {str(e)}")
            raise
    
    def _create_extraction_prompt(self, text_content):
        """Create a prompt for the LLM to extract CV information"""
        prompt = f"""
        You are a CV analysis expert. Extract the following information from this CV text in JSON format:
        
        1. Personal Information (name, email, phone, location)
        2. Education History (institution, degree, field, dates)
        3. Work Experience (company, position, dates, responsibilities)
        4. Skills (technical, soft, proficiency levels if available)
        5. Projects (name, description, technologies, dates)
        6. Certifications (name, issuer, date)
        
        Format the response as valid JSON with these exact keys: "personal_info", "education", "work_experience", "skills", "projects", "certifications".
        
        CV TEXT:
        {text_content}
        """
        return prompt
    
    def _parse_llm_response(self, response):
        """Parse the LLM response and convert to structured data"""
        # Extract JSON content from the response
        json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON without code blocks
            json_str = response.strip()
        
        try:
            # Parse and validate the JSON
            structured_data = json.loads(json_str)
            self._validate_cv_structure(structured_data)
            return structured_data
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            raise ValueError("Failed to extract structured data from CV")
    
    def _validate_cv_structure(self, data):
        """Validate that the CV data has all required sections"""
        required_keys = ["personal_info", "education", "work_experience", "skills", "projects", "certifications"]
        
        for key in required_keys:
            if key not in data:
                data[key] = []  # Initialize as empty if missing
```

### 3. LLM Service Module

```python
# app/services/llm_service.py
import os
import time
import json
import anthropic
import openai
from logging import getLogger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = getLogger(__name__)

class LLMService:
    def __init__(self, config):
        self.config = config
        self.provider = config.LLM_PROVIDER.lower()
        self.setup_client()
    
    def setup_client(self):
        """Set up the appropriate LLM client based on configuration"""
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
            self.model = self.config.ANTHROPIC_MODEL
        elif self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
            self.model = self.config.OPENAI_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(self, prompt, temperature=0.1):
        """Send a query to the LLM and get a response"""
        try:
            if self.provider == "anthropic":
                return self._query_anthropic(prompt, temperature)
            elif self.provider == "openai":
                return self._query_openai(prompt, temperature)
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            raise
    
    def _query_anthropic(self, prompt, temperature):
        """Query the Anthropic Claude API"""
        message = self.client.messages.create(
            model=self.model,
            temperature=temperature,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    
    def _query_openai(self, prompt, temperature):
        """Query the OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are an expert CV analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
```

### 4. Query Engine Module

```python
# app/services/query_engine.py
import json
from logging import getLogger

logger = getLogger(__name__)

class QueryEngine:
    def __init__(self, llm_service, cv_database):
        self.llm_service = llm_service
        self.cv_database = cv_database
        self.conversation_context = {}
    
    def process_query(self, query, user_id=None, conversation_id=None):
        """Process a natural language query about CVs"""
        # Track conversation context
        context_key = f"{user_id}_{conversation_id}" if user_id and conversation_id else None
        context = self._get_conversation_context(context_key)
        
        # Get relevant CV data
        cv_data = self.cv_database.get_all_cvs()
        
        # Create prompt
        prompt = self._create_query_prompt(query, cv_data, context)
        
        try:
            # Get response from LLM
            response = self.llm_service.query(prompt)
            
            # Update conversation context
            if context_key:
                self._update_conversation_context(context_key, query, response)
                
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I'm sorry, I encountered an error processing your query: {str(e)}"
    
    def _create_query_prompt(self, query, cv_data, context):
        """Create a prompt for the LLM to answer questions about CVs"""
        cv_json = json.dumps(cv_data, indent=2)
        context_str = json.dumps(context, indent=2) if context else "No previous context."
        
        prompt = f"""
        You are an expert CV analysis assistant. Answer the following question about these CVs.
        
        CV DATA:
        {cv_json}
        
        PREVIOUS CONVERSATION CONTEXT:
        {context_str}
        
        USER QUESTION:
        {query}
        
        Provide a clear and concise answer based on the CV data provided. 
        If you're comparing candidates, highlight the key differences.
        If you're listing candidates with specific qualifications, be precise about why they match.
        If you need more information, specify what additional details would help.
        """
        return prompt
    
    def _get_conversation_context(self, context_key):
        """Get the conversation context for a specific user/conversation"""
        if not context_key:
            return None
            
        if context_key not in self.conversation_context:
            self.conversation_context[context_key] = []
            
        return self.conversation_context[context_key]
    
    def _update_conversation_context(self, context_key, query, response):
        """Update the conversation context with the new query and response"""
        if not context_key:
            return
            
        context = self._get_conversation_context(context_key)
        context.append({"query": query, "response": response})
        
        # Keep only the last 5 exchanges for context
        self.conversation_context[context_key] = context[-5:]
```

### 5. CV Database Module

```python
# app/services/cv_database.py
import os
import json
from logging import getLogger

logger = getLogger(__name__)

class CVDatabase:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_cv(self, cv_id, cv_data):
        """Save processed CV data to storage"""
        try:
            file_path = os.path.join(self.storage_dir, f"{cv_id}.json")
            with open(file_path, 'w') as f:
                json.dump(cv_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving CV {cv_id}: {str(e)}")
            return False
    
    def get_cv(self, cv_id):
        """Retrieve a specific CV by ID"""
        try:
            file_path = os.path.join(self.storage_dir, f"{cv_id}.json")
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving CV {cv_id}: {str(e)}")
            return None
    
    def get_all_cvs(self):
        """Retrieve all CVs in the database"""
        try:
            cv_files = [f for f in os.listdir(self.storage_dir) if f.endswith('.json')]
            all_cvs = {}
            
            for file_name in cv_files:
                cv_id = os.path.splitext(file_name)[0]
                cv_data = self.get_cv(cv_id)
                if cv_data:
                    all_cvs[cv_id] = cv_data
                    
            return all_cvs
        except Exception as e:
            logger.error(f"Error retrieving all CVs: {str(e)}")
            return {}
    
    def delete_cv(self, cv_id):
        """Delete a CV from storage"""
        try:
            file_path = os.path.join(self.storage_dir, f"{cv_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting CV {cv_id}: {str(e)}")
            return False
```

### 6. Main Application with FastAPI

```python
# app/main.py
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile

from app.core.config import Settings
from app.services.document_processor import DocumentProcessor
from app.services.information_extractor import InformationExtractor
from app.services.llm_service import LLMService
from app.services.query_engine import QueryEngine
from app.services.cv_database import CVDatabase

app = FastAPI(title="CV Analysis System API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get settings
settings = Settings()

# Initialize services
llm_service = LLMService(settings)
document_processor = DocumentProcessor(settings)
information_extractor = InformationExtractor(llm_service)
cv_database = CVDatabase(settings.STORAGE_DIR)
query_engine = QueryEngine(llm_service, cv_database)

# Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str

@app.post("/upload-cv/", status_code=201)
async def upload_cv(cv_file: UploadFile = File(...)):
    """Upload and process a CV file"""
    # Generate unique ID for this CV
    cv_id = str(uuid.uuid4())
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        content = await cv_file.read()
        temp_file.write(content)
    
    try:
        # Process document
        text_content = document_processor.process_document(temp_file_path)
        
        # Extract information
        cv_data = information_extractor.extract_cv_information(text_content)
        
        # Store processed data
        cv_database.save_cv(cv_id, cv_data)
        
        return {"cv_id": cv_id, "filename": cv_file.filename, "status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query/", response_model=QueryResponse)
async def query_cvs(request: QueryRequest):
    """Query the CV database with natural language"""
    response = query_engine.process_query(
        request.query, 
        user_id=request.user_id, 
        conversation_id=request.conversation_id
    )
    return {"response": response}

@app.get("/cvs/", status_code=200)
async def list_cvs():
    """List all processed CVs in the database"""
    cvs = cv_database.get_all_cvs()
    return {"cv_count": len(cvs), "cvs": cvs}

@app.get("/cvs/{cv_id}", status_code=200)
async def get_cv(cv_id: str):
    """Get a specific CV by ID"""
    cv_data = cv_database.get_cv(cv_id)
    if not cv_data:
        raise HTTPException(status_code=404, detail=f"CV with ID {cv_id} not found")
    return cv_data

@app.delete("/cvs/{cv_id}", status_code=200)
async def delete_cv(cv_id: str):
    """Delete a CV by ID"""
    success = cv_database.delete_cv(cv_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"CV with ID {cv_id} not found")
    return {"status": "deleted", "cv_id": cv_id}
```

### 7. Configuration Module

```python
# app/core/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CV Analysis System"
    
    # LLM settings
    LLM_PROVIDER: str = "anthropic"  # "anthropic" or "openai"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    
    # OCR settings
    TESSERACT_PATH: str = "/usr/bin/tesseract"  # Default Linux path
    
    # Storage settings
    STORAGE_DIR: str = "data/processed"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 8. Frontend Chatbot Interface (Streamlit)

```python
# frontend.py
import streamlit as st
import requests
import json
import uuid
import os
from io import BytesIO

# API URL
API_URL = "http://localhost:8000"

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App title
st.title("CV Analysis Chatbot")

# File uploader for CVs
with st.sidebar:
    st.header("Upload CVs")
    uploaded_files = st.file_uploader("Choose CV files", accept_multiple_files=True, type=["pdf", "docx"])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if st.button(f"Process {uploaded_file.name}"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save file to temporary storage
                    bytes_data = uploaded_file.getvalue()
                    
                    # Upload the file to the API
                    files = {"cv_file": (uploaded_file.name, bytes_data, uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload-cv/", files=files)
                    
                    if response.status_code == 201:
                        result = response.json()
                        st.success(f"✅ {uploaded_file.name} processed (ID: {result['cv_id']})")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {response.text}")
    
    # List processed CVs
    st.header("Processed CVs")
    if st.button("Refresh CV List"):
        response = requests.get(f"{API_URL}/cvs/")
        if response.status_code == 200:
            cvs = response.json()
            st.write(f"Total CVs: {cvs['cv_count']}")
            for cv_id, cv_data in cvs.get('cvs', {}).items():
                st.write(f"CV ID: {cv_id}")
                if cv_data.get('personal_info', {}).get('name'):
                    st.write(f"Name: {cv_data['personal_info']['name']}")
        else:
            st.error("Failed to retrieve CV list")
    
    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.success("Conversation cleared")

# Chat interface
st.header("Chat with CV Assistant")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Assistant: {message['content']}")

# Query input
query = st.text_input("Ask about the CVs:")

if query:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Prepare query request
    query_data = {
        "query": query,
        "user_id": st.session_state.user_id,
        "conversation_id": st.session_state.conversation_id
    }
    
    # Send query to API
    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/query/", json=query_data)
        
        if response.status_code == 200:
            result = response.json()
            assistant_message = result["response"]
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
            
            # Force a rerun to update the chat display
            st.experimental_rerun()
        else:
            st.error(f"Error: {response.text}")
```

### 9. Requirements.txt

```
# requirements.txt
fastapi==0.105.0
uvicorn==0.24.0
python-multipart==0.0.6
pydantic==2.5.2
pydantic-settings==2.1.0
pymupdf==1.23.7
pytesseract==0.3.10
python-docx==1.0.1
pillow==10.1.0
anthropic==0.8.1
openai==1.3.8
tenacity==8.2.3
streamlit==1.29.0
requests==2.31.0
python-dotenv==1.0.0
pytest==7.4.3
```

### 10. .env.example

```
# .env.example

# API settings
API_V1_STR=/api/v1
PROJECT_NAME=CV Analysis System

# LLM settings (choose one provider)
LLM_PROVIDER=anthropic  # or openai
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-opus-20240229
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o

# OCR settings
TESSERACT_PATH=/usr/bin/tesseract  # Update based on your OS

# Storage settings
STORAGE_DIR=data/processed
```

### 11. README.md

```markdown
# CV Analysis System

A complete system for processing, analyzing, and querying multiple CV documents with natural language capabilities.

## Features

- Document processing with OCR for PDF and Word documents
- Structured information extraction from CVs
- LLM-powered natural language interface for querying CV data
- API and web interface for easy interaction
- Support for conversation context and follow-up questions

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cv-analysis-system.git
   cd cv-analysis-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

5. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Edit the `.env` file with your API keys and settings.

## Usage

1. Start the API server:
   ```
   uvicorn app.main:app --reload
   ```

2. Start the Streamlit interface:
   ```
   streamlit run frontend.py
   ```

3. Open your browser to the URL shown by Streamlit (typically http://localhost:8501)

4. Upload CV documents using the sidebar interface.

5. Query the system using natural language in the chat interface.

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run the test suite with:
```
pytest
```

## Example Queries

- "Find candidates with Java programming experience"
- "Who has the most relevant experience for a senior developer role?"
- "Compare the education levels of all candidates"
- "Which candidates have worked in the healthcare industry?"
- "Who has the best matching skills for a data scientist position?"
```

### 12. Test Suite Example

```python
# tests/test_document_processor.py
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from app.services.document_processor import DocumentProcessor

class TestConfig:
    TESSERACT_PATH = "/usr/bin/tesseract"

@pytest.fixture
def document_processor():
    return DocumentProcessor(TestConfig())

def test_process_pdf_with_text():
    """Test processing a PDF that has extractable text"""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        # Set up mock for PyMuPDF
        with patch('fitz.open') as mock_open:
            # Configure mock page with text
            mock_page = MagicMock()
            mock_page.get_text.return_value = "This is a test CV document"
            
            # Configure mock document
            mock_doc = MagicMock()
            mock_doc.__iter__.return_value = [mock_page]
            mock_open.return_value = mock_doc
            
            # Create processor and process document
            processor = DocumentProcessor(TestConfig())
            result = processor._process_pdf(temp_file.name)
            
            # Verify the result
            assert result == "This is a test CV document"
            mock_page.get_text.assert_called_once()

def test_process_pdf_with_ocr():
    """Test processing a PDF that requires OCR"""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        # Set up mocks
        with patch('fitz.open') as mock_open, \
             patch('pytesseract.image_to_string') as mock_ocr:
            # Configure mock page with little text
            mock_page = MagicMock()
            mock_page.get_text.return_value = ""  # Empty text, will trigger OCR
            mock_page.get_pixmap.return_value = MagicMock()
            
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
            