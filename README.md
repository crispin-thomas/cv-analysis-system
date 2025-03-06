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