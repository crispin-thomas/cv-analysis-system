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
   git clone https://github.com/crispin-thomas/cv-analysis-system.git
   cd cv-analysis-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```
   Ubuntu/macOS - pip install -r backend/requirements.txt 
   Windows - pip install -r .\backend\requirement.txt
   ```

4. Install frontend dependencies:
   ```
   cd frontend
   npm install
   ```

5. Install Tesseract OCR:
   - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

6. Set up environment variables:
   ```
   cp backend/.env.example backend/.env
   ```
   Edit the `backend/.env` file with your API keys and settings.

## Usage

1. Start the API server:
   ```
   cd backend
   uvicorn app.main:app --reload
   ```

2. Start the Vite frontend:
   ```
   cd frontend
   npm run dev
   ```

3. Open your browser to the URL shown by Vite (typically http://localhost:5173)

4. Upload CV documents using the interface.

5. Query the system using natural language in the chat interface.

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run the test suite with:
```
cd backend
pytest -W ignore::DeprecationWarning
```

## Example Queries

- "Find candidates with Java programming experience"
- "Who has the most relevant experience for a senior developer role?"
- "Compare the education levels of all candidates"
- "Which candidates have worked in the healthcare industry?"
- "Who has the best matching skills for a data scientist position?"