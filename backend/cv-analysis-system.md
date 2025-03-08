## Test Suite (Continued)

```python
# tests/test_information_extractor.py
import pytest
from unittest.mock import MagicMock, patch
from app.services.information_extractor import InformationExtractor

@pytest.fixture
def llm_service_mock():
    mock = MagicMock()
    mock.query.return_value = """
    ```json
    {
        "personal_info": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "123-456-7890",
            "location": "New York, NY"
        },
        "education": [
            {
                "institution": "Stanford University",
                "degree": "Bachelor of Science",
                "field": "Computer Science",
                "dates": "2015-2019"
            }
        ],
        "work_experience": [
            {
                "company": "Tech Solutions Inc.",
                "position": "Software Engineer",
                "dates": "2019-Present",
                "responsibilities": "Developed full-stack applications using React and Django."
            }
        ],
        "skills": [
            "Python", "JavaScript", "React", "SQL", "Git"
        ],
        "projects": [
            {
                "name": "E-commerce Platform",
                "description": "Built a marketplace application",
                "technologies": ["React", "Node.js", "MongoDB"],
                "dates": "2018-2019"
            }
        ],
        "certifications": [
            {
                "name": "AWS Certified Developer",
                "issuer": "Amazon Web Services",
                "date": "2020"
            }
        ]
    }
    ```
    """
    return mock

@pytest.fixture
def information_extractor(llm_service_mock):
    return InformationExtractor(llm_service_mock)

def test_extract_cv_information(information_extractor, llm_service_mock):
    """Test extracting information from CV text"""
    cv_text = "Sample CV text"
    result = information_extractor.extract_cv_information(cv_text)
    
    # Verify LLM was called with proper prompt
    llm_service_mock.query.assert_called_once()
    assert "Sample CV text" in llm_service_mock.query.call_args[0][0]
    
    # Verify the result structure
    assert "personal_info" in result
    assert result["personal_info"]["name"] == "John Doe"
    assert "education" in result
    assert result["education"][0]["institution"] == "Stanford University"
    assert "work_experience" in result
    assert "skills" in result
    assert "projects" in result
    assert "certifications" in result

def test_parse_llm_response_with_code_block(information_extractor):
    """Test parsing LLM response with code block"""
    response = """
    I've extracted the CV information:
    
    ```json
    {"personal_info": {"name": "Jane Smith"}}
    ```
    """
    result = information_extractor._parse_llm_response(response)
    assert result["personal_info"]["name"] == "Jane Smith"

def test_parse_llm_response_without_code_block(information_extractor):
    """Test parsing LLM response without code block"""
    response = """
    {"personal_info": {"name": "Jane Smith"}}
    """
    result = information_extractor._parse_llm_response(response)
    assert result["personal_info"]["name"] == "Jane Smith"

def test_validate_cv_structure(information_extractor):
    """Test validation of CV structure"""
    # Incomplete data
    data = {"personal_info": {"name": "Test"}}
    information_extractor._validate_cv_structure(data)
    
    # Check that missing sections were initialized
    assert "education" in data
    assert "work_experience" in data
    assert "skills" in data
    assert "projects" in data
    assert "certifications" in data
    assert isinstance(data["education"], list)
```

```python
# tests/test_llm_service.py
import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_service import LLMService

class TestConfig:
    LLM_PROVIDER = "anthropic"
    ANTHROPIC_API_KEY = "test_key"
    ANTHROPIC_MODEL = "claude-3-opus-20240229"
    OPENAI_API_KEY = "test_key"
    OPENAI_MODEL = "gpt-4o"

@pytest.fixture
def anthropic_config():
    config = TestConfig()
    config.LLM_PROVIDER = "anthropic"
    return config

@pytest.fixture
def openai_config():
    config = TestConfig()
    config.LLM_PROVIDER = "openai"
    return config

@patch('anthropic.Anthropic')
def test_anthropic_setup(mock_anthropic, anthropic_config):
    """Test that Anthropic client is properly set up"""
    llm_service = LLMService(anthropic_config)
    mock_anthropic.assert_called_once_with(api_key="test_key")
    assert llm_service.model == "claude-3-opus-20240229"

@patch('openai.OpenAI')
def test_openai_setup(mock_openai, openai_config):
    """Test that OpenAI client is properly set up"""
    llm_service = LLMService(openai_config)
    mock_openai.assert_called_once_with(api_key="test_key")
    assert llm_service.model == "gpt-4o"

@patch('anthropic.Anthropic')
def test_anthropic_query(mock_anthropic, anthropic_config):
    """Test querying the Anthropic API"""
    # Set up mock
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Test response"
    mock_message.content = [mock_content]
    mock_client.messages.create.return_value = mock_message
    mock_anthropic.return_value = mock_client
    
    # Create service and query
    llm_service = LLMService(anthropic_config)
    result = llm_service.query("Test prompt")
    
    # Verify result
    assert result == "Test response"
    mock_client.messages.create.assert_called_once_with(
        model="claude-3-opus-20240229",
        temperature=0.1,
        max_tokens=4000,
        messages=[{"role": "user", "content": "Test prompt"}]
    )

@patch('openai.OpenAI')
def test_openai_query(mock_openai, openai_config):
    """Test querying the OpenAI API"""
    # Set up mock
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    mock_chat.completions.create.return_value = mock_completion
    mock_client.chat = mock_chat
    mock_openai.return_value = mock_client
    
    # Create service and query
    llm_service = LLMService(openai_config)
    result = llm_service.query("Test prompt")
    
    # Verify result
    assert result == "Test response"
    mock_chat.completions.create.assert_called_once()
```

```python
# tests/test_query_engine.py
import pytest
from unittest.mock import MagicMock
from app.services.query_engine import QueryEngine

@pytest.fixture
def llm_service_mock():
    mock = MagicMock()
    mock.query.return_value = "This is a test response about CVs"
    return mock

@pytest.fixture
def cv_database_mock():
    mock = MagicMock()
    mock.get_all_cvs.return_value = {
        "cv1": {
            "personal_info": {"name": "John Doe"},
            "skills": ["Python", "Machine Learning"]
        },
        "cv2": {
            "personal_info": {"name": "Jane Smith"},
            "skills": ["JavaScript", "React"]
        }
    }
    return mock

@pytest.fixture
def query_engine(llm_service_mock, cv_database_mock):
    return QueryEngine(llm_service_mock, cv_database_mock)

def test_process_query_simple(query_engine, llm_service_mock, cv_database_mock):
    """Test processing a simple query without context"""
    result = query_engine.process_query("Who has Python skills?")
    
    # Verify database was accessed
    cv_database_mock.get_all_cvs.assert_called_once()
    
    # Verify LLM was called with proper prompt
    llm_service_mock.query.assert_called_once()
    prompt = llm_service_mock.query.call_args[0][0]
    assert "Who has Python skills?" in prompt
    assert "John Doe" in prompt
    assert "Jane Smith" in prompt
    
    # Verify result
    assert result == "This is a test response about CVs"

def test_process_query_with_context(query_engine, llm_service_mock, cv_database_mock):
    """Test processing a query with conversation context"""
    # First query
    query_engine.process_query("Who has Python skills?", user_id="user1", conversation_id="conv1")
    
    # Second query with context
    result = query_engine.process_query("What other skills does this person have?", user_id="user1", conversation_id="conv1")
    
    # Verify context was used in second query
    assert len(llm_service_mock.query.call_args_list) == 2
    second_prompt = llm_service_mock.query.call_args_list[1][0][0]
    assert "Who has Python skills?" in second_prompt
    assert "What other skills does this person have?" in second_prompt
    
    # Verify result
    assert result == "This is a test response about CVs"

def test_conversation_context_management(query_engine):
    """Test that conversation context is properly managed"""
    # First conversation
    query_engine.process_query("Query 1", user_id="user1", conversation_id="conv1")
    query_engine.process_query("Query 2", user_id="user1", conversation_id="conv1")
    
    # Second conversation
    query_engine.process_query("Query A", user_id="user2", conversation_id="conv2")
    
    # Check contexts
    context1 = query_engine._get_conversation_context("user1_conv1")
    context2 = query_engine._get_conversation_context("user2_conv2")
    
    assert len(context1) == 2
    assert context1[0]["query"] == "Query 1"
    assert context1[1]["query"] == "Query 2"
    
    assert len(context2) == 1
    assert context2[0]["query"] == "Query A"
```

## Additional Features and Improvements

### 1. Batch Processing Script

```python
# batch_process.py
import os
import argparse
import requests
from tqdm import tqdm

def process_directory(directory_path, api_url):
    """Process all CV files in a directory"""
    # Get all PDF and DOCX files
    files = []
    for ext in ['.pdf', '.docx', '.doc']:
        files.extend([os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(ext)])
    
    print(f"Found {len(files)} CV files to process")
    
    # Process each file
    results = []
    for file_path in tqdm(files, desc="Processing CVs"):
        try:
            with open(file_path, 'rb') as f:
                filename = os.path.basename(file_path)
                files = {"cv_file": (filename, f, "application/octet-stream")}
                response = requests.post(f"{api_url}/upload-cv/", files=files)
                
                if response.status_code == 201:
                    result = response.json()
                    results.append({
                        "filename": filename,
                        "cv_id": result["cv_id"],
                        "status": "success"
                    })
                    print(f"✅ {filename} processed (ID: {result['cv_id']})")
                else:
                    results.append({
                        "filename": filename,
                        "status": "failed",
                        "error": response.text
                    })
                    print(f"❌ {filename} failed: {response.text}")
        except Exception as e:
            results.append({
                "filename": os.path.basename(file_path),
                "status": "error",
                "error": str(e)
            })
            print(f"❌ {os.path.basename(file_path)} error: {str(e)}")
    
    # Print summary
    success_count = len([r for r in results if r["status"] == "success"])
    print(f"\nProcessed {len(results)} files: {success_count} successful, {len(results) - success_count} failed")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process CV files")
    parser.add_argument("directory", help="Directory containing CV files")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    args = parser.parse_args()
    
    process_directory(args.directory, args.api_url)
```

### 2. Enhanced Document Processing with Table Extraction

```python
# app/services/enhanced_document_processor.py
import os
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
import io
import tempfile
from logging import getLogger
import tabula
import pandas as pd
import re

logger = getLogger(__name__)

class EnhancedDocumentProcessor:
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
            # Extract tables first
            tables = self._extract_tables(file_path)
            table_text = self._format_tables(tables)
            
            # Extract regular text
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
            
            # Combine table text and regular text
            combined_text = text_content
            if table_text:
                combined_text += "\n\nTABLES:\n" + table_text
                
            return combined_text
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def _extract_tables(self, file_path):
        """Extract tables from PDF"""
        try:
            # Use tabula-py to extract tables
            tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            return tables
        except Exception as e:
            logger.warning(f"Table extraction failed for {file_path}: {str(e)}")
            return []
    
    def _format_tables(self, tables):
        """Format extracted tables as text"""
        if not tables:
            return ""
            
        result = []
        for i, table in enumerate(tables):
            if not table.empty:
                result.append(f"Table {i+1}:")
                result.append(table.to_string(index=False))
                result.append("")
                
        return "\n".join(result)
    
    def _process_word(self, file_path):
        """Extract text and tables from Word documents"""
        try:
            doc = docx.Document(file_path)
            
            # Extract regular paragraphs
            paragraphs = [para.text for para in doc.paragraphs]
            
            # Extract tables
            tables = []
            for table in doc.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.text for cell in row.cells]
                    rows.append(cells)
                tables.append(rows)
                
            # Format tables as text
            table_text = ""
            for i, table in enumerate(tables):
                if table:
                    table_text += f"\nTable {i+1}:\n"
                    for row in table:
                        table_text += " | ".join(row) + "\n"
                    table_text += "\n"
            
            # Combine paragraph text and table text
            text_content = "\n".join(paragraphs)
            if table_text:
                text_content += "\n\nTABLES:\n" + table_text
                
            return text_content
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {str(e)}")
            raise
            
    def _extract_sections(self, text):
        """Attempt to identify common CV sections in the text"""
        common_sections = [
            "education", "experience", "work experience", "employment", 
            "skills", "technical skills", "certifications", "projects",
            "personal information", "contact", "summary", "objective",
            "publications", "languages", "references", "awards", "volunteer"
        ]
        
        section_pattern = r'(?i)^\s*(' + '|'.join(common_sections) + r')[\s:]*$'
        
        # Find potential section headers
        sections = {}
        current_section = "preamble"
        sections[current_section] = []
        
        for line in text.split('\n'):
            if re.match(section_pattern, line.strip()):
                current_section = line.strip().lower()
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line)
                
        # Convert back to text
        structured_text = ""
        for section, lines in sections.items():
            if lines:
                if section != "preamble":
                    structured_text += f"\n\n## {section.upper()}\n"
                structured_text += "\n".join(lines)
                
        return structured_text
```

### 3. Comparison and Analytics Module

```python
# app/services/cv_analytics.py
from collections import Counter
from datetime import datetime
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CVAnalytics:
    def __init__(self, cv_database):
        self.cv_database = cv_database
        
    def get_most_common_skills(self, top_n=10):
        """Get the most common skills across all CVs"""
        all_cvs = self.cv_database.get_all_cvs()
        all_skills = []
        
        for cv_id, cv_data in all_cvs.items():
            skills = cv_data.get('skills', [])
            if isinstance(skills, list):
                all_skills.extend([s.lower() for s in skills])
            elif isinstance(skills, dict):
                all_skills.extend([s.lower() for s in skills.keys()])
                
        skill_counter = Counter(all_skills)
        return skill_counter.most_common(top_n)
    
    def calculate_experience_years(self, cv_data):
        """Calculate total years of experience from a CV"""
        work_exp = cv_data.get('work_experience', [])
        total_years = 0
        
        current_year = datetime.now().year
        
        for job in work_exp:
            dates = job.get('dates', '')
            if not dates:
                continue
                
            # Extract years using regex
            years = re.findall(r'\b(19\d\d|20\d\d)\b', dates)
            if len(years) >= 2:
                try:
                    start_year = int(years[0])
                    
                    # Handle "Present" or current roles
                    if "present" in dates.lower() or years[1] == str(current_year):
                        end_year = current_year
                    else:
                        end_year = int(years[1])
                        
                    total_years += (end_year - start_year)
                except (ValueError, IndexError):
                    pass
                    
        return total_years
    
    def get_experience_distribution(self):
        """Get the distribution of experience years across all CVs"""
        all_cvs = self.cv_database.get_all_cvs()
        experience_years = {}
        
        for cv_id, cv_data in all_cvs.items():
            try:
                name = cv_data.get('personal_info', {}).get('name', cv_id)
                years = self.calculate_experience_years(cv_data)
                experience_years[name] = years
            except Exception:
                continue
                
        return experience_years
    
    def find_similar_candidates(self, cv_id, top_n=3):
        """Find the most similar candidates to a given CV"""
        all_cvs = self.cv_database.get_all_cvs()
        if cv_id not in all_cvs:
            return []
            
        target_cv = all_cvs[cv_id]
        
        # Extract content for comparison
        cv_contents = {}
        for cid, cv_data in all_cvs.items():
            if cid == cv_id:
                continue
                
            # Create a text representation of the CV
            content = []
            
            # Add skills
            skills = cv_data.get('skills', [])
            if isinstance(skills, list):
                content.extend(skills)
            elif isinstance(skills, dict):
                content.extend(list(skills.keys()))
                
            # Add work experience details
            for job in cv_data.get('work_experience', []):
                if 'position' in job:
                    content.append(job['position'])
                if 'company' in job:
                    content.append(job['company'])
                if 'responsibilities' in job:
                    content.append(job['responsibilities'])
                    
            # Add education
            for edu in cv_data.get('education', []):
                if 'field' in edu:
                    content.append(edu['field'])
                if 'degree' in edu:
                    content.append(edu['degree'])
                    
            cv_contents[cid] = " ".join(content)
            
        # Extract target CV content
        target_content = []
        skills = target_cv.get('skills', [])
        if isinstance(skills, list):
            target_content.extend(skills)
        elif isinstance(skills, dict):
            target_content.extend(list(skills.keys()))
            
        for job in target_cv.get('work_experience', []):
            if 'position' in job:
                target_content.append(job['position'])
            if 'company' in job:
                target_content.append(job['company'])
            if 'responsibilities' in job:
                target_content.append(job['responsibilities'])
                
        for edu in target_cv.get('education', []):
            if 'field' in edu:
                target_content.append(edu['field'])
            if 'degree' in edu:
                target_content.append(edu['degree'])
                
        target_text = " ".join(target_content)
        
        # Calculate similarity using TF-IDF and cosine similarity
        all_texts = [target_text] + list(cv_contents.values())
        cv_ids = [cv_id] + list(cv_contents.keys())
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get the top N similar CVs
        similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
        
        result = []
        for idx in similar_indices:
            similar_cv_id = cv_ids[idx + 1]  # +1 because we excluded the target CV
            name = all_cvs[similar_cv_id].get('personal_info', {}).get('name', similar_cv_id)
            result.append({
                'cv_id': similar_cv_id,
                'name': name,
                'similarity_score': cosine_similarities[idx]
            })
            
        return result
```

### 4. Job Matching Module

```python
# app/services/job_matcher.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobMatcher:
    def __init__(self, cv_database, llm_service):
        self.cv_database = cv_database
        self.llm_service = llm_service
        
    def extract_job_requirements(self, job_description):
        """Extract structured job requirements from a job description"""
        prompt = f"""
        Extract the key job requirements from this job description. For each category, provide a list of requirements.
        Categories:
        1. Required Skills
        2. Education Requirements
        3. Experience Requirements
        4. Preferred Qualifications
        
        Format the response as JSON with these keys: "required_skills", "education", "experience", "preferred".
        
        JOB DESCRIPTION:
        {job_description}
        """
        
        response = self.llm_service.query(prompt)
        
        # Extract JSON content from the response
        json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
        
        if json_match:
            import json
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
                
        # If parsing failed, use a simplified approach
        requirements = {
            "required_skills": [],
            "education": [],
            "experience": [],
            "preferred": []
        }
        
        # Basic extraction using regex patterns
        skill_patterns = [
            r'skills required:?\s*(.*?)(?=\n\n|\Z)',
            r'technical skills:?\s*(.*?)(?=\n\n|\Z)',
            r'requirements:?\s*(.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in skill_patterns:
            matches = re.search(pattern, job_description, re.IGNORECASE | re.DOTALL)
            if matches:
                skills_text = matches.group(1)
                # Extract items that look like bullet points or numbered items
                items = re.findall(r'(?:^|\n)(?:\*|\-|\d+\.)\s*(.*?)(?=\n|$)', skills_text)
                if items:
                    requirements["required_skills"].extend(items)
                    
        # Extract education requirements
        edu_patterns = [
            r'education:?\s*(.*?)(?=\n\n|\Z)',
            r'qualifications:?\s*(.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in edu_patterns:
            matches = re.search(pattern, job_description, re.IGNORECASE | re.DOTALL)
            if matches:
                edu_text = matches.group(1)
                items = re.findall(r'(?:^|\n)(?:\*|\-|\d+\.)\s*(.*?)(?=\n|$)', edu_text)
                if items:
                    requirements["education"].extend(items)
                    
        # Extract experience requirements
        exp_patterns = [
            r'experience:?\s*(.*?)(?=\n\n|\Z)',
            r'background:?\s*(.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in exp_patterns:
            matches = re.search(pattern, job_description, re.IGNORECASE | re.DOTALL)
            if matches:
                exp_text = matches.group(1)
                items = re.findall(r'(?:^|\n)(?:\*|\-|\d+\.)\s*(.*?)(?=\n|$)', exp_text)
                if items:
                    requirements["experience"].extend(items)
                    
        return requirements
    
    def match_candidates(self, job_description, top_n=5):
        """Match candidates to a job description"""
        # Extract job requirements
        job_reqs = self.extract_job_requirements(job_description)
        
        # Get all CVs
        all_cvs = self.cv_database.get_all_cvs()
        
        # Convert job requirements to text
        job_req_text = " ".join([
            " ".join(job_reqs.get("required_skills", [])),
            " ".join(job_reqs.get("education", [])),
            " ".join(job_reqs.get("experience", [])),
            " ".join(job_reqs.get("preferred", []))
        ])
        
        # For each CV, create a comparable text representation
        cv_texts = {}
        for cv_id, cv_data in all_cvs.items():
            # Extract skills
            skills = cv_data.get("skills", [])
            if isinstance(skills, list):
                skill_text = " ".join(skills)
            elif isinstance(skills, dict):
                skill_text = " ".join(skills.keys())
            else:
                skill_text = ""
                
            # Extract education
            edu_text = ""
            for edu in cv_data.get("education", []):
                if isinstance(edu, dict):
                    edu_values = [str(v) for v in edu.values() if v]
                    edu_text += " ".join(edu_values) + " "
                elif isinstance(edu, str):
                    edu_text += edu + " "
                    
            # Extract experience
            exp_text = ""
            for exp in cv_data.get("work_experience", []):
                if isinstance(exp, dict):
                    exp_values = [str(v) for v in exp.values() if v]
                    exp_text += " ".join(exp_values) + " "
                elif isinstance(exp, str):
                    exp_text += exp + " "
                    
            # Combine all text
            cv_texts[cv_id] = f"{skill_text} {edu_text} {exp_text}"
            
        # Use TF-IDF and cosine