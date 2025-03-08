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
