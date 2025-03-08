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
        """Create a prompt for the LLM to extract CV information in a consistent format"""
        prompt = f"""
        You are a CV analysis expert. Extract structured information from the given CV text and return a JSON object. 

        Follow this exact schema:

        ```json
        {{
        "personal_info": {{
            "name": "Full Name",
            "email": "Email Address",
            "phone": "Phone Number",
            "location": "City, State (or Country if applicable)"
        }},
        "education": [
            {{
            "institution": "University Name",
            "degree": "Degree Earned",
            "field": "Field of Study",
            "dates": "Start Year - End Year",
            "location": "City, State (or Country if applicable)"
            }}
        ],
        "work_experience": [
            {{
            "company": "Company Name",
            "position": "Job Title",
            "dates": "Start Month Year - End Month Year (or Present)",
            "location": "City, State (or Country if applicable)",
            "responsibilities": [
                "Responsibility 1",
                "Responsibility 2",
                "Responsibility 3"
            ]
            }}
        ],
        "skills": [
            {{
            "name": "Skill Name",
            "proficiency": ""
            }}
        ],
        "projects": [
            {{
            "name": "Project Name",
            "description": "Short description of the project",
            "technologies": ["Tech1", "Tech2"],
            "dates": "Start Year - End Year"
            }}
        ],
        "certifications": [
            {{
            "name": "Certification Name",
            "issuer": "Issuing Organization",
            "date": "Month Year"
            }}
        ]
        }}
        ```

        Ensure:
        - Use the exact keys and structure above.
        - Maintain a consistent date format (`Month Year` or `Year - Year`).
        - Keep skills as objects with `name` and `proficiency` fields.
        - If no data exists for a section, return an empty array `[]` instead of omitting it.

        CV TEXT:
        {text_content}
        """
        return prompt


    def _parse_llm_response(self, response):
        """Parse the LLM response and convert to structured data"""
        # Extract JSON content from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)

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
        required_keys = [
            "personal_info",
            "education",
            "work_experience",
            "skills",
            "projects",
            "certifications",
        ]

        for key in required_keys:
            if key not in data:
                data[key] = []  # Initialize as empty if missing
