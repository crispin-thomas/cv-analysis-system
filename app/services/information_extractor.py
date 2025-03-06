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
            print(response)
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
        Please wrap the response in a markdown code block with ```json.
        
        CV TEXT:
        {text_content}
        """
        return prompt

    def _parse_llm_response(self, response):
        """Parse the LLM response and convert to structured data"""
        # Extract JSON content from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        print(json_match)
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
