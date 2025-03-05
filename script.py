import pytesseract
import cv2
import pymupdf  # PyMuPDF
from pdf2image import convert_from_path
import docx
import re
import spacy
import os

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Set path for Tesseract (only needed for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image(image):
    """Extract text from an image using Tesseract OCR."""
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding for better OCR accuracy
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Extract text
    text = pytesseract.image_to_string(thresh)
    return text


def extract_text_from_pdf(pdf_path):
    """Extract text from both selectable and scanned PDFs."""
    text = ""
    doc = pymupdf.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        extracted_text = page.get_text("text")
        
        # If no text found, convert page to image and use OCR
        if not extracted_text.strip():
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            for image in images:
                image = cv2.cvtColor(cv2.imread(image.filename), cv2.COLOR_RGB2BGR)
                extracted_text += ocr_image(image) + "\n"
        
        text += extracted_text + "\n"
    
    return text


def extract_text_from_docx(docx_path):
    """Extract text from DOCX, including OCR for embedded images."""
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])

    # Process images inside DOCX
    for shape in doc.inline_shapes:
        image_path = "temp_image.png"
        with open(image_path, "wb") as f:
            f.write(shape._inline.graphic.graphicData.pic.blipFill.blip._blob)
        image = cv2.imread(image_path)
        text += ocr_image(image) + "\n"
        os.remove(image_path)

    return text


def extract_personal_info(text):
    """Extract personal information using regex and NLP."""
    name = None
    email = re.findall(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+", text)
    phone = re.findall(r"\+?\d[\d -]{8,15}\d", text)

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    return {
        "name": name,
        "email": email[0] if email else None,
        "phone": phone[0] if phone else None,
    }


def extract_education(text):
    """Extract education history from text."""
    edu_pattern = r"(Bachelor|Master|PhD|B\.Sc|M\.Sc|B\.Tech|M\.Tech|Diploma|Associate).+"
    return re.findall(edu_pattern, text, re.IGNORECASE)


def extract_experience(text):
    """Extract work experience details."""
    experience = []
    exp_pattern = r"(?P<title>[\w\s]+)\s+at\s+(?P<company>[\w\s]+)\s+\((?P<years>[\d]{4}-[\d]{4})\)"
    matches = re.finditer(exp_pattern, text)

    for match in matches:
        experience.append({
            "title": match.group("title"),
            "company": match.group("company"),
            "years": match.group("years"),
        })

    return experience


def extract_skills(text, predefined_skills=None):
    """Extract skills from text."""
    if predefined_skills is None:
        predefined_skills = ["Python", "JavaScript", "React", "Node.js", "FastAPI", "SQL", "Machine Learning"]

    return [skill for skill in predefined_skills if skill.lower() in text.lower()]


def extract_certifications(text):
    """Extract certifications from text."""
    cert_pattern = r"(Certified|Certification|Certificate) in ([\w\s]+)"
    return [match[1] for match in re.findall(cert_pattern, text, re.IGNORECASE)]


def extract_projects(text):
    """Extract project details."""
    projects = []
    project_sections = re.split(r"(?i)projects", text)
    if len(project_sections) > 1:
        projects_text = project_sections[1]
        projects = projects_text.split("\n")[:5]
    return [proj.strip() for proj in projects if proj.strip()]


def parse_resume(file_path):
    """Main function to process resumes, including OCR for scanned documents."""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

    return {
        "personal_info": extract_personal_info(text),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "skills": extract_skills(text),
        "projects": extract_projects(text),
        "certifications": extract_certifications(text),
    }


# Example Usage
if __name__ == "__main__":
    file_path = "./data/sample_cvs/File.pdf"  # Change to your actual file
    parsed_data = parse_resume(file_path)
    print(parsed_data)
