import pdfplumber

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text