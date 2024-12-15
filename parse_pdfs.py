from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.
    :param file_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Example usage
if __name__ == "__main__":
    # Replace these with the paths to your PDFs
    pdf_paths = [
        "Alphabet_Form_10K.pdf",
        "Tesla_Form_10K.pdf",
        "Uber_Form_10K.pdf"
    ]

    # Extract text from each PDF
    for path in pdf_paths:
        print(f"Extracting text from: {path}")
        try:
            pdf_text = extract_text_from_pdf(path)
            print(f"Extracted {len(pdf_text)} characters.\n")
        except FileNotFoundError:
            print(f"File not found: {path}")
