"""
Extract raw text from an uploaded CV PDF, same tool the original paper used.
"""
import pdfplumber


def extract_text_from_pdf(file_obj) -> str:
    """file_obj: a file path (str) or a file-like object (e.g. Streamlit UploadedFile)."""
    text_parts = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)
