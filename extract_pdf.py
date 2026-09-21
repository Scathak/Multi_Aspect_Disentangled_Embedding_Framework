import fitz  # PyMuPDF
from schema import PaperMetadata

def parse_pdf_to_metadata(pdf_path: str, paper_id: str) -> PaperMetadata:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Simple section parser heuristic (Can be swapped with GROBID for higher precision)
    title = full_text.split("\n")[0]  # First line as title heuristic
    
    # Extract sections using common headings
    abstract = extract_section(full_text, "ABSTRACT", "INTRODUCTION")
    methods = extract_section(full_text, "METHOD", "RESULTS")
    results = extract_section(full_text, "RESULTS", "DISCUSSION")

    return PaperMetadata(
        paper_id=paper_id,
        title=title if title else "Unknown Title",
        abstract=abstract if abstract else full_text[:1000], # Fallback
        methods_text=methods if methods else full_text[1000:3000],
        results_text=results if results else full_text[3000:5000],
        code_url="",
        citations=[],
        authors=[],
        venue="Custom Upload",
        publication_year=2024
    )

def extract_section(text: str, start_kw: str, end_kw: str) -> str:
    lower_text = text.lower()
    start_idx = lower_text.find(start_kw.lower())
    end_idx = lower_text.find(end_kw.lower(), start_idx)
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx].strip()
    return ""