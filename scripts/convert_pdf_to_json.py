import fitz  # PyMuPDF
import json
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from the PDF and structures it into JSON format."""
    doc = fitz.open(pdf_path)
    sections = []
    current_section = None

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            match = re.match(r"Section (\d+): (.+)", line.strip())
            if match:
                if current_section:
                    sections.append(current_section)
                current_section = {"section": int(match.group(1)), "title": match.group(2), "content": []}
            elif current_section:
                current_section["content"].append(line.strip())

    if current_section:
        sections.append(current_section)

    return sections

# Convert PDF to JSON
pdf_path = "../data/d:\Faizan\Chatbot\Income Tax Ordinance, 2001 all files\Income Tax Ordinance, 2001 amended upto 30th June 2024.pdf"
tax_data = extract_text_from_pdf(pdf_path)

with open("../data/income_tax_ordinance_2001.json", "w", encoding="utf-8") as f:
    json.dump(tax_data, f, indent=4, ensure_ascii=False)

print("✅ PDF converted to JSON!")
