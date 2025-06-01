import fitz  # PyMuPDF
from docx import Document

def pdf_to_doc(pdf_path, doc_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Create a new Word Document
    doc = Document()
    
    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        
        # Add page text to the Word document
        doc.add_paragraph(text)
    
    # Save the Word document
    doc.save(doc_path)
    print(f"Converted '{pdf_path}' to '{doc_path}'")

# Example usage
pdf_to_doc(r"C:\Users\sanik\Downloads\Title Defense Evaluation Report.pdf", r"C:\Users\sanik\Downloads\output.docx")

