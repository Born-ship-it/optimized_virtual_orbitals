#!/usr/bin/env python3
"""
Batch PDF to Text Extractor
Uses PyPDF2 as primary method (best quality) with fallback options
Accepts PDF filename or pattern as command-line argument
"""

import sys
from pathlib import Path

def extract_pdf_text(pdf_filename):
    """Extract text from a single PDF using best available method"""
    
    # Set paths - script is in scripts/ subdirectory, need to go up to articles/
    script_location = Path(__file__).parent  # scripts/ folder
    articles_folder = script_location.parent.parent  # Go up to articles/
    output_folder = articles_folder / "pdf_text_extracts"
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    # Find the PDF
    pdf_path = articles_folder / pdf_filename
    
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return None
    
    print(f"📄 Processing: {pdf_filename}")
    
    # Try PyPDF2 first (best quality)
    try:
        from PyPDF2 import PdfReader
        print("   Using PyPDF2 (best quality)...")
        
        reader = PdfReader(pdf_path)
        full_text = ""
        total_pages = len(reader.pages)
        
        print(f"   Total pages: {total_pages}")
        
        for i, page in enumerate(reader.pages, 1):
            print(f"   Extracting page {i}/{total_pages}...", end="\r")
            text = page.extract_text()
            full_text += f"\n--- PAGE {i} ---\n{text}\n"
        
        print(f"   ✓ PyPDF2 extraction complete!     ")
        method_used = "PyPDF2"
        
    except ImportError:
        print("   PyPDF2 not available, trying PyMuPDF...")
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            full_text = ""
            total_pages = len(doc)
            
            print(f"   Total pages: {total_pages}")
            
            for i, page in enumerate(doc, 1):
                print(f"   Extracting page {i}/{total_pages}...", end="\r")
                text = page.get_text()
                full_text += f"\n--- PAGE {i} ---\n{text}\n"
            
            doc.close()
            print(f"   ✓ PyMuPDF extraction complete!    ")
            method_used = "PyMuPDF"
            
        except ImportError:
            print("   PyMuPDF not available, falling back to pdfplumber...")
            
            try:
                import pdfplumber
                
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ""
                    total_pages = len(pdf.pages)
                    
                    print(f"   Total pages: {total_pages}")
                    
                    for i, page in enumerate(pdf.pages, 1):
                        print(f"   Extracting page {i}/{total_pages}...", end="\r")
                        text = page.extract_text()
                        full_text += f"\n--- PAGE {i} ---\n{text}\n"
                    
                    print(f"   ✓ pdfplumber extraction complete!")
                    method_used = "pdfplumber"
                    
            except ImportError:
                print("   ERROR: No PDF extraction library available")
                print("   Install one: pip install PyPDF2 PyMuPDF pdfplumber")
                return None
    
    # Generate output filename from PDF name (remove .pdf, add .txt)
    # For files like "Paper by Author.pdf" -> "Paper_by_Author.txt"
    # Use only first 50 chars + .txt to avoid path length issues with special chars
    safe_name = pdf_filename.replace(".pdf", "").replace(" & ", "_").replace("Ø", "O").replace("ö", "o")
    if len(safe_name) > 60:
        safe_name = safe_name[:50] + "_doc"
    output_filename = safe_name + ".txt"
    output_path = output_folder / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\n✅ Text saved to: {output_path}")
    print(f"   File size: {len(full_text):,} characters")
    print(f"   Method used: {method_used}")
    print(f"   Pages extracted: {total_pages}\n")
    
    return str(output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_batch.py <pdf_filename>")
        print("Example: python extract_pdf_batch.py 'My Paper by Author.pdf'")
        sys.exit(1)
    
    pdf_filename = sys.argv[1]
    extract_pdf_text(pdf_filename)
