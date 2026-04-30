#!/usr/bin/env python3
"""
Brueckner Paper PDF to Text Extractor
Uses PyPDF2 as primary method (best quality) with fallback options
"""

import sys
from pathlib import Path

def extract_brueckner_pdf():
    """Extract text from Brueckner PDF using best available method"""
    
    # Set paths
    script_location = Path(__file__).parent  # scripts/ folder
    articles_folder = script_location.parent.parent  # Go up to articles/
    output_folder = articles_folder / "pdf_text_extracts"
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    # Find the Brueckner PDF (note the typo in the filename)
    pdf_filename = "Original Brueclner Orbitals.pdf"
    pdf_path = articles_folder / pdf_filename
    
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)
    
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
                sys.exit(1)
    
    # Save to text file
    output_filename = "Brueckner1954_OriginalPaper.txt"
    output_path = output_folder / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\n✅ Text saved to: {output_path}")
    print(f"   File size: {len(full_text):,} characters")
    print(f"   Method used: {method_used}")
    print(f"   Pages extracted: {total_pages}")

if __name__ == "__main__":
    extract_brueckner_pdf()
