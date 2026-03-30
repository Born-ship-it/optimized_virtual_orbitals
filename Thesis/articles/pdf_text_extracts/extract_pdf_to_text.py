#!/usr/bin/env python3
"""
PDF to Text Extractor
Extracts text from PDF files and saves to text format
"""

import os
import sys
from pathlib import Path

def extract_pdf_text():
    """Extract text from PDFs in articles folder"""
    
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not installed")
        print("Install it using one of:")
        print("  - pip install pdfplumber")
        print("  - uv add pdfplumber")
        sys.exit(1)
    
    # Set paths
    articles_folder = Path(__file__).parent
    output_folder = articles_folder / "pdf_text_extracts"
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    # Find the Adamowicz PDF
    pdf_filename = "Optimized virtual orbital space for high‐level correlated calculations by Adamowicz, L. & Bartlett, R. J..pdf"
    pdf_path = articles_folder / pdf_filename
    
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)
    
    print(f"📄 Processing: {pdf_filename}")
    
    try:
        # Extract text using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            total_pages = len(pdf.pages)
            
            print(f"   Total pages: {total_pages}")
            
            for i, page in enumerate(pdf.pages, 1):
                print(f"   Extracting page {i}/{total_pages}...", end="\r")
                text = page.extract_text()
                full_text += f"\n--- PAGE {i} ---\n{text}\n"
            
            print(f"   ✓ Extraction complete!                    ")
        
        # Save to text file
        output_filename = "Adamowicz1987_OVOS.txt"
        output_path = output_folder / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"\n✅ Text saved to: {output_path}")
        print(f"   File size: {len(full_text)} characters")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    extract_pdf_text()
