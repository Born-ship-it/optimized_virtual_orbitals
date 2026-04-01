#!/usr/bin/env python3
"""
Advanced PDF to Text Extractor with multiple methods and quality comparison
"""

import os
import sys
from pathlib import Path

def extract_with_pdfplumber():
    """Extract using pdfplumber (current method)"""
    try:
        import pdfplumber
    except ImportError:
        return None, "pdfplumber not installed"
    
    articles_folder = Path(__file__).parent
    pdf_filename = "Optimized virtual orbital space for high‐level correlated calculations by Adamowicz, L. & Bartlett, R. J..pdf"
    pdf_path = articles_folder / pdf_filename
    
    if not pdf_path.exists():
        return None, "PDF not found"
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                full_text += f"\n--- PAGE {i} ---\n{text}\n"
        return full_text, "Success"
    except Exception as e:
        return None, str(e)

def extract_with_pypdf():
    """Extract using PyPDF2 (alternative method)"""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return None, "PyPDF2 not installed"
    
    articles_folder = Path(__file__).parent
    pdf_filename = "Optimized virtual orbital space for high‐level correlated calculations by Adamowicz, L. & Bartlett, R. J..pdf"
    pdf_path = articles_folder / pdf_filename
    
    if not pdf_path.exists():
        return None, "PDF not found"
    
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            full_text += f"\n--- PAGE {i} ---\n{text}\n"
        return full_text, "Success"
    except Exception as e:
        return None, str(e)

def extract_with_pymupdf():
    """Extract using PyMuPDF/fitz (often better for scanned PDFs)"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, "PyMuPDF not installed"
    
    articles_folder = Path(__file__).parent
    pdf_filename = "Optimized virtual orbital space for high‐level correlated calculations by Adamowicz, L. & Bartlett, R. J..pdf"
    pdf_path = articles_folder / pdf_filename
    
    if not pdf_path.exists():
        return None, "PDF not found"
    
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for i, page in enumerate(doc, 1):
            text = page.get_text()
            full_text += f"\n--- PAGE {i} ---\n{text}\n"
        doc.close()
        return full_text, "Success"
    except Exception as e:
        return None, str(e)

def extract_with_ocr():
    """Extract using Tesseract OCR (best for scanned PDFs but slower)"""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return None, "pytesseract/pdf2image not installed"
    
    articles_folder = Path(__file__).parent
    pdf_filename = "Optimized virtual orbital space for high‐level correlated calculations by Adamowicz, L. & Bartlett, R. J..pdf"
    pdf_path = articles_folder / pdf_filename
    
    if not pdf_path.exists():
        return None, "PDF not found"
    
    try:
        print("   Converting PDF to images...", end="\r")
        images = convert_from_path(pdf_path)
        
        full_text = ""
        for i, image in enumerate(images, 1):
            print(f"   OCR processing page {i}/{len(images)}...", end="\r")
            text = pytesseract.image_to_string(image)
            full_text += f"\n--- PAGE {i} ---\n{text}\n"
        
        print(f"   OCR complete!                    ")
        return full_text, "Success"
    except Exception as e:
        return None, str(e)

def compare_methods():
    """Compare all available extraction methods"""
    
    print("\n" + "="*70)
    print("PDF EXTRACTION METHOD COMPARISON")
    print("="*70)
    
    methods = [
        ("pdfplumber (current)", extract_with_pdfplumber),
        ("PyPDF2", extract_with_pypdf),
        ("PyMuPDF (fitz)", extract_with_pymupdf),
        ("Tesseract OCR", extract_with_ocr),
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"\n📋 Testing: {method_name}")
        text, status = method_func()
        
        if text:
            char_count = len(text)
            line_count = len(text.split('\n'))
            
            # Check for corruption indicators
            corruption_indicators = [
                text.count('~'),  # Tildes (encoding issue)
                text.count('ó'),  # Mangled characters
                sum(1 for c in text if c.isupper() and c not in text[:100]),  # Random caps
            ]
            corruption_score = sum(corruption_indicators)
            
            results[method_name] = {
                "success": True,
                "chars": char_count,
                "lines": line_count,
                "corruption": corruption_score,
                "status": status,
                "text": text
            }
            
            print(f"   ✓ {status}")
            print(f"   Characters: {char_count:,}")
            print(f"   Lines: {line_count:,}")
            print(f"   Corruption score: {corruption_score} (lower is better)")
            
        else:
            results[method_name] = {
                "success": False,
                "status": status
            }
            print(f"   ✗ {status}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    successful = {k: v for k, v in results.items() if v["success"]}
    
    if successful:
        best = min(successful.items(), key=lambda x: x[1]["corruption"])
        print(f"\n🏆 Best Method: {best[0]}")
        print(f"   Corruption Score: {best[1]['corruption']}")
        print(f"   Character Count: {best[1]['chars']:,}")
        
        print("\nRecommendations:")
        if best[1]["corruption"] < 50:
            print("   ✓ Current extraction quality is acceptable for reading")
        elif best[1]["corruption"] < 200:
            print("   ⚠ Consider manual review of technical sections")
        else:
            print("   ✗ Quality issues require significant manual correction")
        
        print("\nNext Steps:")
        print(f"   1. Try the recommended method: {best[0]}")
        print("   2. For this paper: Focus on key equations and tables")
        print("   3. For technical accuracy: Manual verification recommended")
    else:
        print("\n❌ No extraction methods succeeded. Install required libraries:")
        print("   pip install pdfplumber PyPDF2 PyMuPDF pdf2image pytesseract")

if __name__ == "__main__":
    compare_methods()
