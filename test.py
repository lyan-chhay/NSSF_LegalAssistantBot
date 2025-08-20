import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import os
import argparse
from PIL import Image
import re

def extract_khmer_text_from_pdf(pdf_path, output_txt=None, dpi=300):
    """
    Extract Khmer text from a PDF file using Tesseract OCR
    
    Args:
        pdf_path (str): Path to the PDF file
        output_txt (str, optional): Path to output text file. If None, prints to console
        dpi (int): DPI for image conversion (higher is better quality but slower)
    
    Returns:
        str: Extracted text from the PDF
    """
    
    # Validate input file
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Check if Tesseract is installed and Khmer language is available
    try:
        tess_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {tess_version}")
    except:
        raise Exception("Tesseract not installed or not in PATH. Please install it.")
    
    # Check for Khmer language support
    languages = pytesseract.get_languages(config='')
    if 'khm' not in languages:
        raise Exception("Khmer language pack (khm) not installed for Tesseract.")
    
    print("Extracting text from PDF...")
    
    # Extract text using two methods:
    # 1. First try to extract directly if it's a text-based PDF
    # 2. If that fails or returns little text, use OCR
    
    text_content = ""
    
    # Method 1: Try direct text extraction (for text-based PDFs)
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n\n"
        
        # Check if we got meaningful text (not just a few characters)
        khmer_chars = re.findall(r'[\u1780-\u17FF]+', text_content)
        if len(khmer_chars) > 3:  # If we found several Khmer words
            print("Text-based PDF detected. Using direct text extraction.")
            
            if output_txt:
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"Text saved to: {output_txt}")
            
            return text_content
    except Exception as e:
        print(f"Direct text extraction failed: {e}. Switching to OCR.")
        text_content = ""  # Reset content for OCR method
    
    # Method 2: OCR-based extraction (for scanned PDFs or image-based PDFs)
    print("Scanned PDF detected. Using OCR extraction...")
    
    # Convert PDF to images
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        raise Exception(f"Failed to convert PDF to images: {e}")
    
    print(f"Converted {len(images)} pages to images for processing")
    
    # Process each image with Tesseract
    full_text = ""
    for i, image in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}...")
        
        # Preprocess image to improve OCR accuracy
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Use Tesseract to extract Khmer text
        page_text = pytesseract.image_to_string(
            image, 
            lang='khm',  # Khmer language
            config='--psm 3'  # Page segmentation mode: fully automatic
        )
        
        full_text += f"--- Page {i+1} ---\n{page_text}\n\n"
    
    # Save to file if output path provided
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Extracted text saved to: {output_txt}")
    
    return full_text

def main():
    parser = argparse.ArgumentParser(description='Extract Khmer text from PDF using Tesseract OCR')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', help='Output text file path (optional)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for image conversion (default: 300)')
    
    args = parser.parse_args()
    
    try:
        text = extract_khmer_text_from_pdf(args.pdf_path, args.output, args.dpi)
        
        if not args.output:
            # Print first 1000 characters if no output file specified
            print("\nExtracted Text Preview:")
            print(text[:1000] + "..." if len(text) > 1000 else text)
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Tesseract is installed: https://github.com/tesseract-ocr/tesseract")
        print("2. Install Khmer language pack: https://github.com/tesseract-ocr/tessdata/blob/main/khm.traineddata")
        print("3. Place khm.traineddata in Tesseract's tessdata directory")
        print("4. Install required Python packages: pip install pytesseract pdf2image Pillow PyPDF2")

if __name__ == "__main__":
    main()