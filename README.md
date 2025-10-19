# OCR Docs Renamer – PaddleOCR PDF Text Extraction

This project provides a simple Python script to:
- Extract text from PDF files using PaddleOCR (v3.x).
- Process an entire directory of PDFs, search for a keyword in the OCR text, and move matching files to a destination folder.

It uses PyMuPDF to rasterize pages to images and PaddleOCR to recognize text.

## Requirements
- Python 3.9+ (tested on Windows with Python 3.12)
- Internet connection on first run (models are downloaded automatically)
- Optional on Windows: Microsoft Visual C++ Redistributable

## Setup (Windows)
1) Create and activate a virtual environment:
```
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies:
```
pip install -r requirements.txt
```

## Usage
There are two modes: single PDF and directory processing.

### 1) Single PDF – print recognized text
```
.\.venv\Scripts\python .\ocr_pdf.py --pdf ".\docs\file.pdf" --lang en --dpi 200
```
- --pdf: path to the PDF file
- --lang: OCR language (default: en). In PaddleOCR 3.x, use "en" or other supported languages; "latin" is not available.
- --dpi: page rasterization DPI (default: 200). For low-quality scans, try 300.

### 2) Directory mode – move files containing a keyword
Process all PDFs in a folder, and move those that contain the keyword in the OCR text to a destination folder.
```
.\.venv\Scripts\python .\ocr_pdf.py --dir ".\docs" --keyword "sentencia" --lang en --dpi 200
```
- --dir: directory containing PDFs
- --keyword: text to search within the OCR result (default: "sentencia")
- --out-dir: destination folder for matched files (default: "<dir>\\procesados"). The folder will be created if it does not exist.
- --lang: OCR language (default: en)
- --dpi: page rasterization DPI (default: 200)
- --gpu: accepted by the CLI for backward compatibility, but PaddleOCR 3.x manages device selection internally and this flag is currently not used.

Example with custom destination:
```
.\.venv\Scripts\python .\ocr_pdf.py --dir ".\docs" --keyword "sentencia" --out-dir ".\processed" --lang en --dpi 300
```

## Notes
- First execution may download PP-OCRv5 models automatically; give it a moment.
- If recognition quality is poor, increase `--dpi` to 300.
- If you see an error like "No models are available for the language ...", switch `--lang` to a supported value (e.g., `en`).
- PaddleOCR 3.x updated its API compared to 2.x; this script uses `predict()` and extracts text from `rec_texts`.

## Project Structure
```
ocr_docs_renamer/
├── .venv/
├── docs/
│   ├── file.pdf
│   └── procesados/
│       ├── file2.pdf
│       └── file3.pdf
├── ocr_pdf.py
└── requirements.txt
```

## License
This project is licensed under the MIT License.

## Contact
Developed by:
- Paul Realpe
- Jimmy Realpe

Email: co.devpaul@gmail.com

Phone: 3148580454

Website: https://devpaul.pro/

Feel free to reach out for any inquiries or collaborations!