"""
Script de OCR para PDF usando PaddleOCR.

Funciones principales:
- Convierte cada página del PDF a imagen (vía PyMuPDF) y aplica OCR (PaddleOCR).
- Imprime el texto reconocido por página en la consola.

Requisitos:
- paddlepaddle, paddleocr, pymupdf, opencv-python (ver requirements.txt)

Uso:
    python ocr_pdf.py --pdf ruta/al/archivo.pdf [--lang en] [--dpi 200] [--gpu]

Notas:
- El primer uso descargará automáticamente los modelos de PaddleOCR (puede tardar).
- En Windows, si la instalación de paddlepaddle falla, revise las notas al final.
"""

import argparse
import os
import sys
import shutil
from typing import List, Tuple

import numpy as np
import fitz  # PyMuPDF
import cv2
import re
import unicodedata

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("[ERROR] No se encontró PaddleOCR. Asegúrate de instalar dependencias: pip install -r requirements.txt")
    sys.exit(1)


def pdf_page_to_bgr(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """Renderiza una página de PDF a imagen BGR (OpenCV).

    - Usa PyMuPDF para rasterizar la página.
    - Desactiva alpha para obtener 3 canales.
    - Convierte de RGB a BGR para PaddleOCR.
    """
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img_rgb = np.frombuffer(pix.samples, dtype=np.uint8)
    img_rgb = img_rgb.reshape((pix.height, pix.width, 3))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def ocr_image_bgr(ocr: PaddleOCR, img_bgr: np.ndarray, min_rec_score: float = 0.0) -> List[str]:
    """Aplica OCR y devuelve una lista de textos reconocidos en la imagen.

    En PaddleOCR 3.x se usa `predict` y el resultado contiene claves como 'rec_texts'.
    Si `min_rec_score` > 0, filtra por el puntaje de reconocimiento para intentar
    ignorar trazos manuscritos o resultados poco confiables.
    """
    result = ocr.predict(img_bgr)
    texts: List[str] = []
    if not result:
        return texts
    res0 = result[0]
    if isinstance(res0, dict) and 'rec_texts' in res0:
        rec_texts = res0.get('rec_texts') or []
        rec_scores = res0.get('rec_scores') or [1.0] * len(rec_texts)
        for t, s in zip(rec_texts, rec_scores):
            if not isinstance(t, str):
                continue
            if min_rec_score and s < float(min_rec_score):
                continue
            texts.append(t)
    return texts


def run_ocr_pdf(pdf_path: str, lang: str = "latin", dpi: int = 300, use_gpu: bool = False):
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] Archivo no encontrado: {pdf_path}")
        sys.exit(1)

    print(f"[INFO] Inicializando PaddleOCR (lang={lang})...")
    # En PaddleOCR 3.x, use_angle_cls está deprecado; se recomienda use_textline_orientation.
    # El parámetro use_gpu ya no es aceptado; el dispositivo se gestiona internamente.
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    doc = fitz.open(pdf_path)
    print(f"[INFO] Páginas del PDF: {doc.page_count}")
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        img_bgr = pdf_page_to_bgr(page, dpi=dpi)
        texts = ocr_image_bgr(ocr, img_bgr)
        print(f"\n----- Página {page_index + 1} / {doc.page_count} -----")
        if texts:
            for line in texts:
                print(line)
        else:
            print("[AVISO] No se reconoció texto en esta página.")

# --- NUEVAS FUNCIONES: primera página y radicado ---

def extract_text_first_page(ocr: PaddleOCR, pdf_path: str, dpi: int = 200, min_score: float = 0.0) -> str:
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        return ""
    page = doc.load_page(0)
    img_bgr = pdf_page_to_bgr(page, dpi=dpi)
    texts = ocr_image_bgr(ocr, img_bgr, min_rec_score=min_score)
    return "\n".join(texts or [])


def extract_text_from_pdf(ocr: PaddleOCR, pdf_path: str, dpi: int = 200, min_score: float = 0.0) -> str:
    """Extrae texto de todo el PDF usando OCR y devuelve un único string."""
    doc = fitz.open(pdf_path)
    all_texts: List[str] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        img_bgr = pdf_page_to_bgr(page, dpi=dpi)
        texts = ocr_image_bgr(ocr, img_bgr, min_rec_score=min_score)
        if texts:
            all_texts.extend(texts)
    return "\n".join(all_texts)

def ensure_unique_path(directory: str, filename: str) -> str:
    """Genera una ruta única en 'directory' para 'filename' si ya existe."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    idx = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}_{idx}{ext}")
        idx += 1
    return candidate

def process_directory(dir_path: str, keyword: str = "sentencia", out_dir: str | None = None,
                      lang: str = "latin", dpi: int = 200, use_gpu: bool = False) -> Tuple[List[str], List[str]]:
    """
    Procesa todos los PDFs de 'dir_path'. Si el texto OCR contiene 'keyword', mueve el archivo a 'out_dir'.

    Devuelve (moved, not_moved) con las rutas de archivos movidos y no movidos.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directorio no encontrado: {dir_path}")

    if out_dir is None:
        out_dir = os.path.join(dir_path, "procesados")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Inicializando PaddleOCR (lang={lang})...")
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    moved: List[str] = []
    not_moved: List[str] = []
    key_lower = keyword.lower()

    pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("[AVISO] No se encontraron archivos PDF en el directorio.")

    for filename in sorted(pdf_files):
        src_path = os.path.join(dir_path, filename)
        print(f"[INFO] Procesando: {src_path}")
        try:
            text = extract_text_from_pdf(ocr, src_path, dpi=dpi)
        except Exception as e:
            print(f"[ERROR] Falló OCR para {src_path}: {e}")
            not_moved.append(src_path)
            continue

        if key_lower in text.lower():
            dst_path = ensure_unique_path(out_dir, filename)
            try:
                shutil.move(src_path, dst_path)
                print(f"[OK] Coincidencia encontrada. Archivo movido a: {dst_path}")
                moved.append(dst_path)
            except Exception as e:
                print(f"[ERROR] No se pudo mover {src_path} -> {dst_path}: {e}")
                not_moved.append(src_path)
        else:
            print("[INFO] No contiene la palabra clave; no se mueve.")
            not_moved.append(src_path)

    return moved, not_moved


def parse_args():
    parser = argparse.ArgumentParser(description="OCR de PDF con PaddleOCR")
    parser.add_argument("--pdf", required=False, help="Ruta al archivo PDF")
    parser.add_argument("--dir", required=False, help="Directorio con archivos PDF a procesar")
    parser.add_argument("--lang", default="en", help="Idioma del modelo (ej. 'en', 'ch', 'latin')")
    parser.add_argument("--dpi", type=int, default=200, help="Resolución de rasterizado de las páginas")
    parser.add_argument("--gpu", action="store_true", help="Usar GPU si está disponible")
    parser.add_argument("--keyword", default="sentencia", help="Palabra/frase a buscar en el texto OCR (modo carpeta)")
    parser.add_argument("--out-dir", default=None, help="Carpeta destino para mover coincidencias (por defecto '<dir>/procesados')")
    parser.add_argument("--predial-rename", action="store_true", help="Buscar 'predial' en la primera página y renombrar a predial_{numero}.pdf")
    parser.add_argument("--predial-out-dir", default=None, help="Carpeta destino para renombrados del modo predial (por defecto '<dir>/predial')")
    parser.add_argument("--min-score", type=float, default=0.0, help="Puntaje mínimo de reconocimiento para filtrar líneas (ej. 0.8 para ignorar manuscritos)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if getattr(args, "predial_rename", False) and args.dir:
            renamed, skipped = process_predial_rename(
                dir_path=args.dir,
                lang=args.lang,
                dpi=args.dpi,
                out_dir=args.predial_out_dir,
                min_score=args.min_score,
            )
            print("\n===== Resumen (Predial-Renombrar) =====")
            print(f"Renombrados ({len(renamed)}):")
            for src, dst in renamed:
                print(f" - {os.path.basename(src)} -> {os.path.basename(dst)}")
            print(f"Omitidos ({len(skipped)}):")
            for src, motivo in skipped:
                print(f" - {os.path.basename(src)} ({motivo})")
        elif args.dir:
            moved, not_moved = process_directory(
                dir_path=args.dir,
                keyword=args.keyword,
                out_dir=args.out_dir,
                lang=args.lang,
                dpi=args.dpi,
                use_gpu=args.gpu,
            )
            print("\n===== Resumen =====")
            print(f"Movidos ({len(moved)}):")
            for p in moved:
                print(f" - {p}")
            print(f"No movidos ({len(not_moved)}):")
            for p in not_moved:
                print(f" - {p}")
        elif args.pdf:
            run_ocr_pdf(args.pdf, lang=args.lang, dpi=args.dpi, use_gpu=args.gpu)
        else:
            print("[ERROR] Debes proporcionar --pdf o --dir.")
            sys.exit(1)
    except Exception as e:
        print("[ERROR] Ocurrió un error durante el OCR:", e)
        print("""\nSugerencias:
- Verifica que el PDF no esté corrupto y que tienes permisos de lectura.
- Prueba con un dpi más alto (p. ej., --dpi 300) para mejorar la calidad.
- Si la instalación de paddlepaddle falla en Windows, revisa la guía oficial de instalación y asegúrate de tener las dependencias del sistema (Visual C++ Redistributable).\n""")
        sys.exit(1)