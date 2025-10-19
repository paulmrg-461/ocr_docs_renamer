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

# Helper para inicializar PaddleOCR con fallback si el idioma solicitado no está disponible

def create_ocr(lang: str) -> PaddleOCR:
    try:
        return PaddleOCR(use_textline_orientation=True, lang=lang)
    except Exception as e:
        print(f"[AVISO] Falló inicialización de PaddleOCR con lang='{lang}': {e}")
        # Fallbacks recomendados: 'es' (español) y 'en'
        fallbacks = ["es", "en"] if lang.lower() == "latin" else ["en"]
        for fb in fallbacks:
            try:
                print(f"[INFO] Intentando fallback lang='{fb}'...")
                return PaddleOCR(use_textline_orientation=True, lang=fb)
            except Exception as e2:
                print(f"[AVISO] Falló fallback lang='{fb}': {e2}")
        # Si no se logró, relanza el error original
        raise


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
    ocr = create_ocr(lang)

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

# --- Utilidades de normalización y extracción de números ---

def _normalize_text(s: str) -> str:
    """Normaliza texto: quita acentos, colapsa espacios y recorta."""
    if not isinstance(s, str):
        return ""
    # Normaliza a NFKD y elimina marcas diacríticas
    s_norm = unicodedata.normalize("NFKD", s)
    s_no_accents = "".join(c for c in s_norm if not unicodedata.combining(c))
    # Reemplaza espacios no separables y colapsa espacios
    s_no_nbsp = s_no_accents.replace("\u00A0", " ")
    s_clean = re.sub(r"[ \t]+", " ", s_no_nbsp)
    return s_clean.strip()


def find_radicado_number(text: str) -> str | None:
    """
    Extrae número priorizando:
    1) 'Factura' (si aparece), dígitos >= 6
    2) 'Radicación/Radicado' dígitos >= 12
    3) 'Referencia'/'Referencia del Predio' dígitos >= 12
    4) Fallback: secuencia más larga de dígitos (>= 12)
    """
    if not text:
        return None
    norm = _normalize_text(text)

    # 1) Factura
    patterns_invoice = [
        r"(?i)factura\s*(?:n[oº]\.??|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{5,})",
    ]
    for pat in patterns_invoice:
        m = re.search(pat, norm)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) >= 6:
                return digits

    # 2) Radicación/Radicado
    patterns_radicado = [
        r"(?i)radicaci[oó]n\s*(?:n[oº]\.??|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{8,})",
        r"(?i)radicado\s*(?:n[oº]\.??|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{8,})",
    ]
    for pat in patterns_radicado:
        m = re.search(pat, norm)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) >= 12:
                return digits

    # 3) Referencia / Referencia del Predio
    patterns_ref = [
        r"(?i)referencia(?:\s+del\s+predio)?\s*(?:n[oº]\.??|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{8,})",
        r"(?i)referencia\s*(?:catastral|predio|predial)?\s*(?:n[oº]\.??|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{8,})",
    ]
    for pat in patterns_ref:
        m = re.search(pat, norm)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) >= 12:
                return digits

    # 4) Fallback: mayor secuencia de dígitos
    longest = ""
    for m in re.finditer(r"(\d{12,})", norm):
        seq = m.group(1)
        if len(seq) > len(longest):
            longest = seq
    return longest if len(longest) >= 12 else None

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
    ocr = create_ocr(lang)

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


def find_radicado_number_strict(text: str) -> str | None:
    """
    Extrae SOLO números asociados a Radicación/Radicado.
    No prioriza ni usa números de 'Factura'. Devuelve dígitos si longitud >= 12.
    """
    candidates: list[str] = []
    patterns_radicado = [
        r"(?i)radicaci[oó]n\s*(?:n[oº]\.?|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{8,})",
        r"(?i)radicado\s*(?:n[oº]\.?|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{8,})",
    ]
    for pat in patterns_radicado:
        m = re.search(pat, text)
        if m:
            candidates.append(m.group(1))
    for m in re.finditer(r"(?i)radicaci[oó]n|radicado", text):
        segment = text[m.end(): m.end() + 120]
        m2 = re.search(r"([0-9][\d\-\.\s]{8,})", segment)
        if m2:
            candidates.append(m2.group(1))
    for cand in candidates:
        digits = re.sub(r"\D", "", cand)
        if len(digits) >= 12:
            return digits
    return None


def find_expediente_number(text: str) -> str | None:
    """
    Extrae número de 'Expediente'. Devuelve dígitos si longitud >= 6.
    Acepta variaciones: "Expediente No", "Nº", "Numero", "Num.", "#".
    """
    candidates: list[str] = []
    patterns_exp = [
        r"(?i)expediente\s*(?:n[oº]\.?|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{5,})",
    ]
    for pat in patterns_exp:
        m = re.search(pat, text)
        if m:
            candidates.append(m.group(1))
    for m in re.finditer(r"(?i)expediente", text):
        segment = text[m.end(): m.end() + 120]
        m2 = re.search(r"([0-9][\d\-\.\s]{5,})", segment)
        if m2:
            candidates.append(m2.group(1))
    for cand in candidates:
        digits = re.sub(r"\D", "", cand)
        if len(digits) >= 6:
            return digits
    return None


def find_invoice_number_official(text: str) -> str | None:
    """
    Extrae número de 'Factura oficial' evitando confundirlo con NIT.
    Busca variaciones de indicador: "No", "Nº", "N°", "Nro", "Numero", "Num.", "#",
    ancladas a la frase "Factura oficial" y dentro de una ventana cercana.
    Devuelve solo dígitos si longitud >= 6.
    """
    norm = _normalize_text(text).lower()
    candidates: list[str] = []

    # Patrón principal: 'factura oficial' seguido de No/Nº/N°/Nro/Numero/Num/# y el número
    pat_main = r"(?:factura\s+oficial(?:es)?[^\n]{0,120}?(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([0-9][\d\-\.\s]{5,}))"
    for m in re.finditer(pat_main, norm):
        # Si cerca aparece 'nit', descartamos (probable NIT de entidad)
        window = norm[max(0, m.start()-40): m.end()+40]
        if "nit" in window:
            continue
        candidates.append(m.group(1))

    # Fallback: localizar 'factura oficial' y buscar el indicador en la vecindad
    for mf in re.finditer(r"factura\s+oficial(?:es)?", norm):
        segment = norm[mf.end(): mf.end() + 300]
        m2 = re.search(r"(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([0-9][\d\-\.\s]{5,})", segment)
        if m2:
            window = segment[max(0, m2.start()-30): m2.end()+30]
            if "nit" in window:
                continue
            candidates.append(m2.group(1))

    # Post-proceso: devolver el primer candidato válido
    for cand in candidates:
        digits = re.sub(r"\D", "", cand)
        if len(digits) >= 6:
            return digits
    return None


def find_resolution_mandamiento_pago(text: str) -> str | None:
    """
    Extrae el número/código de la "Resolución de mandamiento de pago" en la primera página.
    - Busca "resolución" cerca de "mandamiento de pago" con indicadores: No, Nº, N°, Nro, Numero, Num., #
    - Devuelve un token saneado (A-Z, 0-9, '-', '.') y en mayúsculas.
    """
    norm = _normalize_text(text).lower()
    candidates: list[str] = []

    # Patrón principal: resolucion ~ mandamiento de pago + indicador y número/código
    pat = r"(?:resoluci[oó]n[^\n]{0,150}?mandamiento\s+de\s+pago|mandamiento\s+de\s+pago[^\n]{0,150}?resoluci[oó]n)[^\n]{0,150}?(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})"
    for m in re.finditer(pat, norm):
        candidates.append(m.group(1))

    # Fallback 1: localizar "mandamiento de pago" y luego buscar "resolución ... No" con número
    for mp in re.finditer(r"mandamiento\s+de\s+pago", norm):
        segment = norm[mp.end(): mp.end() + 220]
        m2 = re.search(r"resoluci[oó]n[^\n]{0,150}?(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})", segment)
        if m2:
            candidates.append(m2.group(1))

    # Fallback 2: solo indicador cercano tras "mandamiento de pago"
    if not candidates:
        for mp in re.finditer(r"mandamiento\s+de\s+pago", norm):
            segment = norm[mp.end(): mp.end() + 220]
            m3 = re.search(r"(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})", segment)
            if m3:
                candidates.append(m3.group(1))

    # Sanitizar y validar
    for cand in candidates:
        c = re.sub(r"[ ]+", "", cand)
        c = c.upper().replace("/", "-").replace("\\", "-").replace(":", "-")
        c = re.sub(r"[^A-Z0-9\-.]", "", c).strip(".-")
        # exigir al menos 5 caracteres alfanuméricos totales
        if len(re.sub(r"[^A-Z0-9]", "", c)) >= 5:
            return c
    return None


def find_invoice_number(text: str) -> str | None:
    """
    Extrae número de 'Factura'. Devuelve dígitos si longitud >= 6.
    Acepta variaciones: "Factura No", "Nº", "Numero", "Num.", "#".
    """
    norm = _normalize_text(text)
    patterns_invoice = [
        r"(?i)factura\s*(?:n[oº]\.??|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{5,})",
    ]
    for pat in patterns_invoice:
        m = re.search(pat, norm)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) >= 6:
                return digits
    # búsqueda cercana tras la palabra 'Factura'
    for m in re.finditer(r"(?i)factura", norm):
        segment = norm[m.end(): m.end() + 120]
        m2 = re.search(r"([0-9][\d\-\.\s]{5,})", segment)
        if m2:
            digits = re.sub(r"\D", "", m2.group(1))
            if len(digits) >= 6:
                return digits
    return None


def find_resolution_acuerdo_pago(text: str) -> str | None:
    """
    Extrae el número/código de la "Resolución de acuerdo de pago" en la primera página.
    - Busca "resolución" cerca de "acuerdo de pago" (o "acuerdo de pago de impuesto") con indicadores: No, Nº, N°, Nro, Numero, Num., #
    - Devuelve un token saneado (A-Z, 0-9, '-', '.') y en mayúsculas.
    """
    norm = _normalize_text(text).lower()
    candidates: list[str] = []

    # Patrón principal: resolucion ~ acuerdo de pago + indicador y número/código
    pat = r"(?:resoluci[oó]n[^\n]{0,150}?acuerdo\s+de\s+pago(?:\s+de\s+impuesto)?|acuerdo\s+de\s+pago(?:\s+de\s+impuesto)?[^\n]{0,150}?resoluci[oó]n)[^\n]{0,150}?(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})"
    for m in re.finditer(pat, norm):
        candidates.append(m.group(1))

    # Fallback 1: localizar "acuerdo de pago" y luego buscar "resolución ... No" con número
    for ap in re.finditer(r"acuerdo\s+de\s+pago(?:\s+de\s+impuesto)?", norm):
        segment = norm[ap.end(): ap.end() + 220]
        m2 = re.search(r"resoluci[oó]n[^\n]{0,150}?(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})", segment)
        if m2:
            candidates.append(m2.group(1))

    # Fallback 2: solo indicador cercano tras "acuerdo de pago"
    if not candidates:
        for ap in re.finditer(r"acuerdo\s+de\s+pago(?:\s+de\s+impuesto)?", norm):
            segment = norm[ap.end(): ap.end() + 220]
            m3 = re.search(r"(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})", segment)
            if m3:
                candidates.append(m3.group(1))

    # Sanitizar y validar
    for cand in candidates:
        c = re.sub(r"[ ]+", "", cand)
        c = c.upper().replace("/", "-").replace("\\", "-").replace(":", "-")
        c = re.sub(r"[^A-Z0-9\-.]", "", c).strip(".-")
        if len(re.sub(r"[^A-Z0-9]", "", c)) >= 5:
            return c
    return None


# General extractor for 'Resolución' tokens (used for RREX and similar)
# Looks for 'resolución' or 'res.' followed by an indicator (No, Nº, N°, Nro, Numero, Num., #)
# and a code/number token. Sanitizes to uppercase and replaces slashes with hyphens.
# Returns None if no plausible token (>=5 alphanumerics) is found.

def find_resolution_general(text: str) -> str | None:
    norm = _normalize_text(text).lower()
    candidates: list[str] = []

    # Main pattern: 'resolución' anywhere on the page with optional indicator and code
    pat_main = r"resoluci[oó]n[^\n]{0,150}?(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)?\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})"
    for m in re.finditer(pat_main, norm):
        candidates.append(m.group(1))

    # Abbreviation 'Res.'
    pat_abbr = r"\bres\.?\s*(?:nro|n°|nº|no(?:\.)?|numero|num(?:\.)?|#)?\s*[:\-]?\s*([a-z0-9][a-z0-9\-\.\/]{4,})"
    for m in re.finditer(pat_abbr, norm):
        candidates.append(m.group(1))

    # Sanitize candidates
    for cand in candidates:
        c = re.sub(r"[ ]+", "", cand)
        c = c.upper().replace("/", "-").replace("\\", "-").replace(":", "-")
        c = re.sub(r"[^A-Z0-9\-.]", "", c).strip(".-")
        if len(re.sub(r"[^A-Z0-9]", "", c)) >= 5:
            return c
    return None


def find_radicado_near_tokens(text: str) -> str | None:
    """
    Extrae el número de radicado alrededor de las palabras 'Radicación' o 'Radicado',
    permitiendo que el número aparezca ANTES o DESPUÉS del término.
    - Soporta formas: "RADICACION {numero}", "{numero} RADICACION", "Radicado No {numero}".
    - Devuelve dígitos si longitud >= 12.
    """
    if not text:
        return None
    norm = _normalize_text(text)
    candidates: list[str] = []
    for m in re.finditer(r"(?i)radicaci[oó]n|radicado", norm):
        # Búsqueda después del término
        seg_after = norm[m.end(): m.end() + 140]
        m_after = re.search(r"(?:n[oº]\.??|no\.?|numero|#|num\.?)?\s*[:\-]?\s*([0-9][\d\-\.\s]{8,})", seg_after)
        if m_after:
            candidates.append(m_after.group(1))
        # Búsqueda antes del término (número inmediatamente antes)
        seg_before = norm[max(0, m.start() - 80): m.start()]
        m_before = re.search(r"([0-9][\d\-\.\s]{8,})\s*$", seg_before)
        if m_before:
            candidates.append(m_before.group(1))
    for cand in candidates:
        digits = re.sub(r"\D", "", cand)
        if len(digits) >= 12:
            return digits
    return None


def is_cartelera_hint(text: str) -> bool:
    norm = _normalize_text(text).lower()
    has_radicado = bool(re.search(r"(?i)radicaci[oó]n|radicado", norm))
    if "carteler" in norm:
        return True
    hints = 0
    for token in ["publique", "publicese", "circular", "certifico", "alcaldia"]:
        if token in norm:
            hints += 1
    return has_radicado and hints >= 2


def process_classify_rename(
    dir_path: str,
    lang: str = "latin",
    dpi: int = 300,
    out_root: str | None = None,
    min_score: float = 0.0,
    allow_fallback: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Clasifica y renombra PDFs según palabras clave en la PRIMERA página:
    - predial -> predial_{numero}  (prioriza 'Factura', luego Radicado, luego Referencia)
    - cartelera -> cartelera_{radicado} (solo radicado)
    - nota de secretaria -> nota_secretaria_{radicado} (solo radicado)
    - expediente -> expediente_{numero_expediente}

    Mueve los archivos a subcarpetas dentro `out_root` (predial, cartelera, nota_secretaria, expediente).

    Devuelve (renamed, skipped): listas de (src, dst) y (src, motivo).
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directorio no encontrado: {dir_path}")

    if out_root is None:
        out_root = dir_path

    subdirs = {
        "predial": os.path.join(out_root, "predial"),
        "cartelera": os.path.join(out_root, "cartelera"),
        "nota_secretaria": os.path.join(out_root, "nota_secretaria"),
        "expediente": os.path.join(out_root, "expediente"),
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)

    print(f"[INFO] Inicializando PaddleOCR para clasificar y renombrar (lang={lang})...")
    ocr = create_ocr(lang)

    renamed: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("[AVISO] No se encontraron archivos PDF en el directorio.")

    for filename in sorted(pdf_files):
        src_path = os.path.join(dir_path, filename)
        print(f"[INFO] Analizando primera página: {src_path}")
        try:
            page_text = extract_text_first_page(ocr, src_path, dpi=dpi, min_score=min_score)
        except Exception as e:
            print(f"[ERROR] Falló OCR en primera página para {src_path}: {e}")
            skipped.append((src_path, "error_ocr"))
            continue

        norm = _normalize_text(page_text).lower()

        category = None
        number = None

        # Prioridad ajustada: nota_secretaria > cartelera > expediente > predial
        if ("nota" in norm and "secretar" in norm):
            category = "nota_secretaria"
            # Primero intentamos radicado estricto; si no, usar numero de factura
            number = find_radicado_number_strict(page_text)
            if not number:
                number = find_invoice_number(page_text)
            if not number and allow_fallback:
                # como último recurso, intenta patrones menos estrictos
                number = find_radicado_number(page_text)
        elif ("carteler" in norm) or is_cartelera_hint(page_text):
            # Acepta 'cartelera', 'carteleras' y casos con pistas fuertes + Radicación
            category = "cartelera"
            number = find_radicado_number_strict(page_text)
            if not number and allow_fallback:
                number = find_radicado_number(page_text)
        elif "expediente" in norm:
            category = "expediente"
            number = find_expediente_number(page_text)
            if not number and allow_fallback:
                longest = ""
                for m in re.finditer(r"(\d{6,})", _normalize_text(page_text)):
                    if len(m.group(1)) > len(longest):
                        longest = m.group(1)
                number = longest if len(longest) >= 6 else None
        elif "predial" in norm:
            category = "predial"
            number = find_radicado_number(page_text)
            if not number and allow_fallback:
                # intentar radicado estricto y luego dígitos largos
                number = find_radicado_number_strict(page_text) or number
                if not number:
                    longest = ""
                    for m in re.finditer(r"(\d{12,})", _normalize_text(page_text)):
                        if len(m.group(1)) > len(longest):
                            longest = m.group(1)
                    number = longest if len(longest) >= 12 else None
        else:
            skipped.append((src_path, "sin_categoria"))
            continue

        if not number:
            motivo = "sin_numero"
            if category == "expediente":
                motivo = "sin_expediente"
            elif category == "cartelera":
                motivo = "sin_radicado"
            elif category == "nota_secretaria":
                motivo = "sin_radicado_y_factura"
            skipped.append((src_path, motivo))
            print(f"[AVISO] {category}: palabra clave encontrada, pero sin número; se omite.")
            continue

        new_name = f"{category}_{number}.pdf" if category != "nota_secretaria" else f"nota_secretaria_{number}.pdf"
        dst_dir = subdirs[category]
        dst_path = ensure_unique_path(dst_dir, new_name)
        try:
            shutil.move(src_path, dst_path)
            print(f"[OK] Renombrado y movido: {os.path.basename(src_path)} -> {dst_path}")
            renamed.append((src_path, dst_path))
        except Exception as e:
            print(f"[ERROR] No se pudo renombrar/mover {src_path} -> {dst_path}: {e}")
            skipped.append((src_path, "error_renombrar"))

    return renamed, skipped


def process_prefix_rename(
    dir_path: str,
    lang: str = "latin",
    dpi: int = 300,
    min_score: float = 0.0,
    out_dir: str | None = None,
    allow_fallback: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Renombra PDFs en el MISMO directorio según prefijos basados EXCLUSIVAMENTE en la primera página.

    Categorías implementadas (todo en mayúsculas):
     - TE {numero_factura}: cuando se detecte "factura oficial" en la primera página.
     - MP {resolucion}: si el documento contiene simultáneamente "nota" y "secretaria" y además "mandamiento de pago"; se extrae la resolución de mandamiento de pago.
     - CE {numero_factura}: cuando aparezcan las palabras "nota" y "secretaria" en la primera página (se extrae el número de factura).
     - CMP {numero_radicado}: cuando se detecten simultáneamente "citacion", "notificacion" y "mandamiento de pago"; se extrae el número de radicado/radicación.
     - NPMP {numero_radicado}: cuando se detecte "asunto" y "notificación por correo"; se extrae el número de radicado (soporta "RADICACION {numero}", "{numero} RADICACION" y "Radicado ...").
     - AP {numero_resolucion}: cuando se detecte "acuerdo de pago" o "acuerdo de pago de impuesto"; se extrae la resolución.
     - RREX {numero_resolucion}: cuando se detecte "prescripción" junto con "predial" (p. ej., "prescripción de impuesto predial"); se extrae el número de resolución.
     - AC {numero_expediente}: cuando se detecte "auto" junto con "avoca/avocar" y "conocimiento"; se extrae el número de expediente.
     - DF {numero_radicado}: cuando se detecte "prensa y comunicaciones" o "solicitud de publicacion de medios".
     - NC {numero_radicado}: cuando se detecten ambas palabras "publicacion" y "cartelera" (o "cartela").
     - FJ {numero_radicado}: cuando se detecte "publiquese".
  
     El número de radicado se obtiene con búsqueda estricta y, si allow_fallback=True, se intenta con heurística flexible.
  
     Devuelve (renamed, skipped): listas de (src, dst) y (src, motivo).
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directorio no encontrado: {dir_path}")

    print(f"[INFO] Inicializando PaddleOCR para renombrar por prefijos (lang={lang})...")
    ocr = create_ocr(lang)

    renamed: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("[AVISO] No se encontraron archivos PDF en el directorio.")

    def _get_radicado(text: str) -> str | None:
        rad = find_radicado_number_strict(text)
        if not rad and allow_fallback:
            rad = find_radicado_number(text)
        return rad

    for filename in sorted(pdf_files):
        src_path = os.path.join(dir_path, filename)
        print(f"[INFO] Analizando primera página: {src_path}")
        try:
            page_text = extract_text_first_page(ocr, src_path, dpi=dpi, min_score=min_score)
        except Exception as e:
            print(f"[ERROR] Falló OCR en primera página para {src_path}: {e}")
            skipped.append((src_path, "error_ocr"))
            continue

        norm = _normalize_text(page_text).lower()
        target_dir = out_dir if out_dir else dir_path

        # 1) TE {numero_factura}: requiere "factura oficial"
        if re.search(r"\bfactura\s+oficial(es)?\b", norm):
            inv = find_invoice_number_official(page_text)
            if inv:
                new_name = f"TE {inv}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_numero_factura"))
                print("[AVISO] 'factura oficial' detectado pero sin número de factura válido.")
            continue
 
        # 2) CE {numero_factura} o MP {resolucion}: cuando aparezcan 'nota' y 'secretaria' en la primera página
        has_nota = re.search(r"\bnota\b", norm)
        has_secretaria = re.search(r"\bsecretar\w*\b", norm) or re.search(r"\bsecretaria\b", norm)
        if has_nota and has_secretaria:
            # Si también contiene 'mandamiento de pago', priorizar MP
            if re.search(r"mandamiento\s+de\s+pago", norm):
                res_mp = find_resolution_mandamiento_pago(page_text)
                if res_mp:
                    new_name = f"MP {res_mp}.pdf"
                    dst_path = ensure_unique_path(target_dir, new_name)
                    try:
                        shutil.move(src_path, dst_path)
                        print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                        renamed.append((src_path, dst_path))
                    except Exception as e:
                        print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                        skipped.append((src_path, "error_renombrar"))
                else:
                    skipped.append((src_path, "sin_resolucion_mp"))
                    print("[AVISO] 'Nota/Secretaria' + 'Mandamiento de pago' detectado pero sin resolución válida.")
                # En ambos casos no intentamos CE; pasamos al siguiente archivo
                continue
            # Si no es mandamiento de pago, aplicar CE
            inv_ce = find_invoice_number_official(page_text) or find_invoice_number(page_text)
            if inv_ce:
                new_name = f"CE {inv_ce}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_numero_factura"))
                print("[AVISO] 'Nota/Secretaria' detectado pero sin número de factura válido.")
            continue
 
        # 3) CMP {numero_radicado}: 'citacion' + 'notificacion' + 'mandamiento de pago'
        has_citacion = re.search(r"\bcitaci\w*\b", norm)
        has_notificacion = re.search(r"\bnotific\w*\b", norm)
        has_mp = re.search(r"mandamiento\s+de\s+pago", norm)
        if has_citacion and has_notificacion and has_mp:
            rad = _get_radicado(page_text)
            if rad:
                new_name = f"CMP {rad}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_radicado"))
                print("[AVISO] 'Citacion/Notificacion/Mandamiento de pago' detectados pero sin número de radicado.")
            continue
 
        # 4) NPMP {numero_radicado}: 'Asunto' + 'Notificación por correo'
        has_asunto = re.search(r"\basunto\b", norm)
        has_notif_correo = re.search(r"notificaci[oó]n\s+por\s+correo", norm)
        if has_asunto and has_notif_correo:
            rad = _get_radicado(page_text) or find_radicado_near_tokens(page_text)
            if rad:
                new_name = f"NPMP {rad}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_radicado_npmp"))
                print("[AVISO] 'Asunto: Notificación por correo' detectado pero sin número de radicado.")
            continue
 
        # 5) RREX {numero_resolucion}: 'peticion' + 'prescripcion' + 'impuesto predial'
        has_peticion = re.search(r"\bpetici[oó]n\b", norm)
        # Señales más amplias: cualquiera de 'prescrib' o 'prescripc' y 'predial' o 'impuesto predial'
        has_presc_predial = bool(re.search(r"prescripci[oó]n[^\n]{0,120}?impuesto\s+predial", norm)) or bool(re.search(r"impuesto\s+predial[^\n]{0,120}?prescripci[oó]n", norm))
        has_presc = bool(re.search(r"\bprescrib|prescripc", norm))
        has_predial = bool(re.search(r"\bpredial\b", norm)) or bool(re.search(r"impuesto\s+predial", norm))
        if has_presc_predial or (has_presc and has_predial):
            res_rrex = find_resolution_general(page_text)
            if res_rrex:
                new_name = f"RREX {res_rrex}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_resolucion_rrex"))
                print("[AVISO] 'Peticion sobre prescripcion de impuesto predial' detectada pero sin número de resolución.")
            continue
 
        # 6) AP {numero_resolucion}: 'acuerdo de pago' o 'acuerdo de pago de impuesto'
        has_ap = re.search(r"acuerdo\s+de\s+pago(?:\s+de\s+impuesto)?", norm)
        if has_ap:
            res_ap = find_resolution_acuerdo_pago(page_text)
            if res_ap:
                new_name = f"AP {res_ap}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_resolucion_ap"))
                print("[AVISO] 'Acuerdo de pago' detectado pero sin número de resolución.")
            continue
 
        # 5.1) RREX {numero_resolucion}: 'peticion' + 'prescripcion' + 'impuesto predial'
        # Señales más amplias: cualquiera de 'prescrib' o 'prescripc' y 'predial' o 'impuesto predial'
        has_presc_predial = bool(re.search(r"prescripci[oó]n[^\n]{0,120}?impuesto\s+predial", norm)) or bool(re.search(r"impuesto\s+predial[^\n]{0,120}?prescripci[oó]n", norm))
        has_presc = bool(re.search(r"\bprescrib|prescripc", norm))
        has_predial = bool(re.search(r"\bpredial\b", norm)) or bool(re.search(r"impuesto\s+predial", norm))
        if has_presc_predial or (has_presc and has_predial):
            res_rrex = find_resolution_general(page_text)
            if res_rrex:
                new_name = f"RREX {res_rrex}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_resolucion_rrex"))
                print("[AVISO] 'Peticion sobre prescripcion de impuesto predial' detectada pero sin número de resolución.")
            continue
 
        # 6) AC {numero_expediente}: 'auto' + 'avoca/avocar' + 'conocimiento'
        has_auto = re.search(r"\bauto\b", norm)
        has_avoca = re.search(r"\bavoc\w*\b", norm)
        has_conocimiento = re.search(r"\bconocim\w*\b", norm)
        if has_auto and has_avoca and has_conocimiento:
            exp = find_expediente_number(page_text)
            if not exp and allow_fallback:
                # Fallback: dígitos largos (>=6)
                longest = ""
                for m in re.finditer(r"(\d{6,})", _normalize_text(page_text)):
                    if len(m.group(1)) > len(longest):
                        longest = m.group(1)
                exp = longest if len(longest) >= 6 else None
            if exp:
                new_name = f"AC {exp}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_expediente"))
                print("[AVISO] 'Auto que avoca/avocar conocimiento' detectado pero sin número de expediente.")
            continue

        # 7) DF {numero_radicado}: 'prensa y comunicaciones' o 'solicitud de publicacion de medios'
        has_prensa_comms = re.search(r"prensa\s+y\s+comunicaciones", norm)
        has_solic_pub_medios = re.search(r"solicitud\s+de\s+publicacion\s+(?:en|de)\s+medios", norm)
        if has_prensa_comms or has_solic_pub_medios:
            rad = _get_radicado(page_text) or find_radicado_near_tokens(page_text)
            if rad:
                new_name = f"DF {rad}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_radicado_df"))
                print("[AVISO] 'Prensa y comunicaciones/Solicitud de publicacion de medios' detectado pero sin número de radicado.")
            continue

        # 8) NC {numero_radicado}: 'publicacion' + 'cartelera' (o 'cartela')
        has_publicacion = re.search(r"publicaci[oó]n", norm)
        has_cartelera = ("carteler" in norm)
        if has_publicacion and has_cartelera:
            rad = _get_radicado(page_text) or find_radicado_near_tokens(page_text)
            if rad:
                new_name = f"NC {rad}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_radicado_nc"))
                print("[AVISO] 'Publicacion/Cartelera' detectados pero sin número de radicado.")
            continue

        # 9) FJ {numero_radicado}: 'publiquese'
        if re.search(r"\bpubliquese\b", norm):
            rad = _get_radicado(page_text) or find_radicado_near_tokens(page_text)
            if rad:
                new_name = f"FJ {rad}.pdf"
                dst_path = ensure_unique_path(target_dir, new_name)
                try:
                    shutil.move(src_path, dst_path)
                    print(f"[OK] Renombrado: {os.path.basename(src_path)} -> {dst_path}")
                    renamed.append((src_path, dst_path))
                except Exception as e:
                    print(f"[ERROR] No se pudo renombrar {src_path} -> {dst_path}: {e}")
                    skipped.append((src_path, "error_renombrar"))
            else:
                skipped.append((src_path, "sin_radicado_fj"))
                print("[AVISO] 'Publiquese' detectado pero sin número de radicado.")
            continue

        # Si no coincide ninguna regla:
        skipped.append((src_path, "sin_prefijo"))
        continue

    return renamed, skipped


def parse_args():
    parser = argparse.ArgumentParser(description="OCR de PDF con PaddleOCR")
    parser.add_argument("--pdf", required=False, help="Ruta al archivo PDF")
    parser.add_argument("--dir", required=False, help="Directorio con archivos PDF a procesar")
    parser.add_argument("--lang", default="latin", help="Idioma del modelo (ej. 'en', 'ch', 'latin') [por defecto: latin]")
    parser.add_argument("--dpi", type=int, default=300, help="Resolución de rasterizado de las páginas [por defecto: 300]")
    parser.add_argument("--gpu", action="store_true", help="Usar GPU si está disponible")
    parser.add_argument("--keyword", default="sentencia", help="Palabra/frase a buscar en el texto OCR (modo carpeta)")
    parser.add_argument("--out-dir", default=None, help="Carpeta destino para mover coincidencias (por defecto '<dir>/procesados')")
    parser.add_argument("--min-score", type=float, default=0.80, help="Puntaje mínimo de reconocimiento para filtrar líneas (ej. 0.80 para ignorar manuscritos) [por defecto: 0.80]")
    parser.add_argument("--classify-rename", action="store_true", help="Clasificar y renombrar por categorías: predial/cartelera/nota_secretaria/expediente")
    parser.add_argument("--classify-out-root", default=None, help="Carpeta raíz de salida para clasificación (crea subcarpetas)")
    parser.add_argument("--allow-fallback", action="store_true", help="Permitir renombrar con dígitos largos si no se halla el radicado/expediente")
    parser.add_argument("--prefix-rename", action="store_true", help="Renombrar por prefijos en el mismo directorio basados en la primera página (ej. 'TE {numero_factura}', 'CE', 'MP', 'CMP', 'NPMP', 'AP', 'RREX', 'AC', 'DF', 'NC', 'FJ')")
    parser.add_argument("--prefix-out-dir", default=None, help="Carpeta destino para el modo prefijos (por defecto usa el mismo directorio)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        # Defaults: usar carpeta 'docs' y modo prefijos si no se pasan argumentos
        default_dir = os.path.join(os.getcwd(), "docs")
        if not args.pdf and not args.dir:
            if os.path.isdir(default_dir):
                args.dir = default_dir
                if not getattr(args, "classify_rename", False) and not getattr(args, "prefix_rename", False):
                    setattr(args, "prefix_rename", True)
                print(f"[INFO] No se proporcionó --pdf ni --dir. Usando modo prefijos por defecto (lang={args.lang}, dpi={args.dpi}, min-score={args.min_score}) en: {args.dir}")
            else:
                print("[ERROR] Debes proporcionar --pdf o --dir. No se encontró carpeta 'docs' en el directorio actual.")
                sys.exit(1)
        if getattr(args, "prefix_rename", False) and args.dir:
            renamed, skipped = process_prefix_rename(
                dir_path=args.dir,
                lang=args.lang,
                dpi=args.dpi,
                min_score=args.min_score,
                out_dir=args.prefix_out_dir,
                allow_fallback=args.allow_fallback,
            )
            print("\n===== Resumen (Prefijos-Renombrar) =====")
            print(f"Renombrados ({len(renamed)}):")
            for src, dst in renamed:
                print(f" - {os.path.basename(src)} -> {os.path.basename(dst)}")
            print(f"Omitidos ({len(skipped)}):")
            for src, motivo in skipped:
                print(f" - {os.path.basename(src)} ({motivo})")
        elif getattr(args, "classify_rename", False) and args.dir:
            renamed, skipped = process_classify_rename(
                dir_path=args.dir,
                lang=args.lang,
                dpi=args.dpi,
                out_root=args.classify_out_root,
                min_score=args.min_score,
                allow_fallback=args.allow_fallback,
            )
            print("\n===== Resumen (Clasificar y Renombrar) =====")
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