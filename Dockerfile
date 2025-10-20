# Base con PaddlePaddle GPU + CUDA 12.0 + cuDNN 8 ya instalado
# Esto garantiza compatibilidad sin tener que instalar CUDA/cuDNN manualmente.
FROM paddlepaddle/paddle:2.6.0-gpu-cuda12.0-cudnn8

# Dependencias del sistema para OpenCV (libGL) y utilidades
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero para aprovechar caché de build
COPY requirements.txt /app/requirements.txt

# Crear un requirements compatible con GPU (evita instalar 'paddlepaddle' CPU)
RUN python - <<'PY' \
import re
input_path='requirements.txt'
output_path='requirements.gpu.txt'
lines=[l.strip() for l in open(input_path, 'r')]
keep=[l for l in lines if l and not re.match(r'(?i)^paddlepaddle\s*$', l)]
open(output_path, 'w').write('\n'.join(keep)+'\n')
print('requirements.gpu.txt:', keep)
PY

# Instalar dependencias de Python (Paddle ya viene en la imagen base)
RUN pip install --no-cache-dir -r requirements.gpu.txt

# Copiar el resto del proyecto
COPY . /app

# Variables de entorno recomendadas para GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PADDLEOCR_LANG=es

# Comando por defecto (docker compose lo sobreescribe)
CMD ["python", "ocr_pdf.py", "--help"]