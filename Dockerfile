# ============================================
# STAGE 1: Base image con Python 3.11
# ============================================
FROM python:3.11-slim as base

# Metadatos de la imagen
LABEL maintainer="Churn Prediction System"
LABEL description="Sistema de predicción de churn con IA y chat en lenguaje natural"

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================
# STAGE 2: Instalar dependencias del sistema
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# STAGE 3: Configurar directorio de trabajo
# ============================================
WORKDIR /app

# ============================================
# STAGE 4: Instalar dependencias de Python
# ============================================
# Copiar solo requirements.txt primero (para aprovechar caché de Docker)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# STAGE 5: Copiar código de la aplicación
# ============================================
# Copiar archivos Python
COPY *.py ./

# Copiar dataset (si existe)
COPY Churn_Modelling.csv ./

# Copiar interfaz HTML (si existe)
COPY *.html ./ 2>/dev/null || true

# ============================================
# STAGE 6: Crear directorios para modelos
# ============================================
RUN mkdir -p /app/churn_model /app/trained_model

# ============================================
# STAGE 7: Configurar usuario no-root (seguridad)
# ============================================
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# ============================================
# STAGE 8: Exponer puerto
# ============================================
EXPOSE 8000

# ============================================
# STAGE 9: Health check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ============================================
# STAGE 10: Comando de inicio
# ============================================
CMD ["python", "churn_chat_api.py"]
