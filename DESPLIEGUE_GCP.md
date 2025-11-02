# 🚀 Guía de Despliegue en Google Cloud Platform (GCP)

Esta guía te muestra cómo desplegar el sistema Churnito en Google Cloud Platform usando diferentes opciones según tus necesidades.

---

## 📋 Índice

1. [Comparación de Opciones](#comparación-de-opciones)
2. [Opción 1: Cloud Run (Recomendado)](#opción-1-cloud-run-recomendado)
3. [Opción 2: Compute Engine (VM)](#opción-2-compute-engine-vm)
4. [Opción 3: Google Kubernetes Engine (Avanzado)](#opción-3-google-kubernetes-engine-avanzado)
5. [Configuración de Secrets](#configuración-de-secrets)
6. [Estimación de Costos](#estimación-de-costos)
7. [Monitoreo y Logs](#monitoreo-y-logs)
8. [Troubleshooting](#troubleshooting)

---

## 📊 Comparación de Opciones

| Característica | Cloud Run | Compute Engine | GKE |
|----------------|-----------|----------------|-----|
| **Facilidad** | ⭐⭐⭐⭐⭐ Muy fácil | ⭐⭐⭐ Media | ⭐⭐ Complejo |
| **Costo mensual** | $5-20 | $30-100 | $100-300 |
| **Escalabilidad** | Automática | Manual | Automática |
| **Tiempo setup** | 10-15 min | 20-30 min | 1-2 horas |
| **Mantenimiento** | Cero | Bajo | Alto |
| **Control** | Limitado | Total | Total |
| **Recomendado para** | Producción pequeña/media | Apps 24/7 | Aplicaciones grandes |

**✅ Recomendación:** Cloud Run para empezar (serverless, pago por uso, zero mantenimiento)

---

## ⭐ Opción 1: Cloud Run (Recomendado)

Cloud Run es un servicio serverless que ejecuta contenedores Docker. Ideal para Churnito porque:
- ✅ Pago solo por uso (escala a cero cuando no hay tráfico)
- ✅ Despliegue automático desde GitHub
- ✅ HTTPS y dominio incluidos
- ✅ Sin gestión de servidores

### 📦 Requisitos Previos

1. **Cuenta de Google Cloud Platform**
   - Ve a: https://console.cloud.google.com
   - Activa la prueba gratuita ($300 de crédito por 90 días)

2. **Instalar Google Cloud SDK** (opcional, pero recomendado)
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Linux
   curl https://sdk.cloud.google.com | bash

   # Windows
   # Descarga desde: https://cloud.google.com/sdk/docs/install
   ```

3. **Habilitar APIs necesarias**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

---

### 🚀 Pasos de Despliegue en Cloud Run

#### Paso 1: Preparar el Proyecto

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/CuchoLeo/Fuga.git
   cd Fuga
   git checkout claude/create-docker-image-011CUWiCdkyttEZPktomfqF1
   ```

2. **Crear proyecto en GCP**
   ```bash
   # Autenticarse
   gcloud auth login

   # Crear proyecto (elige un ID único)
   gcloud projects create churnito-prod --name="Churnito Production"

   # Configurar proyecto como actual
   gcloud config set project churnito-prod

   # Habilitar facturación (requerido para Cloud Run)
   # Ve a: https://console.cloud.google.com/billing
   ```

#### Paso 2: Configurar Variables de Entorno (Secrets)

```bash
# Crear secret para el token de HuggingFace (si lo necesitas)
echo -n "tu_token_aqui" | gcloud secrets create HUGGING_FACE_HUB_TOKEN \
    --data-file=- \
    --replication-policy="automatic"
```

#### Paso 3: Modificar Dockerfile para Cloud Run

Cloud Run requiere que la app escuche en el puerto especificado por `$PORT`. Crea un nuevo archivo `Dockerfile.cloudrun`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos de la aplicación
COPY *.py ./
COPY Churn_Modelling.csv ./
COPY chat_interface.html ./

# Crear usuario no-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Cloud Run asigna PORT dinámicamente
ENV PORT=8080

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD uvicorn churn_chat_api:app --host 0.0.0.0 --port $PORT
```

**Guardar como:** `Dockerfile.cloudrun`

#### Paso 4: Build y Deploy

**Opción A: Desde línea de comandos (Recomendado)**

```bash
# Build y deploy en un solo comando
gcloud run deploy churnito \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars="HUGGING_FACE_HUB_TOKEN=tu_token_aqui"
```

**Opciones explicadas:**
- `--memory 4Gi`: 4GB RAM (necesario para LLM)
- `--cpu 2`: 2 CPUs (acelera inferencia)
- `--timeout 300`: 5 minutos timeout (LLM puede tardar)
- `--min-instances 0`: Escala a cero cuando no hay tráfico
- `--max-instances 10`: Máximo 10 instancias concurrentes
- `--allow-unauthenticated`: Acceso público (quitar para privado)

**Opción B: Desde Google Cloud Console**

1. Ve a: https://console.cloud.google.com/run
2. Click en "CREATE SERVICE"
3. Selecciona "Continuously deploy from a repository (source or function)"
4. Conecta tu repositorio de GitHub
5. Configura:
   - **Branch:** `claude/create-docker-image-011CUWiCdkyttEZPktomfqF1`
   - **Build type:** Dockerfile
   - **Dockerfile path:** `Dockerfile.cloudrun`
6. En "Container, Networking, Security":
   - **Memory:** 4 GiB
   - **CPU:** 2
   - **Request timeout:** 300 seconds
   - **Environment variables:** Agregar `HUGGING_FACE_HUB_TOKEN`
7. Click "CREATE"

#### Paso 5: Entrenar el Modelo

El modelo debe entrenarse antes de que la app pueda usarlo. Dos opciones:

**Opción A: Entrenar localmente y subir**

```bash
# 1. Entrenar localmente
python train_churn_prediction.py

# 2. Crear bucket de GCS
gsutil mb gs://churnito-models

# 3. Subir modelo a GCS
gsutil -m cp -r churn_model/ gs://churnito-models/

# 4. Descargar en Cloud Run startup
# (Modificar churn_chat_api.py para descargar de GCS)
```

**Opción B: Cloud Build Job (Recomendado)**

Crea `cloudbuild.yaml`:

```yaml
steps:
  # Entrenar modelo
  - name: 'python:3.11'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r requirements.txt
        python train_churn_prediction.py

  # Subir a Cloud Storage
  - name: 'gcr.io/cloud-builders/gsutil'
    args: ['cp', '-r', 'churn_model/', 'gs://churnito-models/']
```

Ejecutar:
```bash
gcloud builds submit --config=cloudbuild.yaml
```

#### Paso 6: Verificar Despliegue

```bash
# Obtener URL del servicio
gcloud run services describe churnito \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)'

# Probar endpoint
curl https://churnito-xxxxx-uc.a.run.app/health
```

**Resultado esperado:**
```json
{
  "status": "healthy",
  "churn_model_loaded": true,
  "llm_loaded": true,
  "timestamp": "2025-10-30T10:00:00"
}
```

#### Paso 7: Configurar Dominio Personalizado (Opcional)

```bash
# Mapear dominio
gcloud run domain-mappings create \
    --service churnito \
    --domain churnito.tudominio.com \
    --region us-central1
```

---

### 🔧 Optimizaciones para Cloud Run

#### 1. Usar Cloud Storage para Modelos

Modificar `churn_chat_api.py`:

```python
from google.cloud import storage
import os

def download_model_from_gcs():
    """Descargar modelo de Cloud Storage al iniciar"""
    if not os.path.exists("churn_model"):
        storage_client = storage.Client()
        bucket = storage_client.bucket("churnito-models")

        # Descargar todos los archivos del modelo
        blobs = bucket.list_blobs(prefix="churn_model/")
        for blob in blobs:
            blob.download_to_filename(blob.name)

        print("✅ Modelo descargado de GCS")

# Llamar al inicio
download_model_from_gcs()
```

#### 2. Caché de Modelos con Persistent Volumes

```bash
# Cloud Run permite volúmenes montados
gcloud run deploy churnito \
    --add-volume name=model-cache,type=cloud-storage,bucket=churnito-models \
    --add-volume-mount volume=model-cache,mount-path=/app/models
```

#### 3. Reducir Cold Start

```bash
# Mantener 1 instancia siempre activa
gcloud run services update churnito \
    --min-instances 1 \
    --region us-central1
```

**Nota:** Esto aumenta costos (~$30/mes) pero elimina cold starts.

---

## 💻 Opción 2: Compute Engine (VM)

Compute Engine te da una máquina virtual con control total. Ideal si:
- Necesitas acceso SSH completo
- Quieres instalar software adicional
- Prefieres gestión tradicional de servidores

### 🚀 Pasos de Despliegue en Compute Engine

#### Paso 1: Crear VM

```bash
# Crear VM con Ubuntu 22.04
gcloud compute instances create churnito-vm \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-balanced \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server,https-server
```

**Especificaciones VM:**
- **Tipo:** e2-standard-4 (4 vCPUs, 16GB RAM)
- **Disco:** 50GB SSD
- **SO:** Ubuntu 22.04 LTS
- **Costo:** ~$100/mes (24/7)

#### Paso 2: Configurar Firewall

```bash
# Permitir tráfico HTTP/HTTPS
gcloud compute firewall-rules create allow-churnito \
    --allow tcp:8000 \
    --target-tags http-server
```

#### Paso 3: Conectarse e Instalar Docker

```bash
# Conectarse a la VM
gcloud compute ssh churnito-vm --zone=us-central1-a

# En la VM, instalar Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose git

# Agregar usuario al grupo docker
sudo usermod -aG docker $USER
newgrp docker
```

#### Paso 4: Clonar y Desplegar

```bash
# Clonar repositorio
git clone https://github.com/CuchoLeo/Fuga.git
cd Fuga
git checkout claude/create-docker-image-011CUWiCdkyttEZPktomfqF1

# Descargar dataset (desde Kaggle)
# Opción 1: Usar Kaggle API
pip install kaggle
mkdir ~/.kaggle
# Subir tu kaggle.json a ~/.kaggle/
kaggle datasets download -d shrutimechlearn/churn-modelling
unzip churn-modelling.zip

# Opción 2: Transferir desde local
# En tu máquina local:
# gcloud compute scp Churn_Modelling.csv churnito-vm:~/Fuga/

# Configurar variables de entorno
cp .env.example .env
nano .env  # Agregar token de HuggingFace

# Build y deploy con Docker
docker-compose build
docker-compose run --rm churn-api python train_churn_prediction.py
docker-compose up -d

# Verificar
curl http://localhost:8000/health
```

#### Paso 5: Configurar Inicio Automático

```bash
# Crear servicio systemd
sudo nano /etc/systemd/system/churnito.service
```

Contenido del servicio:

```ini
[Unit]
Description=Churnito Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/YOUR_USER/Fuga
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
User=YOUR_USER

[Install]
WantedBy=multi-user.target
```

Habilitar servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl enable churnito
sudo systemctl start churnito
```

#### Paso 6: Configurar IP Estática (Opcional)

```bash
# Reservar IP estática
gcloud compute addresses create churnito-ip --region=us-central1

# Asignar a la VM
gcloud compute instances delete-access-config churnito-vm \
    --access-config-name="external-nat" \
    --zone=us-central1-a

gcloud compute instances add-access-config churnito-vm \
    --access-config-name="external-nat" \
    --address=$(gcloud compute addresses describe churnito-ip \
        --region=us-central1 --format="value(address)") \
    --zone=us-central1-a
```

#### Paso 7: Configurar Nginx como Reverse Proxy (Opcional)

```bash
# Instalar Nginx
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Configurar Nginx
sudo nano /etc/nginx/sites-available/churnito
```

Contenido de la configuración:

```nginx
server {
    listen 80;
    server_name churnito.tudominio.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts largos para LLM
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

Activar configuración:

```bash
sudo ln -s /etc/nginx/sites-available/churnito /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Configurar HTTPS con Let's Encrypt
sudo certbot --nginx -d churnito.tudominio.com
```

---

## 🎯 Opción 3: Google Kubernetes Engine (Avanzado)

**Solo recomendado si:**
- Necesitas alta disponibilidad (99.95% SLA)
- Esperas >10,000 requests/día
- Tienes experiencia con Kubernetes

### 🚀 Pasos Básicos

```bash
# 1. Crear cluster
gcloud container clusters create churnito-cluster \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --zone=us-central1-a

# 2. Obtener credenciales
gcloud container clusters get-credentials churnito-cluster

# 3. Deploy usando kubectl
kubectl create deployment churnito --image=gcr.io/churnito-prod/churnito:latest
kubectl expose deployment churnito --type=LoadBalancer --port=80 --target-port=8000

# 4. Obtener IP externa
kubectl get services churnito
```

**Costo estimado:** $250-500/mes (cluster de 3 nodos 24/7)

**Nota:** No detallamos más GKE porque es complejo y probablemente excesivo para este proyecto.

---

## 🔐 Configuración de Secrets

### Opción 1: Secret Manager (Recomendado)

```bash
# Crear secret
gcloud secrets create huggingface-token \
    --replication-policy="automatic"

# Agregar versión
echo -n "tu_token_aqui" | gcloud secrets versions add huggingface-token \
    --data-file=-

# Dar acceso a Cloud Run
gcloud secrets add-iam-policy-binding huggingface-token \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Usar en Cloud Run
gcloud run services update churnito \
    --update-secrets=HUGGING_FACE_HUB_TOKEN=huggingface-token:latest
```

### Opción 2: Variables de Entorno

```bash
# Cloud Run
gcloud run services update churnito \
    --set-env-vars="HUGGING_FACE_HUB_TOKEN=tu_token"

# Compute Engine
# Agregar a .env y usar docker-compose
```

---

## 💰 Estimación de Costos

### Cloud Run (Pago por Uso)

**Escenario 1: Uso Ligero (100 requests/día)**
- Tiempo de ejecución: ~50 horas/mes
- CPU: 2 vCPU × 50h × $0.024 = $2.40
- Memoria: 4GB × 50h × $0.0025 = $0.50
- Requests: 3,000 × $0.0000004 = $0.001
- **Total: ~$3/mes** ✅

**Escenario 2: Uso Moderado (1,000 requests/día)**
- Tiempo de ejecución: ~300 horas/mes
- CPU: 2 vCPU × 300h × $0.024 = $14.40
- Memoria: 4GB × 300h × $0.0025 = $3.00
- Requests: 30,000 × $0.0000004 = $0.012
- **Total: ~$17/mes** ✅

**Escenario 3: Uso Alto (10,000 requests/día) + 1 instancia mínima**
- Instancia siempre activa: ~$30/mes
- Tiempo adicional: ~200 horas/mes
- CPU: 2 vCPU × 200h × $0.024 = $9.60
- Memoria: 4GB × 200h × $0.0025 = $2.00
- **Total: ~$42/mes** ✅

### Compute Engine (24/7)

**e2-standard-4 (4 vCPU, 16GB RAM)**
- VM: $120/mes
- Disco 50GB: $8/mes
- IP estática: $3/mes
- Egress (50GB/mes): $5/mes
- **Total: ~$136/mes** ⚠️

**e2-medium (2 vCPU, 4GB RAM)** - Mínimo viable
- VM: $25/mes
- Disco 50GB: $8/mes
- IP estática: $3/mes
- **Total: ~$36/mes** ✅
- **Nota:** Puede ser lento para LLM

### Google Kubernetes Engine

**Cluster básico (3 nodos e2-standard-4)**
- Nodos: 3 × $120 = $360/mes
- Control plane: Gratis (hasta 1 cluster zonal)
- Load Balancer: $18/mes
- **Total: ~$378/mes** ❌ (caro para este proyecto)

---

## 📊 Monitoreo y Logs

### Cloud Run

```bash
# Ver logs en tiempo real
gcloud run services logs tail churnito --region=us-central1

# Ver métricas
gcloud monitoring dashboards create --config-from-file=dashboard.yaml
```

Dashboard en Cloud Console:
- https://console.cloud.google.com/run
- Click en el servicio "churnito"
- Tab "METRICS" → CPU, memoria, requests, latencia

### Compute Engine

```bash
# SSH a la VM
gcloud compute ssh churnito-vm

# Ver logs de Docker
docker-compose logs -f

# Ver uso de recursos
docker stats
```

### Alertas

```bash
# Crear alerta de alta latencia
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="Churnito High Latency" \
    --condition-display-name="Response time > 10s" \
    --condition-threshold-value=10 \
    --condition-threshold-duration=60s
```

---

## 🐛 Troubleshooting

### Error: "Container failed to start"

**Cloud Run**
```bash
# Ver logs detallados
gcloud run services logs read churnito --region=us-central1 --limit=50

# Causas comunes:
# 1. Puerto incorrecto → La app debe escuchar $PORT
# 2. Memoria insuficiente → Aumentar a 4-8GB
# 3. Timeout en startup → Aumentar timeout a 300s
```

**Solución:**
```bash
gcloud run services update churnito \
    --memory 8Gi \
    --timeout 300 \
    --region=us-central1
```

### Error: "Model not found"

**Causa:** El modelo no se entrenó o no está en el contenedor.

**Solución para Cloud Run:**
1. Entrenar modelo localmente
2. Subir a Cloud Storage
3. Descargar en startup

```python
# En churn_chat_api.py
from google.cloud import storage

def download_model():
    client = storage.Client()
    bucket = client.bucket("churnito-models")

    # Descargar churn_model/
    blobs = bucket.list_blobs(prefix="churn_model/")
    for blob in blobs:
        blob.download_to_filename(blob.name)
```

### Error: "Out of memory"

**Cloud Run**
```bash
# Aumentar memoria
gcloud run services update churnito --memory 8Gi
```

**Compute Engine**
```bash
# Agregar swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Latencia alta (>30s)

**Optimizaciones:**

1. **Reducir max_new_tokens del LLM**
   ```python
   # En churn_chat_api.py
   max_new_tokens=100  # Reducir de 150 a 100
   ```

2. **Usar GPU en Compute Engine**
   ```bash
   gcloud compute instances create churnito-gpu \
       --accelerator type=nvidia-tesla-t4,count=1 \
       --machine-type n1-standard-4
   ```

3. **Caché de respuestas comunes**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=100)
   def cached_llm_response(query: str):
       # ...
   ```

---

## ✅ Checklist de Despliegue

### Pre-despliegue
- [ ] Cuenta GCP creada y verificada
- [ ] Facturación habilitada
- [ ] APIs habilitadas (Cloud Run / Compute Engine)
- [ ] Dataset descargado (Churn_Modelling.csv)
- [ ] Token de HuggingFace (opcional)

### Durante el despliegue
- [ ] Repositorio clonado
- [ ] Rama correcta checked out
- [ ] Modelo entrenado (local o en la nube)
- [ ] Dockerfile adaptado para GCP
- [ ] Variables de entorno configuradas
- [ ] Servicio desplegado

### Post-despliegue
- [ ] Endpoint `/health` responde correctamente
- [ ] Predicciones funcionan (`/predict`)
- [ ] Chat responde (`/chat`)
- [ ] Logs revisados (sin errores críticos)
- [ ] Alertas configuradas
- [ ] Dominio personalizado configurado (opcional)
- [ ] HTTPS habilitado

---

## 🎓 Recursos Adicionales

- **Cloud Run Docs:** https://cloud.google.com/run/docs
- **Compute Engine Docs:** https://cloud.google.com/compute/docs
- **Secret Manager:** https://cloud.google.com/secret-manager/docs
- **Cloud Storage:** https://cloud.google.com/storage/docs
- **Pricing Calculator:** https://cloud.google.com/products/calculator

---

## 📞 Soporte

Si tienes problemas:

1. **Revisa logs:**
   ```bash
   gcloud run services logs read churnito --limit=100
   ```

2. **Verifica configuración:**
   ```bash
   gcloud run services describe churnito --region=us-central1
   ```

3. **Consulta Stack Overflow:**
   - Tag: `google-cloud-run`
   - Tag: `google-compute-engine`

---

**¡Sistema listo para producción en GCP! 🚀☁️**

**Última actualización:** 30 de octubre de 2024
