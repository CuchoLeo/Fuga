# 💰 Guía de Despliegue Low-Cost / Gratis en la Nube

Esta guía te muestra cómo desplegar Churnito con **costo cero o muy bajo** (<$5/mes) usando capas gratuitas y optimizaciones.

---

## 📋 Índice

1. [Comparación de Opciones Gratuitas](#comparación-de-opciones-gratuitas)
2. [⭐ Opción 1: GCP Free Tier (Recomendado)](#opción-1-gcp-free-tier-recomendado)
3. [🚀 Opción 2: Oracle Cloud Always Free](#opción-2-oracle-cloud-always-free)
4. [🎨 Opción 3: Render Free Tier](#opción-3-render-free-tier)
5. [🚂 Opción 4: Railway (Starter Plan)](#opción-4-railway-starter-plan)
6. [⚡ Optimizaciones para Reducir Recursos](#optimizaciones-para-reducir-recursos)
7. [💡 Trucos para Maximizar Free Tier](#trucos-para-maximizar-free-tier)

---

## 📊 Comparación de Opciones Gratuitas

| Plataforma | Costo | RAM | CPU | Almacenamiento | Duración | Limitaciones |
|------------|-------|-----|-----|----------------|----------|--------------|
| **GCP Cloud Run** ⭐ | $0-3/mes | 4GB | 2 vCPU | Ilimitado | Permanente | 2M requests/mes, 360K vCPU-seg/mes |
| **Oracle Cloud** 🏆 | $0 | 24GB | 4 cores | 200GB | **Permanente** | ARM architecture (compatible) |
| **Render** | $0 | 512MB | Compartido | 1GB | Permanente | Se duerme tras 15min inactividad |
| **Railway** | $5/mes | 8GB | Compartido | 100GB | Permanente | $5 crédito gratis/mes |
| **Fly.io** | $0-5/mes | 256MB | Compartido | 3GB | Permanente | 1 VM gratis, resto paga |

**🏆 Ganador absoluto:** Oracle Cloud Always Free (24GB RAM, 4 cores, ARM64, gratis PARA SIEMPRE)

**⭐ Más fácil:** GCP Cloud Run (serverless, escala a cero, casi gratis para tráfico bajo)

---

## ⭐ Opción 1: GCP Free Tier (Recomendado)

### 💰 Costo Real Estimado

**GCP Cloud Run Free Tier incluye (MENSUAL, PERMANENTE):**
- ✅ 2,000,000 de requests
- ✅ 360,000 vCPU-segundos
- ✅ 180,000 GiB-segundos de memoria
- ✅ 1GB egress a Norteamérica

**¿Cuánto puedes usar gratis?**

```python
# Cálculo conservador:
# - Request promedio: 30 segundos (LLM puede tardar)
# - Memoria: 4GB
# - CPU: 2 vCPU

# Límite por vCPU-segundos:
requests_gratis_cpu = 360000 / (30 seg × 2 vCPU) = 6,000 requests/mes

# Límite por memoria:
requests_gratis_mem = 180000 / (30 seg × 4 GB) = 1,500 requests/mes

# LÍMITE REAL: 1,500 requests/mes GRATIS
# = 50 requests/día
# = ~2 requests/hora

# COSTO si excedes:
# 1,000 requests adicionales × 30s × 2 vCPU × $0.00002400 = $1.44
# 1,000 requests adicionales × 30s × 4 GB × $0.00000250 = $0.30
# Total por 1,000 requests extra: ~$1.74
```

**✅ Conclusión:**
- **Uso bajo (50 req/día):** $0/mes (completamente gratis)
- **Uso moderado (200 req/día):** ~$8/mes
- **Uso alto (500 req/día):** ~$25/mes

---

### 🚀 Despliegue en GCP Cloud Run (Maximizando Free Tier)

#### Paso 1: Crear Cuenta y Proyecto

```bash
# 1. Crear cuenta en https://console.cloud.google.com
# Obtienes $300 de crédito gratis por 90 días + Free Tier permanente

# 2. Instalar gcloud CLI
# macOS:
brew install google-cloud-sdk

# Linux:
curl https://sdk.cloud.google.com | bash

# 3. Autenticarse
gcloud auth login

# 4. Crear proyecto
gcloud projects create churnito-free --name="Churnito Free"
gcloud config set project churnito-free

# 5. Habilitar APIs (GRATIS)
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

#### Paso 2: Optimizar Configuración para Free Tier

Crear **Dockerfile.lowcost**:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar solo dependencias esenciales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar Python packages (sin build tools pesados)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar app
COPY churn_chat_api.py ./
COPY train_churn_prediction.py ./
COPY Churn_Modelling.csv ./
COPY chat_interface.html ./

# Usuario no-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080

EXPOSE 8080

# Comando optimizado
CMD uvicorn churn_chat_api:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 60
```

#### Paso 3: Deploy con Configuración Minimal

```bash
# Deploy con mínimos recursos (se ajusta a Free Tier)
gcloud run deploy churnito-free \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 60 \
    --min-instances 0 \
    --max-instances 1 \
    --concurrency 1 \
    --cpu-throttling \
    --execution-environment gen2
```

**Explicación de parámetros para minimizar costo:**
- `--memory 2Gi` - Mínimo viable para LLM (en vez de 4Gi)
- `--cpu 1` - 1 vCPU en vez de 2 (más lento pero más barato)
- `--timeout 60` - 1 minuto timeout (reduce costo por request)
- `--min-instances 0` - Escala a cero cuando no hay tráfico
- `--max-instances 1` - Solo 1 instancia (evita costos múltiples)
- `--concurrency 1` - 1 request a la vez (evita saturación)
- `--cpu-throttling` - CPU throttling cuando no está procesando
- `--execution-environment gen2` - Más eficiente

#### Paso 4: Monitorear Uso (Free Tier)

```bash
# Ver cuánto has usado del Free Tier
gcloud run services describe churnito-free \
    --region us-central1 \
    --format="table(status.traffic)"

# Ver métricas en Cloud Console
echo "Abre: https://console.cloud.google.com/run/detail/us-central1/churnito-free/metrics"
```

#### Paso 5: Alertas de Costo

```bash
# Crear alerta cuando superes $1/mes
gcloud alpha billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Churnito Budget Alert" \
    --budget-amount=1.00 \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=100
```

---

## 🏆 Opción 2: Oracle Cloud Always Free (Mejor Opción Gratis)

**Oracle Cloud ofrece GRATIS PARA SIEMPRE:**
- ✅ 4 OCPU ARM Ampere (equiv. a 4 cores)
- ✅ 24 GB RAM
- ✅ 200 GB storage (block + object)
- ✅ 10 TB egress/mes
- ✅ **NO expira, NO requiere tarjeta de crédito después del trial**

**Ventajas:**
- 24GB RAM = Suficiente para LLM + modelo de churn + caché
- ARM64 compatible con PyTorch y transformers
- Recursos dedicados (no compartidos)
- Sin cold start

**Desventajas:**
- Setup más manual que Cloud Run
- Arquitectura ARM (requiere builds específicos)

---

### 🚀 Despliegue en Oracle Cloud Always Free

#### Paso 1: Crear Cuenta

1. Ve a: https://www.oracle.com/cloud/free/
2. Regístrate (requiere tarjeta para verificación, NO se cobra)
3. Selecciona región (ej: **US East (Ashburn)**)

#### Paso 2: Crear VM ARM

```bash
# En Oracle Cloud Console:
# 1. Ir a "Compute" → "Instances"
# 2. Click "Create Instance"

# Configuración:
Name: churnito-vm
Image: Ubuntu 22.04 (ARM)
Shape: VM.Standard.A1.Flex
  - OCPUs: 4
  - Memory: 24 GB
  - ✅ Esto es 100% GRATIS PARA SIEMPRE

# Networking:
- Create new VCN
- Assign public IP
- SSH keys: Generar o subir tu clave

# Click "Create"
```

#### Paso 3: Configurar Firewall

```bash
# En la VM instance page:
# 1. Click en "Subnet"
# 2. Click en "Default Security List"
# 3. Click "Add Ingress Rules"

Source CIDR: 0.0.0.0/0
Destination Port: 8000
Description: Churnito API

# Guardar
```

#### Paso 4: Conectar e Instalar

```bash
# Conectar por SSH (desde tu máquina local)
ssh -i ~/.ssh/oracle_key ubuntu@<PUBLIC_IP>

# En la VM, abrir firewall Ubuntu
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save

# Instalar Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose git

# Agregar usuario a docker
sudo usermod -aG docker ubuntu
newgrp docker

# Clonar repo
git clone https://github.com/CuchoLeo/Fuga.git
cd Fuga
git checkout claude/create-docker-image-011CUWiCdkyttEZPktomfqF1

# Configurar .env
cp .env.example .env
nano .env  # Agregar token HF si lo tienes

# Build y deploy
docker-compose build
docker-compose run --rm churn-api python train_churn_prediction.py
docker-compose up -d

# Verificar
curl http://localhost:8000/health
```

#### Paso 5: Acceso Desde Internet

```bash
# Obtener IP pública de la VM
# En Oracle Console: Compute → Instances → churnito-vm → Public IP

# Probar desde tu máquina local
curl http://<PUBLIC_IP>:8000/health
```

#### Paso 6: Configurar Inicio Automático

```bash
# Crear servicio systemd
sudo nano /etc/systemd/system/churnito.service
```

Contenido:
```ini
[Unit]
Description=Churnito Docker Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/Fuga
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
```

Activar:
```bash
sudo systemctl daemon-reload
sudo systemctl enable churnito
sudo systemctl start churnito
```

**Resultado:**
- ✅ Sistema corriendo 24/7
- ✅ 24GB RAM = Sin problemas de memoria
- ✅ Costo: $0/mes PARA SIEMPRE

---

## 🎨 Opción 3: Render Free Tier

**Render Free Tier incluye:**
- ✅ 750 horas gratis/mes (suficiente para 1 servicio 24/7)
- ✅ 512 MB RAM
- ✅ CPU compartido
- ❌ Se duerme tras 15 min de inactividad (cold start ~30s)

**⚠️ Problema:** 512MB RAM es insuficiente para Qwen2.5 (necesita 3-4GB)

**Solución:** Desplegar sin LLM, solo con modelo de churn + respuestas estructuradas

---

### 🚀 Despliegue en Render (Sin LLM)

#### Paso 1: Modificar Código para Desactivar LLM

Crear **churn_chat_api_minimal.py**:

```python
# Copiar churn_chat_api.py pero comentar esta línea:
# self.llm_model = AutoModelForCausalLM.from_pretrained(...)
# self.llm_tokenizer = AutoTokenizer.from_pretrained(...)

# En generate_llm_response(), siempre usar:
return self._generate_recommendations(context)
```

#### Paso 2: Crear render.yaml

```yaml
services:
  - type: web
    name: churnito-free
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    dockerContext: .
    envVars:
      - key: PORT
        value: 8000
      - key: DISABLE_LLM
        value: true
    healthCheckPath: /health
```

#### Paso 3: Deploy desde GitHub

1. Ve a: https://dashboard.render.com/
2. Click "New" → "Web Service"
3. Conecta tu repositorio de GitHub
4. Selecciona rama: `claude/create-docker-image-011CUWiCdkyttEZPktomfqF1`
5. Render detecta Dockerfile automáticamente
6. Click "Create Web Service"

**Resultado:**
- ✅ Predicciones funcionan
- ✅ Chat con respuestas estructuradas (sin LLM)
- ⚠️ Se duerme tras 15 min inactividad
- ✅ Costo: $0/mes

---

## 🚂 Opción 4: Railway (Starter Plan)

**Railway ofrece:**
- ✅ $5 de crédito GRATIS cada mes
- ✅ 8 GB RAM
- ✅ CPU compartido
- ✅ 100 GB egress
- ✅ Sin sleep/cold start

**Costo real:**
```
$5 crédito/mes - ~$4 uso real = $1 neto/mes o GRATIS
```

---

### 🚀 Despliegue en Railway

#### Paso 1: Crear Cuenta

1. Ve a: https://railway.app
2. Registra con GitHub
3. Obtienes $5/mes gratis automáticamente

#### Paso 2: Deploy desde GitHub

```bash
# Opción A: Desde Railway Dashboard
# 1. Click "New Project"
# 2. Select "Deploy from GitHub repo"
# 3. Selecciona tu repo
# 4. Railway auto-detecta Dockerfile

# Opción B: Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

#### Paso 3: Configurar Variables

```bash
# En Railway Dashboard:
# Variables → Add Variable

HUGGING_FACE_HUB_TOKEN=tu_token_aqui
PORT=8000
```

#### Paso 4: Configurar Recursos

```bash
# En Settings:
Memory: 4GB
vCPUs: 2
Replicas: 1
```

**Resultado:**
- ✅ Sistema completo con LLM
- ✅ Sin cold start
- ✅ Costo: ~$0-1/mes (dentro del crédito de $5)

---

## ⚡ Optimizaciones para Reducir Recursos

### 1. Reducir Uso de Memoria del LLM

Modificar **churn_chat_api.py**:

```python
# Cargar modelo con cuantización (reduce 50% memoria)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Cuantización a 8 bits
    llm_int8_threshold=6.0
)

self.llm_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

**Resultado:** 3GB → 1.5GB de RAM

### 2. Usar Modelo LLM Más Pequeño

```python
# Cambiar de Qwen2.5-1.5B a Qwen2.5-0.5B
model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # 500M parámetros

# O usar TinyLlama (1.1B parámetros, muy eficiente)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Resultado:** 3GB → 1GB de RAM

### 3. Caché Agresivo

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_llm(query_hash: str):
    # Caché respuestas del LLM
    pass

def generate_llm_response(self, query: str, context):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return cached_llm(query_hash)
```

### 4. Desactivar LLM por Completo

```python
# Usar solo respuestas estructuradas
DISABLE_LLM = os.getenv("DISABLE_LLM", "false") == "true"

if DISABLE_LLM:
    # No cargar LLM
    self.llm_model = None
    self.llm_tokenizer = None
```

---

## 💡 Trucos para Maximizar Free Tier

### 1. Usar Múltiples Cuentas (Legal)

```bash
# Crear proyectos en diferentes cuentas
# GCP: 1 proyecto por cuenta = múltiples Free Tiers
# Oracle: 1 cuenta = 4 VMs Always Free

# Ejemplo:
# Cuenta 1: Producción
# Cuenta 2: Staging
# Cuenta 3: Desarrollo
```

### 2. Combinar Servicios

```bash
# Backend: Oracle Cloud Always Free (24GB RAM)
# Frontend: Vercel/Netlify (gratis para static sites)
# Database: Supabase Free Tier (500MB)
# Storage: Cloudflare R2 Free Tier (10GB)
```

### 3. Optimizar Tráfico

```python
# Implementar rate limiting
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # Máximo 10 requests/min
async def chat(...):
    pass
```

### 4. Comprimir Respuestas

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZIPMiddleware, minimum_size=1000)
```

### 5. Programar Horarios de Actividad

```bash
# Cloud Run: Programar min-instances=0 durante la noche
gcloud scheduler jobs create http shutdown-night \
    --schedule="0 2 * * *" \
    --uri="https://run.googleapis.com/v2/projects/.../services/churnito" \
    --http-method=PATCH \
    --message-body='{"spec": {"template": {"scaling": {"minInstanceCount": 0}}}}'

# Despertar en la mañana
gcloud scheduler jobs create http wakeup-morning \
    --schedule="0 8 * * *" \
    --uri="https://churnito-xxx.run.app/health"
```

---

## 📊 Resumen de Costos Reales

| Plataforma | Setup | Costo/mes | Memoria | LLM | Cold Start | Recomendado para |
|------------|-------|-----------|---------|-----|------------|------------------|
| **Oracle Cloud Always Free** 🏆 | 30 min | **$0** | 24GB | ✅ | ❌ | **Mejor opción gratis** |
| **GCP Cloud Run** ⭐ | 15 min | **$0-3** | 2-4GB | ✅ | ✅ ~5s | Tráfico bajo/medio |
| **Railway** | 10 min | **$0-1** | 8GB | ✅ | ❌ | Desarrollo/staging |
| **Render** | 5 min | **$0** | 512MB | ❌ | ✅ ~30s | Solo predicciones |
| **Fly.io** | 15 min | **$0-5** | 256MB | ❌ | ✅ ~10s | Microservicios |

---

## ✅ Recomendación Final

### Para uso personal/demo:
**🏆 Oracle Cloud Always Free**
- 24GB RAM gratis PARA SIEMPRE
- Sin limitaciones de requests
- Sin cold start
- Setup: 30 minutos

### Para producción con bajo tráfico:
**⭐ GCP Cloud Run**
- ~50 requests/día gratis
- Escala automáticamente
- Más fácil de mantener
- Setup: 15 minutos

### Para desarrollo:
**🚂 Railway**
- $5 crédito mensual
- Sin configuración compleja
- Perfecto para iterar rápido
- Setup: 10 minutos

---

## 🐛 Troubleshooting Free Tier

### Error: "Quota exceeded"

**GCP:**
```bash
# Ver cuotas usadas
gcloud compute project-info describe \
    --project=churnito-free

# Solicitar aumento de cuota (gratis, solo pedir)
# https://console.cloud.google.com/iam-admin/quotas
```

**Oracle:**
```bash
# Oracle limita a 4 OCPU ARM por cuenta
# Si necesitas más: crear segunda cuenta (legal)
```

### Error: "Out of memory"

```bash
# Reducir memoria del LLM
# Ver sección "Optimizaciones" arriba

# O desactivar LLM completamente
export DISABLE_LLM=true
```

### Error: "Instance always sleeping" (Render)

```bash
# Render Free duerme después de 15 min
# Soluciones:
# 1. Ping cada 10 min con cron job
# 2. Usar UptimeRobot (gratis)
# 3. Upgraar a plan paid ($7/mes)
```

---

## 📞 Recursos Adicionales

- **GCP Free Tier:** https://cloud.google.com/free
- **Oracle Cloud Free:** https://www.oracle.com/cloud/free/
- **Railway Pricing:** https://railway.app/pricing
- **Render Free Tier:** https://render.com/docs/free
- **Free Tier Tracker:** https://free-for.dev/

---

**¡Sistema en la nube con costo cero! 🎉💰**

**Última actualización:** 30 de octubre de 2024
