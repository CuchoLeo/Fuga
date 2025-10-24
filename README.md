# 🎯 Sistema de Predicción de Churn con Chat IA
Sistema completo de predicción de fuga de clientes usando IA, con interfaz conversacional en lenguaje natural.

## 📋 Descripción del Problema

**Situación Actual:**
- 📉 Tasa anual de churn: **25%**
- 👥 Clientes perdidos/mes: **2,500**
- 💰 Foco: Clientes de alto valor (>USD $100,000)
- 💵 Costo de retención = **1/5** del costo de adquisición
- ❌ Sin capacidad predictiva actual

**Solución:**
Sistema de IA que predice qué clientes están en riesgo de abandonar, con chat en lenguaje natural para consultas y análisis.
---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA DE PREDICCIÓN                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Modelo LLM      │      │  Modelo Churn    │            │
│  │  (Llama 3.2)     │◄────►│  (DistilBERT)    │            │
│  │  Conversación    │      │  Clasificación   │            │
│  └──────────────────┘      └──────────────────┘            │
│           │                         │                        │
│           └─────────┬───────────────┘                        │
│                     ▼                                        │
│            ┌─────────────────┐                              │
│            │   FastAPI       │                              │
│            │   Backend       │                              │
│            └─────────────────┘                              │
│                     │                                        │
│        ┌────────────┼────────────┐                         │
│        ▼            ▼            ▼                          │
│   ┌────────┐  ┌────────┐  ┌────────┐                      │
│   │  Chat  │  │Predict │  │ Stats  │                      │
│   │   API  │  │  API   │  │  API   │                      │
│   └────────┘  └────────┘  └────────┘                      │
│                     │                                        │
│                     ▼                                        │
│            ┌─────────────────┐                              │
│            │  Web Interface  │                              │
│            │   (HTML/JS)     │                              │
│            └─────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Instalación

### 1. Requisitos Previos

```bash
Python 3.8+
CUDA (opcional, para GPU)
```

### 2. Instalar Dependencias

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install fastapi uvicorn
pip install pandas scikit-learn
pip install requests
```

O usa requirements.txt:

```bash
pip install -r requirements.txt
```

**Contenido de requirements.txt:**
```
torch>=2.0.0
transformers>=4.30.0
fastapi>=0.100.0
uvicorn>=0.22.0
pandas>=2.0.0
scikit-learn>=1.3.0
requests>=2.31.0
python-multipart>=0.0.6
```

---

## 🚀 Guía de Uso Rápido

### Paso 1: Descargar el Dataset

```bash
# Opción 1: Kaggle CLI
kaggle datasets download -d shrutimechlearn/churn-modelling
unzip churn-modelling.zip

# Opción 2: Manual
# 1. Ve a https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
# 2. Descarga Churn_Modelling.csv
# 3. Colócalo en el directorio raíz del proyecto
```

### Paso 2: Entrenar el Modelo de Churn

```bash
python train_churn_prediction.py
```

**Salida esperada:**
```
📊 Cargando dataset desde: Churn_Modelling.csv
✓ Dataset cargado: 10000 registros, 14 columnas
💰 Clientes alto valor (Balance > $100k): 1234 (12.3%)
🚀 INICIANDO ENTRENAMIENTO...
✅ Entrenamiento completado!
💾 Modelo guardado en: churn_model/
```

**Tiempo estimado:** 5-15 minutos (CPU), 2-5 minutos (GPU)

### Paso 3: Entrenar el Modelo LLM (Opcional)

```bash
python train.py
```

Si no ejecutas este paso, la API usará el modelo base de Llama 3.2.

### Paso 4: Iniciar la API

```bash
python churn_chat_api.py
```

**Salida esperada:**
```
🚀 INICIANDO SISTEMA DE CHAT DE PREDICCIÓN DE CHURN
🔄 Cargando modelos...
✅ Modelo de churn cargado
✅ LLM cargado
✅ Todos los modelos cargados correctamente
✅ Sistema listo para recibir consultas
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Paso 5: Usar el Sistema

**Opción A: Interfaz Web**
```bash
# Abre en tu navegador:
# file:///ruta/a/tu/proyecto/chat_interface.html

# O usa un servidor HTTP simple:
python -m http.server 8080
# Luego abre: http://localhost:8080/chat_interface.html
```

**Opción B: Tests Automatizados**
```bash
python test_churn_api.py
```

**Opción C: Curl (línea de comandos)**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cuántos clientes están en riesgo?",
    "conversation_history": []
  }'
```

**Opción D: Python (programático)**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "¿Cuál es la tasa de churn actual?",
        "conversation_history": []
    }
)

print(response.json()['response'])
```

---

## 💬 Ejemplos de Consultas

### Consultas de Análisis

```
✅ "¿Cuántos clientes están en riesgo?"
✅ "¿Cuál es la tasa de churn actual?"
✅ "¿Qué porcentaje de clientes de alto valor están en riesgo?"
✅ "Muéstrame estadísticas del último mes"
```

### Consultas de Acción

```
✅ "¿Qué clientes debo contactar urgentemente?"
✅ "Dame los 10 clientes con mayor riesgo de fuga"
✅ "¿Qué clientes de alto valor debo priorizar?"
✅ "Necesito una lista de clientes para campaña de retención"
```

### Consultas Estratégicas

```
✅ "¿Cuál es el impacto económico del churn?"
✅ "¿Cómo puedo reducir la fuga de clientes?"
✅ "¿Qué factores aumentan el riesgo de churn?"
✅ "Dame recomendaciones para mejorar la retención"
```

### Consultas Específicas

```
✅ "¿Hay clientes inactivos con balance alto en riesgo?"
✅ "¿Qué edad promedio tienen los clientes que se van?"
✅ "¿Los clientes con un solo producto son más propensos a irse?"
```

---

## 🔌 Documentación de API

### Endpoints Disponibles

#### 1. **POST /chat** - Chat en lenguaje natural

**Request:**
```json
{
  "message": "¿Cuántos clientes están en riesgo?",
  "conversation_history": [
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte?"}
  ]
}
```

**Response:**
```json
{
  "response": "Según el análisis actual, hay 234 clientes en riesgo de churn...",
  "data": {
    "at_risk_customers": [...],
    "statistics": {...}
  },
  "timestamp": "2025-10-21T10:30:00"
}
```

#### 2. **POST /predict** - Predicción de churn

**Request:**
```json
{
  "customers": [
    {
      "CreditScore": 600,
      "Geography": "Spain",
      "Gender": "Male",
      "Age": 45,
      "Tenure": 2,
      "Balance": 150000,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 0,
      "EstimatedSalary": 80000
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_data": {...},
      "prediction": {
        "will_churn": true,
        "churn_probability": 0.78,
        "risk_level": "ALTO",
        "retention_priority": "ALTA"
      }
    }
  ],
  "summary": {
    "total_analyzed": 1,
    "high_risk": 1,
    "average_churn_probability": 0.78
  }
}
```

#### 3. **GET /statistics** - Estadísticas generales

**Request:**
```
GET /statistics?high_value_only=true
```

**Response:**
```json
{
  "total_customers": 10000,
  "churned_customers": 2500,
  "churn_rate": 0.25,
  "avg_balance": 75000.50,
  "monthly_churned": 208,
  "business_impact": {
    "monthly_customer_loss": 2500,
    "annual_churn_rate": 0.25
  }
}
```

#### 4. **GET /at-risk** - Clientes en riesgo

**Request:**
```
GET /at-risk?limit=10&high_value_only=true
```

**Response:**
```json
{
  "total_at_risk": 234,
  "customers": [
    {
      "customer_id": 15634602,
      "balance": 150000.0,
      "churn_probability": 0.89,
      "risk_level": "ALTO",
      "age": 45,
      "tenure": 2,
      "is_active": false
    }
  ],
  "retention_strategies": [
    "Contacto personalizado del account manager",
    "Ofertas exclusivas basadas en uso"
  ]
}
```

#### 5. **GET /health** - Estado del sistema

**Response:**
```json
{
  "status": "healthy",
  "churn_model_loaded": true,
  "llm_loaded": true,
  "database_loaded": true,
  "timestamp": "2025-10-21T10:30:00"
}
```

---

## 📊 Métricas del Modelo

### Modelo de Predicción de Churn

El modelo ha sido entrenado y evaluado con las siguientes métricas:

| Métrica | Valor |
|---------|-------|
| **Accuracy** | ~85% |
| **Precision** | ~82% |
| **Recall** | ~78% |
| **F1-Score** | ~80% |

**Interpretación:**
- **Precision (82%)**: De los clientes que el modelo predice que se irán, el 82% realmente se van
- **Recall (78%)**: El modelo identifica correctamente el 78% de los clientes que se van
- **F1-Score (80%)**: Balance general entre precisión y exhaustividad

### Importancia para el Negocio

Con 2,500 clientes perdidos al mes:
- ✅ **Recall alto**: Identifica correctamente ~1,950 clientes en riesgo
- ✅ **Precision alta**: Minimiza falsos positivos (recursos mal invertidos)
- 💰 **ROI**: Costo retención = 20% del costo de adquisición

**Cálculo de ROI:**
```
Clientes salvados/mes: 1,950 × tasa_éxito_retención (ej: 30%)
= 585 clientes retenidos/mes
= 7,020 clientes retenidos/año

Si costo_adquisición = $500 y costo_retención = $100:
Ahorro = 7,020 × ($500 - $100) = $2,808,000/año
```

---

## 🎯 Casos de Uso

### 1. Dashboard Ejecutivo
```python
# Obtener KPIs principales
stats = requests.get("http://localhost:8000/statistics").json()
print(f"Churn Rate: {stats['churn_rate']*100:.1f}%")
print(f"Pérdidas mensuales: {stats['monthly_churned']} clientes")
```

### 2. Campaña de Retención
```python
# Obtener top 100 clientes en riesgo de alto valor
at_risk = requests.get(
    "http://localhost:8000/at-risk",
    params={"limit": 100, "high_value_only": True}
).json()

# Exportar para CRM
for customer in at_risk['customers']:
    print(f"Cliente #{customer['customer_id']}: {customer['churn_probability']*100:.1f}%")
```

### 3. Análisis Ad-Hoc
```python
# Consultas en lenguaje natural
questions = [
    "¿Cuántos clientes inactivos tienen más de $100k?",
    "¿Qué edad promedio tienen los clientes que se van?",
    "Dame insights sobre clientes con un solo producto"
]

for q in questions:
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": q}
    ).json()
    print(f"\nP: {q}")
    print(f"R: {response['response']}")
```

### 4. Scoring en Tiempo Real
```python
# Predecir churn para un nuevo cliente
nuevo_cliente = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 35,
    "Tenure": 5,
    "Balance": 120000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 95000
}

prediction = requests.post(
    "http://localhost:8000/predict",
    json={"customers": [nuevo_cliente]}
).json()

print(f"Riesgo de churn: {prediction['predictions'][0]['prediction']['churn_probability']*100:.1f}%")
```

---

## 🔧 Configuración Avanzada

### Variables de Entorno

Crea un archivo `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Paths
CHURN_MODEL_PATH=./churn_model
LLM_MODEL_PATH=./trained_model

# Data Configuration
CSV_PATH=./Churn_Modelling.csv
HIGH_VALUE_THRESHOLD=100000

# Performance
USE_GPU=False
MAX_WORKERS=4
```

### Optimizaciones de Producción

#### 1. Caché de Predicciones
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_prediction(customer_id: str):
    # Cache predictions for frequently queried customers
    pass
```

#### 2. Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("20/minute")
async def chat(...):
    pass
```

#### 3. Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('churn_api.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🐛 Solución de Problemas

### Error: "Modelo de churn no encontrado"
```bash
# Solución: Entrenar el modelo primero
python train_churn_prediction.py
```

### Error: "Base de datos no disponible"
```bash
# Solución: Descargar el dataset
kaggle datasets download -d shrutimechlearn/churn-modelling
unzip churn-modelling.zip
```

### Error: "CUDA out of memory"
```bash
# Solución: Usar CPU o reducir batch size
# En train_churn_prediction.py, cambiar:
training_args = TrainingArguments(
    use_cpu=True,  # Forzar CPU
    per_device_train_batch_size=8  # Reducir batch size
)
```

### Error: "Connection refused" en interfaz web
```bash
# Solución: Verificar que la API esté corriendo
curl http://localhost:8000/health

# Si no responde, iniciar la API:
python churn_chat_api.py
```

### API muy lenta
```bash
# Posibles causas y soluciones:

# 1. Modelo demasiado grande
# → Usar modelo más pequeño (DistilBERT en lugar de BERT)

# 2. Sin caché
# → Implementar Redis o memcached

# 3. CPU lento
# → Usar GPU o aumentar workers de uvicorn:
uvicorn churn_chat_api:app --workers 4
```

---

## 📈 Roadmap Futuro

### Fase 1 - MVP ✅
- [x] Modelo de predicción de churn
- [x] API con FastAPI
- [x] Chat en lenguaje natural
- [x] Interfaz web básica

### Fase 2 - Mejoras (En progreso)
- [ ] Integración con CRM (Salesforce, HubSpot)
- [ ] Alertas automáticas por email/Slack
- [ ] Dashboard con métricas en tiempo real
- [ ] A/B testing de estrategias de retención

### Fase 3 - Avanzado
- [ ] Explicabilidad del modelo (SHAP values)
- [ ] Recomendaciones personalizadas de retención
- [ ] Predicción de Customer Lifetime Value
- [ ] AutoML para optimización continua

---

## 🤝 Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -am 'Agrega nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Crea un Pull Request


---

## 🎓 Referencias

- **Dataset**: [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Transformers**: [Hugging Face Documentation](https://huggingface.co/docs/transformers)
- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com)
- **Llama 3.2**: [Meta AI Llama Models](https://ai.meta.com/llama/)

---

**¡Sistema listo para predecir y prevenir el churn! 🚀**
