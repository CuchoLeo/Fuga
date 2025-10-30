# 📚 DOCUMENTACIÓN COMPLETA DEL CÓDIGO - CHURNITO

Esta guía explica **línea por línea** cómo funciona el sistema Churnito.

---

## 📁 ESTRUCTURA DEL PROYECTO

```
Fuga/
├── churn_chat_api.py          # 🎯 API principal con FastAPI y chat con LLM
├── train_churn_prediction.py  # 🏋️ Entrenamiento del modelo de predicción
├── run_local.py                # 🚀 Script para ejecutar servidor localmente
├── chat_interface.html         # 💬 Interfaz web del chat
├── Churn_Modelling.csv         # 📊 Dataset de clientes
├── Dockerfile                  # 🐳 Construcción de imagen Docker
├── docker-compose.yml          # 🐳 Orquestación de contenedores
├── requirements.txt            # 📦 Dependencias Python
└── Churnito_Colab.ipynb       # 📓 Notebook para Google Colab
```

---

## 🎯 1. ARCHIVO PRINCIPAL: `churn_chat_api.py`

### **📌 Sección 1: Importaciones (Líneas 1-14)**

```python
from fastapi import FastAPI, HTTPException
# FastAPI: Framework web moderno para crear APIs REST
# HTTPException: Manejo de errores HTTP (404, 500, etc.)

from fastapi.responses import FileResponse
# FileResponse: Sirve archivos HTML/CSS/JS directamente al navegador

from fastapi.middleware.cors import CORSMiddleware
# CORS (Cross-Origin Resource Sharing): Permite que navegadores web
# hagan peticiones a la API desde diferentes dominios
# Sin esto, el navegador bloquearía las peticiones por seguridad

from pydantic import BaseModel
# Pydantic: Validación automática de datos con tipos
# Ejemplo: Si envías "edad" como texto en vez de número, Pydantic lo rechaza

from typing import List, Optional, Dict, Any
# Type hints para mejor documentación y ayuda del IDE
# List[str] = lista de strings
# Optional[int] = número o None
# Dict[str, Any] = diccionario con keys string y values de cualquier tipo

import torch
# PyTorch: Biblioteca de deep learning
# Ejecuta modelos de redes neuronales (DistilBERT, Qwen2.5)
# Maneja tensores (arrays multidimensionales optimizados para GPU/CPU)

import pandas as pd
# Pandas: Manejo de datos tabulares (como Excel en Python)
# DataFrame = tabla con filas y columnas
# Operaciones: filtrado, agrupación, estadísticas

import numpy as np
# NumPy: Operaciones matemáticas rápidas con arrays
# Normalización, promedios, álgebra lineal
# Base de Pandas y PyTorch

from pathlib import Path
# Path: Manejo moderno de rutas de archivos
# Multiplataforma: funciona igual en Windows, Mac y Linux
# Reemplaza os.path con sintaxis más limpia

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
# Transformers (Hugging Face): Biblioteca de modelos pre-entrenados
# AutoModelForCausalLM: LLM para generar texto (GPT, Llama, Qwen)
# AutoModelForSequenceClassification: Clasificación de texto (churn/no-churn)
# AutoTokenizer: Convierte texto a números que entiende el modelo

import pickle
# Pickle: Serialización de objetos Python
# Guarda y carga objetos complejos (scaler, encoders, modelos)
# Formato binario, solo para Python

import json
# JSON: Formato de datos estándar para APIs
# Intercambio de datos entre frontend y backend
# Ejemplo: {"nombre": "Juan", "edad": 30}

from datetime import datetime
# datetime: Manejo de fechas y timestamps
# Para marcar cuándo se hizo cada predicción
# Importante para auditoría y logging

import os
# os: Interacción con el sistema operativo
# Variables de entorno, paths, comandos del sistema
```

**¿Por qué tantas bibliotecas?**
- **FastAPI:** API web moderna y rápida
- **PyTorch + Transformers:** Ejecutar modelos de IA
- **Pandas + NumPy:** Procesamiento de datos
- **Pydantic:** Validación automática
- Cada una tiene un propósito específico

---

### **📌 Sección 2: Configuración de FastAPI (Líneas 16-33)**

```python
# Crear instancia de la aplicación FastAPI
app = FastAPI(
    title="Sistema de Predicción de Churn - Chat API",
    # ↑ Título que aparece en la documentación automática (/docs)

    description="API conversacional para análisis y predicción de fuga de clientes",
    # ↑ Descripción larga que explica qué hace la API

    version="1.0.0"
    # ↑ Versión semántica: MAJOR.MINOR.PATCH
    # Incrementar cuando hay cambios en la API
)

# Configurar CORS (Cross-Origin Resource Sharing)
# CORS es una medida de seguridad del navegador que bloquea peticiones
# entre diferentes dominios. Por ejemplo:
# - Frontend en http://localhost:3000
# - Backend en http://localhost:8000
# Sin CORS, el navegador bloquearía estas peticiones

app.add_middleware(
    CORSMiddleware,  # Middleware que intercepta todas las peticiones

    allow_origins=["*"],
    # ↑ Permitir peticiones desde CUALQUIER origen
    # En producción cambiar a: ["https://miapp.com"]
    # ["*"] es solo para desarrollo

    allow_credentials=True,
    # ↑ Permitir envío de cookies y headers de autenticación
    # Necesario para sesiones y JWT

    allow_methods=["*"],
    # ↑ Permitir todos los métodos HTTP: GET, POST, PUT, DELETE, etc.
    # En producción especificar: ["GET", "POST"]

    allow_headers=["*"],
    # ↑ Permitir todos los headers HTTP
    # En producción especificar: ["Content-Type", "Authorization"]
)
```

**Conceptos clave:**

1. **¿Qué es un Middleware?**
   - Código que se ejecuta ANTES de cada petición
   - Como un "filtro" o "inspector" de peticiones
   - Útil para: logging, autenticación, CORS, rate limiting

2. **¿Por qué CORS?**
   - Navegadores bloquean peticiones entre dominios por seguridad
   - Previene ataques de tipo Cross-Site Request Forgery (CSRF)
   - CORS da permiso explícito para ciertas peticiones

3. **Documentación automática:**
   - FastAPI genera `/docs` (Swagger UI) automáticamente
   - Prueba endpoints directamente desde el navegador
   - No necesitas escribir documentación manualmente

---

### **📌 Sección 3: Modelos de Datos con Pydantic (Líneas 35-65)**

```python
class ChatRequest(BaseModel):
    """
    Estructura de datos para peticiones de chat

    Ejemplo de JSON esperado:
    {
        "message": "¿Cuántos clientes están en riesgo?",
        "conversation_history": [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Hola, soy Churnito"}
        ]
    }
    """
    message: str
    # ↑ Mensaje del usuario (obligatorio)
    # Pydantic valida que sea string y que exista

    conversation_history: Optional[List[Dict[str, str]]] = []
    # ↑ Historial de conversación (opcional)
    # Optional = puede ser None
    # List[Dict[str, str]] = lista de diccionarios con keys y values string
    # = [] → valor por defecto es lista vacía

class ChatResponse(BaseModel):
    """
    Estructura de datos para respuestas de chat

    Ejemplo de JSON enviado:
    {
        "response": "Hay 127 clientes en riesgo de churn...",
        "data": {
            "at_risk_count": 127,
            "high_value_count": 45
        },
        "timestamp": "2025-10-30T18:45:23.123456"
    }
    """
    response: str
    # ↑ Texto de respuesta generado por Churnito

    data: Optional[Dict[str, Any]] = None
    # ↑ Datos estructurados opcionales
    # Dict[str, Any] = diccionario con keys string y values de cualquier tipo
    # Útil para pasar datos que el frontend puede graficar

    timestamp: str
    # ↑ Timestamp ISO 8601 de cuándo se generó la respuesta
    # Formato: "2025-10-30T18:45:23.123456"

class CustomerData(BaseModel):
    """
    Estructura de datos de un cliente para predicción

    Campos explicados:
    """
    CreditScore: float
    # ↑ Puntaje crediticio: 300-850
    # Indica qué tan confiable es el cliente para préstamos
    # Más alto = mejor historial crediticio

    Geography: str
    # ↑ País del cliente: "France", "Spain", "Germany"
    # Se codificará a números con LabelEncoder
    # France=0, Germany=1, Spain=2 (ejemplo)

    Gender: str
    # ↑ Género: "Male" o "Female"
    # Se codificará a números: Male=0, Female=1

    Age: float
    # ↑ Edad: 18-95 años
    # Feature importante: clientes muy jóvenes o muy mayores
    # tienen diferentes tasas de churn

    Tenure: float
    # ↑ Años como cliente: 0-10
    # Cuánto tiempo lleva siendo cliente del banco
    # Más tenure = menor probabilidad de churn (generalmente)

    Balance: float
    # ↑ Saldo en cuenta: $0-$250,000
    # Feature MUY importante para churn
    # Balance alto + churn = pérdida grande

    NumOfProducts: float
    # ↑ Número de productos contratados: 1-4
    # Ejemplos: cuenta corriente, tarjeta, préstamo, seguro
    # Más productos = más "atado" al banco = menor churn

    HasCrCard: float
    # ↑ Tiene tarjeta de crédito: 0=No, 1=Sí
    # Binary feature

    IsActiveMember: float
    # ↑ Cliente activo: 0=Inactivo, 1=Activo
    # Clientes inactivos tienen MUCHO mayor riesgo de churn

    EstimatedSalary: float
    # ↑ Salario estimado anual: $0-$200,000
    # Contexto de capacidad económica del cliente

class PredictionRequest(BaseModel):
    """Petición para predicción de múltiples clientes"""
    customers: List[CustomerData]
    # ↑ Lista de clientes a analizar
    # Permite análisis batch (múltiples clientes a la vez)

class PredictionResponse(BaseModel):
    """Respuesta con predicciones para múltiples clientes"""
    predictions: List[Dict[str, Any]]
    # ↑ Lista de predicciones, una por cada cliente

    summary: Dict[str, Any]
    # ↑ Resumen estadístico de las predicciones
    # Ejemplo: {"high_risk_count": 12, "avg_churn_probability": 0.35}
```

**Beneficios de usar Pydantic:**

1. **Validación automática:**
   ```python
   # JSON inválido → Error 422 automático
   {"message": 123}  # ❌ message debe ser string
   {"message": "Hola", "edad": 30}  # ✅ campos extra se ignoran
   ```

2. **Documentación automática:**
   - FastAPI usa estos modelos para generar /docs
   - Swagger UI muestra ejemplos automáticamente

3. **Type safety:**
   - El IDE te ayuda con autocompletado
   - Detecta errores antes de ejecutar

4. **Conversión automática:**
   ```python
   # Si recibes:
   {"age": "30"}  # String
   # Pydantic convierte a:
   {"age": 30.0}  # Float
   ```

---

### **📌 Sección 4: Clase Principal - ChurnChatSystem**

#### **4.1 Inicialización**

```python
class ChurnChatSystem:
    """
    Sistema central que orquesta:
    1. Predicción de churn (DistilBERT)
    2. Chat conversacional (LLM Qwen2.5)
    3. Análisis de clientes en riesgo
    4. Generación de recomendaciones

    Arquitectura:
    - churn_model: Clasifica si un cliente hará churn (Sí/No)
    - llm_model: Genera respuestas conversacionales en español
    - customer_database: DataFrame con todos los clientes
    - scaler: Normaliza features numéricos
    - label_encoders: Codifica categorías a números
    """

    def __init__(self):
        """
        Inicializa atributos vacíos
        Los modelos se cargan después con load_models()
        para control de memoria y tiempo de inicio
        """
        # Modelos de predicción
        self.churn_model = None          # DistilBERT fine-tuned
        self.churn_tokenizer = None      # Tokenizer de DistilBERT

        # Modelos de chat
        self.llm_model = None            # Qwen2.5-1.5B-Instruct
        self.llm_tokenizer = None        # Tokenizer de Qwen

        # Artefactos de preprocesamiento
        self.scaler = None               # StandardScaler de scikit-learn
        self.label_encoders = None       # Dict: {"Geography": LabelEncoder, "Gender": LabelEncoder}
        self.feature_names = None        # Lista: ["CreditScore", "Geography", ...]

        # Base de datos
        self.customer_database = None    # pandas DataFrame
```

**¿Por qué None?**
- Modelos son archivos grandes (256MB + 3GB)
- Cargarlos al inicio haría que la app tarde minutos en iniciar
- Se cargan bajo demanda en `load_models()`
- Permite tests sin necesidad de modelos

---

## 🏋️ 2. ENTRENAMIENTO: `train_churn_prediction.py`

Este script entrena el modelo de predicción de churn.

### **Flujo del entrenamiento:**

```
1. Cargar datos (Churn_Modelling.csv)
   ↓
2. Preprocesar:
   - Eliminar columnas irrelevantes (RowNumber, CustomerId, Surname)
   - Codificar categóricas (Geography, Gender) → números
   - Normalizar features numéricos (mean=0, std=1)
   ↓
3. Dividir train/test (80%/20%)
   ↓
4. Convertir features a texto descriptivo
   Ejemplo: "Cliente: CreditScore=650 Geography=1 Gender=0 ..."
   ↓
5. Tokenizar texto (convertir a números para el modelo)
   ↓
6. Entrenar DistilBERT (1 época)
   ↓
7. Evaluar en conjunto de test
   ↓
8. Guardar modelo + artefactos
```

### **Código clave:**

```python
# Codificación de categóricas
label_encoders = {}
for col in ['Geography', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    # Ejemplo: ['France', 'Spain', 'Germany'] → [0, 1, 2]

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Transforma cada feature a: (x - mean) / std
# Resultado: mean≈0, std≈1
# Importante para redes neuronales

# Configuración de entrenamiento
training_args = TrainingArguments(
    num_train_epochs=1,              # 1 época (rápido, para demo)
    per_device_train_batch_size=32,  # 32 muestras por batch
    learning_rate=2e-5,               # Tasa de aprendizaje pequeña
    # ...
)

# Entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()  # ¡Magia! Entrena automáticamente
```

---

## 🚀 3. EJECUCIÓN LOCAL: `run_local.py`

Script simple para ejecutar el servidor sin Docker.

```python
def main():
    # 1. Verificar archivos necesarios
    check_requirements()

    # 2. Verificar dependencias instaladas
    check_dependencies()

    # 3. Crear directorios
    os.makedirs("churn_model", exist_ok=True)
    os.makedirs("trained_model", exist_ok=True)

    # 4. Ejecutar servidor con uvicorn
    import uvicorn
    uvicorn.run(
        "churn_chat_api:app",    # módulo:variable
        host="0.0.0.0",           # Escuchar en todas las interfaces
        port=8000,                 # Puerto 8000
        reload=False,              # Sin auto-reload (evita loop infinito)
        log_level="info"           # Nivel de logging
    )
```

**Diferencia con Docker:**
- Docker: Ambiente aislado, mismo en todos los sistemas
- Local: Usa tu Python local, puede variar

---

## 📊 CONCEPTOS CLAVE

### **1. ¿Qué es un Tokenizer?**

Los modelos no entienden texto, solo números.

```
Texto: "Hola mundo"
   ↓ Tokenizer
Tokens: [101, 2534, 3256, 102]
   ↓ Modelo
Resultado: [0.2, 0.8]  # Probabilidades
```

### **2. ¿Qué es StandardScaler?**

Normaliza features a media 0, desviación 1.

```
Antes:  Age=[20, 30, 40, 50, 60]  Balance=[0, 50k, 100k, 150k, 200k]
Después: Age=[-1.4, -0.7, 0, 0.7, 1.4]  Balance=[-1.4, -0.7, 0, 0.7, 1.4]
```

**¿Por qué?**
- Redes neuronales funcionan mejor con datos normalizados
- Evita que features grandes dominen el aprendizaje

### **3. ¿Qué es LabelEncoder?**

Convierte categorías a números.

```
Geography: ['France', 'Spain', 'Germany', 'France']
    ↓ LabelEncoder
Geografia_encoded: [0, 1, 2, 0]
```

### **4. ¿Qué es DistilBERT?**

Modelo de lenguaje pre-entrenado:
- **Base:** BERT (Google, 2018)
- **Distil:** Versión "destilada" (más pequeña, más rápida)
- **Usos:** Clasificación de texto, Q&A, sentiment analysis

**En Churnito:**
- Toma descripción textual del cliente
- Clasifica en: Churn (1) o No-Churn (0)

### **5. ¿Qué es Qwen2.5?**

LLM (Large Language Model) de Alibaba:
- 1.5 mil millones de parámetros
- Multilingüe (español excelente)
- Open source (Apache 2.0)

**En Churnito:**
- Genera respuestas conversacionales
- Analiza contexto y datos
- Recomienda estrategias

---

## 🔄 FLUJO COMPLETO DE UNA CONSULTA

```
Usuario escribe: "Muéstrame clientes en riesgo"
   ↓
1. Navegador → POST /chat
   ↓
2. FastAPI recibe petición
   ↓
3. Pydantic valida JSON
   ↓
4. analyze_query() detecta intención: "requires_analysis=True"
   ↓
5. get_at_risk_customers():
   - Muestrea 100 clientes
   - Para cada uno: predict_churn()
   - Filtra los que tienen prob > 0.5
   - Ordena por probabilidad descendente
   ↓
6. generate_llm_response():
   - Usa sistema estructurado (no LLM)
   - Formatea datos reales
   - Genera respuesta con IDs, probabilidades, balances
   ↓
7. FastAPI devuelve JSON
   ↓
8. Navegador muestra respuesta
```

---

## 🎓 PARA APRENDER MÁS

### **FastAPI:**
- [Documentación oficial](https://fastapi.tiangolo.com/)
- Tutorial interactivo incluido

### **Transformers:**
- [Hugging Face Course](https://huggingface.co/course)
- Gratis, muy didáctico

### **Machine Learning:**
- [Scikit-learn](https://scikit-learn.org/stable/tutorial/index.html)
- Ejemplos prácticos

### **PyTorch:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Desde básico a avanzado

---

**¿Preguntas? Revisa el código con estos conceptos en mente.** 🚀
