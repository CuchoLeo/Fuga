from fastapi import FastAPI, HTTPException  # Framework web para crear la API REST
from fastapi.middleware.cors import CORSMiddleware  # Middleware para permitir peticiones desde otros dominios
from pydantic import BaseModel  # Validación de datos con tipos
from typing import List, Optional, Dict, Any  # Tipos de datos para type hints
import torch  # PyTorch para ejecutar modelos de deep learning
import pandas as pd  # Manejo de datasets y DataFrames
import numpy as np  # Operaciones numéricas y arrays
from pathlib import Path  # Manejo de rutas de archivos multiplataforma
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification  # Modelos de Hugging Face
import pickle  # Serialización de objetos Python (scaler, encoders)
import json  # Manejo de datos JSON
from datetime import datetime  # Timestamps para respuestas
import os  # Acceso a variables de entorno

# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

app = FastAPI(
    title="Sistema de Predicción de Churn - Chat API",
    description="API conversacional para análisis y predicción de fuga de clientes",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELOS DE DATOS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str

class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

class PredictionRequest(BaseModel):
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]

# ============================================================================
# CARGA DE MODELOS Y ARTEFACTOS
# ============================================================================

class ChurnChatSystem:
    def __init__(self):
        self.churn_model = None
        self.churn_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.customer_database = None
        
    def load_models(self):
        """Carga todos los modelos necesarios"""
        print("🔄 Cargando modelos...")
        
        # 1. Cargar modelo de predicción de churn
        churn_model_path = Path("churn_model")
        if churn_model_path.exists():
            print("📦 Cargando modelo de predicción de churn...")
            self.churn_tokenizer = AutoTokenizer.from_pretrained(churn_model_path)
            self.churn_model = AutoModelForSequenceClassification.from_pretrained(
                churn_model_path,
                torch_dtype=torch.float32
            )
            self.churn_model.eval()
            
            # Cargar artefactos de preprocesamiento
            artifacts_path = churn_model_path / "preprocessing_artifacts.pkl"
            if artifacts_path.exists():
                with open(artifacts_path, 'rb') as f:
                    artifacts = pickle.load(f)
                    self.scaler = artifacts['scaler']
                    self.label_encoders = artifacts['label_encoders']
                    self.feature_names = artifacts['feature_names']
            print("✅ Modelo de churn cargado")
        else:
            print("⚠️  Modelo de churn no encontrado. Ejecuta train_churn_prediction.py primero")
        
        # ========================================================================
        # 2. CARGAR MODELO LLM PARA CONVERSACIÓN (Qwen2.5-1.5B-Instruct)
        # ========================================================================
        # Qwen2.5 es un modelo open-source de Alibaba Cloud
        # Ventajas vs Llama 3.2:
        #   - NO requiere autenticación (descarga directa)
        #   - Más ligero (~3GB vs ~4GB)
        #   - Excelente soporte multilingüe (incluyendo español)
        #   - Optimizado para seguir instrucciones
        #   - Licencia Apache 2.0 (completamente open source)
        try:
            # ID del modelo en Hugging Face Hub
            # Qwen2.5-1.5B-Instruct: Modelo optimizado para seguir instrucciones
            # NO requiere autenticación (a diferencia de Llama)
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"

            # Intentar cargar desde disco local primero (para reutilizar descargas previas)
            llm_model_path = Path("trained_model")
            loaded_from_disk = False

            if llm_model_path.exists():
                try:
                    print("🤖 Intentando cargar LLM desde disco local (trained_model/)...")

                    # Intentar cargar el tokenizer
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(
                        llm_model_path,
                        trust_remote_code=True
                    )

                    # Intentar cargar el modelo
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )

                    self.llm_model.eval()

                    if self.llm_tokenizer.pad_token is None:
                        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                    print("✅ LLM cargado exitosamente desde disco local")
                    loaded_from_disk = True

                except Exception as disk_error:
                    print(f"⚠️  Error al cargar desde disco: {disk_error}")
                    print("🔄 Intentando descargar modelo fresco desde Hugging Face...")
                    loaded_from_disk = False

            # Si no se cargó desde disco, descargar de Hugging Face
            if not loaded_from_disk:
                print("⚠️  Modelo LLM no encontrado localmente (o corrupto)")
                print("🌐 Descargando Qwen2.5-1.5B-Instruct desde Hugging Face...")
                print(f"📥 Esto puede tardar varios minutos (descarga ~3GB)...")

                # Descargar tokenizer
                print("📦 Descargando tokenizer...")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )

                # Descargar modelo
                print("📦 Descargando modelo (esto tomará varios minutos)...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    cache_dir="./trained_model"
                )

                self.llm_model.eval()

                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                print("✅ LLM descargado y cargado exitosamente")
                print("💾 Modelo guardado en ./trained_model/ para futuros usos")

        # Manejo de errores en la carga del LLM
        except Exception as e:
            print(f"❌ Error al cargar LLM: {e}")
            print("⚠️  Posibles causas:")
            print("   - Sin conexión a internet para descargar el modelo")
            print("   - Memoria insuficiente (Qwen2.5 requiere ~4GB RAM)")
            print("   - Problemas con trust_remote_code (necesario para Qwen)")
            print("⚠️  La API funcionará con respuestas estructuradas (sin LLM)")

            # Configurar a None para que el sistema use respuestas estructuradas
            self.llm_model = None
            self.llm_tokenizer = None
        
        # 3. Cargar base de datos de clientes (simulada)
        self.load_customer_database()
        
        print("✅ Todos los modelos cargados correctamente\n")
    
    def load_customer_database(self):
        """Carga o simula una base de datos de clientes"""
        csv_path = "Churn_Modelling.csv"
        if Path(csv_path).exists():
            print("📊 Cargando base de datos de clientes...")
            self.customer_database = pd.read_csv(csv_path)
            print(f"✅ {len(self.customer_database)} clientes cargados")
        else:
            print("⚠️  Base de datos no encontrada. Modo simulación activado")
            self.customer_database = None
    
    def predict_churn(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predice churn para un cliente específico"""
        if self.churn_model is None:
            return {"error": "Modelo de churn no disponible"}

        try:
            # Crear una copia de los datos para no modificar el original
            processed_data = customer_data.copy()

            # Codificar variables categóricas usando los label encoders
            if self.label_encoders:
                for col_name, encoder in self.label_encoders.items():
                    if col_name in processed_data:
                        try:
                            # Codificar el valor categórico
                            processed_data[col_name] = encoder.transform([processed_data[col_name]])[0]
                        except ValueError:
                            # Si el valor no existe en el encoder, usar el más común (0)
                            processed_data[col_name] = 0

            # Preparar features en el orden correcto
            features = []
            for feature_name in self.feature_names:
                if feature_name in processed_data:
                    features.append(float(processed_data[feature_name]))
                else:
                    features.append(0.0)

            # Normalizar
            features_scaled = self.scaler.transform([features])

            # Crear texto descriptivo
            text_parts = ["Cliente:"]
            for name, value in zip(self.feature_names, features_scaled[0]):
                text_parts.append(f"{name}={value:.2f}")
            text = " ".join(text_parts)

            # Tokenizar y predecir
            inputs = self.churn_tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256
            )

            with torch.no_grad():
                outputs = self.churn_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                churn_probability = probabilities[0][1].item()

            return {
                "will_churn": bool(prediction),
                "churn_probability": float(churn_probability),
                "risk_level": self._get_risk_level(churn_probability),
                "retention_priority": "ALTA" if churn_probability > 0.7 and customer_data.get('Balance', 0) > 100000 else "MEDIA" if churn_probability > 0.5 else "BAJA"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_risk_level(self, probability: float) -> str:
        """Determina el nivel de riesgo basado en la probabilidad"""
        if probability >= 0.7:
            return "ALTO"
        elif probability >= 0.5:
            return "MEDIO"
        elif probability >= 0.3:
            return "BAJO"
        else:
            return "MUY BAJO"
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analiza la consulta del usuario y extrae la intención"""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "requires_prediction": False,
            "requires_analysis": False,
            "requires_statistics": False
        }
        
        # Detectar intenciones (pueden ser múltiples)
        # Keywords para solicitar análisis de clientes en riesgo
        if any(word in query_lower for word in ["cuántos", "cantidad", "lista", "clientes en riesgo", "top", "dame", "muestra", "quiero ver", "fuga", "mayor riesgo", "más riesgo", "con riesgo", "riesgo de"]):
            intent["type"] = "analysis"
            intent["requires_analysis"] = True

        # Keywords para solicitar estadísticas y situación general
        if any(word in query_lower for word in ["tasa", "porcentaje", "estadística", "métrica", "promedio", "situación", "estado", "cómo está", "cuál es la", "impacto", "análisis general"]):
            intent["requires_statistics"] = True

        # Keywords para predicciones específicas
        if any(word in query_lower for word in ["predice", "predicción", "probabilidad"]):
            intent["requires_prediction"] = True

        # Keywords para filtrar por clientes de alto valor
        if any(word in query_lower for word in ["alto valor", "premium", "balance alto", "mayor"]):
            intent["high_value"] = True

        # Si pregunta qué hacer o cómo reducir, obtener todo el contexto
        if any(word in query_lower for word in ["qué hacer", "cómo reducir", "estrategia", "recomendación", "recomiendas", "sugieres", "plan"]):
            intent["requires_statistics"] = True
            intent["requires_analysis"] = True

        return intent
    
    def get_statistics(self, high_value_only: bool = False) -> Dict[str, Any]:
        """Obtiene estadísticas del dataset"""
        if self.customer_database is None:
            return {
                "error": "Base de datos no disponible",
                "simulated": True,
                "total_customers": 10000,
                "churn_rate": 0.25,
                "high_value_churn_rate": 0.30
            }
        
        df = self.customer_database.copy()
        
        if high_value_only and 'Balance' in df.columns:
            df = df[df['Balance'] > 100000]
        
        stats = {
            "total_customers": len(df),
            "churned_customers": int(df['Exited'].sum()) if 'Exited' in df.columns else 0,
            "churn_rate": float(df['Exited'].mean()) if 'Exited' in df.columns else 0.0,
            "avg_balance": float(df['Balance'].mean()) if 'Balance' in df.columns else 0.0,
            "avg_age": float(df['Age'].mean()) if 'Age' in df.columns else 0.0,
            "avg_credit_score": float(df['CreditScore'].mean()) if 'CreditScore' in df.columns else 0.0
        }
        
        # Calcular pérdidas mensuales
        if 'Exited' in df.columns:
            churned = df[df['Exited'] == 1]
            stats["monthly_churned"] = int(len(churned) / 12) if len(df) > 0 else 0
            stats["estimated_monthly_loss"] = float(churned['Balance'].sum() / 12) if 'Balance' in churned.columns else 0.0
        
        return stats
    
    def get_at_risk_customers(self, limit: int = 10, high_value_only: bool = False) -> List[Dict[str, Any]]:
        """Obtiene lista de clientes en riesgo"""
        if self.customer_database is None or self.churn_model is None:
            return []

        df = self.customer_database.copy()

        # Filtrar alto valor si se requiere
        if high_value_only and 'Balance' in df.columns:
            df = df[df['Balance'] > 100000]

        # Predecir para cada cliente (limitado para performance)
        # Reducido de 1000 a 100 para mejor rendimiento
        sample_size = min(100, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        at_risk = []
        for idx, row in df_sample.iterrows():
            # Early stopping: si ya tenemos suficientes clientes, parar
            if len(at_risk) >= limit * 3:
                break

            customer_data = {
                'CreditScore': row.get('CreditScore', 0),
                'Geography': row.get('Geography', ''),
                'Gender': row.get('Gender', ''),
                'Age': row.get('Age', 0),
                'Tenure': row.get('Tenure', 0),
                'Balance': row.get('Balance', 0),
                'NumOfProducts': row.get('NumOfProducts', 0),
                'HasCrCard': row.get('HasCrCard', 0),
                'IsActiveMember': row.get('IsActiveMember', 0),
                'EstimatedSalary': row.get('EstimatedSalary', 0)
            }

            prediction = self.predict_churn(customer_data)

            if prediction.get('churn_probability', 0) > 0.5:
                at_risk.append({
                    'customer_id': int(row.get('CustomerId', idx)),
                    'balance': float(customer_data['Balance']),
                    'churn_probability': prediction['churn_probability'],
                    'risk_level': prediction['risk_level'],
                    'age': int(customer_data['Age']),
                    'tenure': int(customer_data['Tenure']),
                    'is_active': bool(customer_data['IsActiveMember'])
                })
        
        # Ordenar por probabilidad de churn
        at_risk.sort(key=lambda x: x['churn_probability'], reverse=True)
        
        return at_risk[:limit]
    
    def generate_llm_response(self, query: str, context: Dict[str, Any]) -> str:
        """
        Genera respuesta conversacional usando el LLM (Llama 3.2) con contexto rico

        Args:
            query: Pregunta del usuario en lenguaje natural
            context: Diccionario con datos (estadísticas, clientes en riesgo, etc.)

        Returns:
            Respuesta en texto generada por el LLM
        """
        # Si el LLM no está cargado, usar sistema de recomendaciones estructuradas
        if self.llm_model is None:
            return "Lo siento, el modelo de lenguaje no está disponible."

        try:
            # ====================================================================
            # PASO 1: Construir el prompt con contexto
            # ====================================================================
            # El prompt es la "instrucción completa" que le damos al LLM
            # Incluye: rol del asistente, contexto de negocio, datos actuales, y la pregunta
            prompt = self._build_prompt(query, context)

            # ====================================================================
            # PASO 2: Tokenizar el prompt (convertir texto a números)
            # ====================================================================
            # Los modelos de lenguaje no entienden texto, solo números (tokens)
            inputs = self.llm_tokenizer(
                prompt,                        # Texto a convertir
                return_tensors="pt",           # Devolver tensores de PyTorch
                padding=True,                  # Rellenar para tamaño uniforme
                truncation=True,               # Cortar si es muy largo
                max_length=1024                # Longitud máxima del contexto (más contexto = mejor)
            )

            # ====================================================================
            # PASO 3: Generar respuesta con el modelo LLM
            # ====================================================================
            # torch.no_grad() = no calcular gradientes (más rápido, menos memoria)
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,                      # Pasar todos los inputs tokenizados

                    # Parámetros de generación (OPTIMIZADOS PARA VELOCIDAD):
                    max_new_tokens=150,            # Reducido de 500 a 150 (~120 palabras)
                                                   # Respuestas más cortas = 3-4x más rápido
                    temperature=0.7,               # Controla creatividad (0=determinista, 1=creativo)
                                                   # 0.7 es un buen balance para respuestas profesionales

                    do_sample=True,                # Activar muestreo (permite variedad)
                    top_p=0.9,                     # Nucleus sampling: solo considerar tokens que sumen 90% probabilidad
                    top_k=50,                      # Solo considerar los 50 tokens más probables

                    repetition_penalty=1.2,        # Penalizar palabras repetidas (1.0=sin penalización)
                    no_repeat_ngram_size=3,        # No repetir secuencias de 3 palabras

                    pad_token_id=self.llm_tokenizer.pad_token_id,  # ID del token de relleno
                    eos_token_id=self.llm_tokenizer.eos_token_id   # ID del token de fin de secuencia
                )

            # ====================================================================
            # PASO 4: Decodificar (convertir números de vuelta a texto)
            # ====================================================================
            # skip_special_tokens=True elimina tokens como <pad>, <eos>, etc.
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # ====================================================================
            # PASO 5: Limpiar la respuesta
            # ====================================================================
            # El modelo a veces incluye el prompt completo en la salida, eliminarlo
            if prompt in response:
                response = response.replace(prompt, "").strip()

            # Extraer solo la parte de respuesta después de "Respuesta:"
            if "Respuesta:" in response:
                response = response.split("Respuesta:")[-1].strip()

            # Si la respuesta es muy corta o vacía, usar recomendaciones estructuradas
            # Esto es un fallback por si el LLM no generó bien
            # Umbral reducido a 30 porque ahora generamos respuestas más concisas
            if len(response) < 30:
                response = self._generate_recommendations(context)

            return response

        # ====================================================================
        # Manejo de errores durante la generación
        # ====================================================================
        except Exception as e:
            print(f"❌ Error generando respuesta LLM: {e}")
            # Fallback: usar sistema de recomendaciones estructuradas
            return self._generate_recommendations(context)
    
    def _generate_recommendations(self, context: Dict[str, Any]) -> str:
        """Genera recomendaciones personalizadas basadas en el contexto"""
        recommendations = []

        # Recomendaciones basadas en clientes en riesgo
        if "at_risk_customers" in context:
            at_risk = context["at_risk_customers"]
            if at_risk:
                high_value_count = sum(1 for c in at_risk if c['balance'] > 100000)
                inactive_count = sum(1 for c in at_risk if not c['is_active'])

                recommendations.append(f"🎯 **Análisis de Clientes en Riesgo:**")
                recommendations.append(f"   • {len(at_risk)} clientes identificados con alta probabilidad de churn")
                recommendations.append(f"   • {high_value_count} son clientes de alto valor (Balance > $100k)")
                recommendations.append(f"   • {inactive_count} clientes están inactivos")

                recommendations.append("\n💡 **Recomendaciones Prioritarias:**")

                if high_value_count > 0:
                    recommendations.append(
                        f"   1. **URGENTE**: Contactar a los {high_value_count} clientes de alto valor en riesgo\n"
                        "      - Asignar account manager dedicado\n"
                        "      - Ofrecer consultoría financiera personalizada\n"
                        "      - Incentivos exclusivos por lealtad"
                    )

                if inactive_count > 0:
                    recommendations.append(
                        f"   2. Reactivar {inactive_count} clientes inactivos:\n"
                        "      - Campaña de re-engagement con beneficios especiales\n"
                        "      - Encuesta para entender razones de inactividad\n"
                        "      - Simplificar proceso de uso del servicio"
                    )

                recommendations.append(
                    "   3. Estrategias de retención general:\n"
                    "      - Programa de fidelización escalonado\n"
                    "      - Comunicación proactiva trimestral\n"
                    "      - Mejoras en servicio al cliente"
                )

                # Detalles de top clientes
                recommendations.append("\n📊 **Top 3 Clientes Prioritarios:**")
                for i, customer in enumerate(at_risk[:3], 1):
                    prob_pct = customer['churn_probability'] * 100
                    recommendations.append(
                        f"   {i}. Cliente #{customer['customer_id']}: {prob_pct:.1f}% riesgo, "
                        f"${customer['balance']:,.0f} balance\n"
                        f"      → {'🔴 INACTIVO - Contactar inmediatamente' if not customer['is_active'] else '🟡 Activo - Programa de retención preventivo'}"
                    )

        # Recomendaciones basadas en estadísticas
        elif "statistics" in context:
            stats = context["statistics"]
            churn_rate = stats.get('churn_rate', 0) * 100

            recommendations.append(f"📊 **Análisis de la Situación Actual:**")
            recommendations.append(f"   • Tasa de churn: {churn_rate:.1f}%")
            recommendations.append(f"   • Total de clientes: {stats.get('total_customers', 0):,}")

            if churn_rate > 20:
                recommendations.append("\n⚠️ **ALERTA**: Tasa de churn crítica (>20%)")
                recommendations.append("\n💡 **Acciones Recomendadas Inmediatas:**")
                recommendations.append(
                    "   1. Auditoría de experiencia del cliente\n"
                    "   2. Análisis de competencia y benchmarking\n"
                    "   3. Implementar sistema de alertas tempranas\n"
                    "   4. Crear equipo dedicado a retención"
                )

            if "monthly_churned" in stats:
                monthly_loss = stats['monthly_churned']
                recommendations.append(
                    f"\n💰 **Impacto Económico:**\n"
                    f"   • Pérdida mensual: ~{monthly_loss:,} clientes\n"
                    f"   • ROI de retención: 5x (costo retención = 1/5 costo adquisición)\n"
                    f"   • Priorizar inversión en retención predictiva"
                )

        else:
            recommendations.append(
                "💬 **Puedo ayudarte con:**\n"
                "   • Identificar clientes en riesgo de churn\n"
                "   • Analizar estadísticas y tendencias\n"
                "   • Generar recomendaciones personalizadas\n"
                "   • Priorizar acciones de retención\n\n"
                "Pregúntame sobre clientes en riesgo, estadísticas de churn, o recomendaciones específicas."
            )

        return "\n".join(recommendations)

    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Construye un prompt conversacional con contexto rico para el LLM

        El prompt es fundamental para obtener buenas respuestas del LLM.
        Incluye: rol, contexto de negocio, datos actuales, y la pregunta del usuario.

        Args:
            query: Pregunta del usuario
            context: Datos relevantes (estadísticas, clientes en riesgo, etc.)

        Returns:
            Prompt formateado listo para enviar al LLM
        """
        # ====================================================================
        # SECCIÓN 1: Definir el rol del asistente (system message)
        # ====================================================================
        # Esto le dice al LLM "quién es" y cómo debe comportarse
        prompt_parts = [
            "Eres Churnito, un asistente experto en retención de clientes y análisis de churn.",
            "Tu nombre es Churnito y tu rol es ayudar a empresas a reducir la fuga de clientes mediante insights accionables.",
            "Eres amigable, profesional y siempre proporcionas recomendaciones concretas.",

            # ================================================================
            # SECCIÓN 2: Contexto del negocio (información estática)
            # ================================================================
            # Información que siempre es relevante, independiente de la consulta
            "\n### CONTEXTO DEL NEGOCIO:",
            "- Industria: Servicios financieros/bancarios",
            "- Tasa de churn anual actual: 25% (crítico)",
            "- Clientes perdidos por mes: ~2,500",
            "- Enfoque prioritario: Clientes de alto valor (Balance > $100,000)",
            "- Economía de retención: El costo de retener un cliente es 1/5 del costo de adquirir uno nuevo",
            "- Impacto: Cada cliente perdido representa pérdida de ingresos recurrentes y valor de vida del cliente",
        ]

        # ====================================================================
        # SECCIÓN 3: Datos actuales (dinámicos según la consulta)
        # ====================================================================
        # Si el contexto incluye estadísticas, agregarlas al prompt
        if "statistics" in context:
            stats = context["statistics"]
            prompt_parts.append("\n### DATOS ACTUALES:")
            # Formatear números con separadores de miles para mejor lectura
            prompt_parts.append(f"- Total de clientes en base: {stats.get('total_customers', 'N/A'):,}")
            prompt_parts.append(f"- Tasa de churn actual: {stats.get('churn_rate', 0)*100:.1f}%")
            prompt_parts.append(f"- Balance promedio: ${stats.get('avg_balance', 0):,.2f}")
            prompt_parts.append(f"- Edad promedio: {stats.get('avg_age', 0):.0f} años")

            # Agregar métricas opcionales si están disponibles
            if "monthly_churned" in stats:
                prompt_parts.append(f"- Clientes perdidos este mes: {stats['monthly_churned']:,}")
            if "estimated_monthly_loss" in stats:
                prompt_parts.append(f"- Pérdida estimada mensual: ${stats['estimated_monthly_loss']:,.2f}")

        # ====================================================================
        # SECCIÓN 4: Clientes en riesgo (si aplica)
        # ====================================================================
        # Si la consulta requiere análisis de clientes específicos
        if "at_risk_customers" in context:
            at_risk = context["at_risk_customers"]
            if at_risk:
                prompt_parts.append(f"\n### CLIENTES EN RIESGO IDENTIFICADOS: {len(at_risk)}")

                # Mostrar los top 5 clientes con más riesgo
                # Esto ayuda al LLM a dar recomendaciones específicas
                for i, customer in enumerate(at_risk[:5], 1):  # Solo top 5 para no saturar el prompt
                    prompt_parts.append(
                        f"{i}. Cliente #{customer['customer_id']}: "
                        f"{customer['churn_probability']*100:.1f}% probabilidad, "
                        f"Balance ${customer['balance']:,.0f}, "
                        f"Edad {customer['age']}, "
                        f"{'Activo' if customer['is_active'] else 'Inactivo'}"
                    )

                # Si hay más de 5, indicarlo
                if len(at_risk) > 5:
                    prompt_parts.append(f"... y {len(at_risk) - 5} clientes más en riesgo")

        # ====================================================================
        # SECCIÓN 5: Predicción específica (si aplica)
        # ====================================================================
        # Si se hizo una predicción para un cliente particular
        if "prediction" in context:
            pred = context["prediction"]
            prompt_parts.append("\n### PREDICCIÓN ESPECÍFICA:")
            prompt_parts.append(f"- Probabilidad de churn: {pred.get('churn_probability', 0)*100:.1f}%")
            prompt_parts.append(f"- Nivel de riesgo: {pred.get('risk_level', 'N/A')}")
            prompt_parts.append(f"- Prioridad de retención: {pred.get('retention_priority', 'N/A')}")

        # ====================================================================
        # SECCIÓN 6: La pregunta del usuario
        # ====================================================================
        prompt_parts.append(f"\n### PREGUNTA DEL USUARIO:\n{query}")

        # ====================================================================
        # SECCIÓN 7: Instrucciones para la respuesta
        # ====================================================================
        # Esto guía al LLM sobre qué tipo de respuesta queremos
        prompt_parts.append("\n### TU RESPUESTA:")
        prompt_parts.append("Proporciona una respuesta conversacional, clara y accionable que incluya:")
        prompt_parts.append("1. Análisis de la situación basado en los datos")
        prompt_parts.append("2. Insights específicos y relevantes")
        prompt_parts.append("3. Recomendaciones concretas y priorizadas")
        prompt_parts.append("4. Próximos pasos sugeridos")
        prompt_parts.append("\nRespuesta:")

        # Unir todas las partes con saltos de línea
        return "\n".join(prompt_parts)

# Inicializar sistema global
chat_system = ChurnChatSystem()

# ============================================================================
# EVENTOS DE CICLO DE VIDA
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Cargar modelos al iniciar"""
    print("="*70)
    print("🚀 INICIANDO SISTEMA DE CHAT DE PREDICCIÓN DE CHURN")
    print("="*70)
    chat_system.load_models()
    print("="*70)
    print("✅ Sistema listo para recibir consultas")
    print("="*70)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "API de Predicción de Churn con Chat en Lenguaje Natural",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat - Conversa en lenguaje natural",
            "predict": "/predict - Predicción de churn para clientes",
            "statistics": "/statistics - Estadísticas generales",
            "at_risk": "/at-risk - Lista de clientes en riesgo",
            "health": "/health - Estado del sistema"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal de chat en lenguaje natural
    
    Ejemplos de consultas:
    - "¿Cuántos clientes están en riesgo?"
    - "¿Cuál es la tasa de churn actual?"
    - "Muéstrame los 10 clientes con mayor riesgo de fuga"
    - "¿Cuál es el impacto económico del churn?"
    """
    try:
        query = request.message
        
        # Analizar intención
        intent = chat_system.analyze_query(query)
        
        # Preparar contexto
        context = {}
        
        # Ejecutar acciones según intención
        if intent.get("requires_statistics"):
            context["statistics"] = chat_system.get_statistics(
                high_value_only=intent.get("high_value", False)
            )
        
        if intent.get("requires_analysis"):
            context["at_risk_customers"] = chat_system.get_at_risk_customers(
                limit=10,
                high_value_only=intent.get("high_value", False)
            )
        
        # Generar respuesta con LLM o recomendaciones
        response_text = chat_system.generate_llm_response(query, context)
        
        return ChatResponse(
            response=response_text,
            data=context,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """
    Predice churn para uno o más clientes
    """
    try:
        predictions = []
        high_risk_count = 0
        total_churn_prob = 0
        
        for customer in request.customers:
            customer_dict = customer.dict()
            prediction = chat_system.predict_churn(customer_dict)
            
            result = {
                "customer_data": customer_dict,
                "prediction": prediction
            }
            predictions.append(result)
            
            if prediction.get('churn_probability', 0) > 0.7:
                high_risk_count += 1
            total_churn_prob += prediction.get('churn_probability', 0)
        
        summary = {
            "total_analyzed": len(predictions),
            "high_risk": high_risk_count,
            "average_churn_probability": total_churn_prob / len(predictions) if predictions else 0,
            "recommendation": "Implementar estrategias de retención inmediata" if high_risk_count > 0 else "Monitoreo rutinario"
        }
        
        return PredictionResponse(
            predictions=predictions,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics(high_value_only: bool = False):
    """
    Obtiene estadísticas generales del churn
    """
    try:
        stats = chat_system.get_statistics(high_value_only=high_value_only)
        
        # Calcular métricas adicionales de negocio
        if not stats.get("error"):
            stats["business_impact"] = {
                "monthly_customer_loss": 2500,  # Dato del problema
                "annual_churn_rate": 0.25,
                "retention_cost_ratio": 0.2,  # 1/5 del costo de adquisición
                "estimated_savings": "Potencial reducción de pérdidas mediante retención predictiva"
            }
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/at-risk")
async def get_at_risk(limit: int = 10, high_value_only: bool = False):
    """
    Obtiene lista de clientes en riesgo de churn
    """
    try:
        at_risk = chat_system.get_at_risk_customers(limit=limit, high_value_only=high_value_only)
        
        return {
            "total_at_risk": len(at_risk),
            "customers": at_risk,
            "recommendation": "Priorizar acciones de retención en los clientes listados",
            "retention_strategies": [
                "Contacto personalizado del account manager",
                "Ofertas exclusivas basadas en uso",
                "Mejora de servicios específicos",
                "Incentivos por renovación anticipada"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Verifica el estado del sistema
    """
    return {
        "status": "healthy",
        "churn_model_loaded": chat_system.churn_model is not None,
        "llm_loaded": chat_system.llm_model is not None,
        "database_loaded": chat_system.customer_database is not None,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def _create_structured_response(self, query: str, context: Dict[str, Any]) -> str:
    """Crea una respuesta estructurada cuando el LLM no está disponible"""
    query_lower = query.lower()
    
    if "estadística" in query_lower or "tasa" in query_lower:
        stats = context.get("statistics", {})
        return f"""📊 Estadísticas de Churn:
        
• Total de clientes: {stats.get('total_customers', 'N/A'):,}
• Tasa de churn: {stats.get('churn_rate', 0)*100:.1f}%
• Clientes perdidos/mes: {stats.get('monthly_churned', 2500):,}
• Balance promedio: ${stats.get('avg_balance', 0):,.2f}

💡 Con una tasa de churn del 25% anual, estás perdiendo aproximadamente 2,500 clientes al mes. El costo de retención es 1/5 del costo de adquisición, por lo que invertir en retención predictiva es altamente rentable."""
    
    elif "riesgo" in query_lower or "clientes" in query_lower:
        at_risk = context.get("at_risk_customers", [])
        if at_risk:
            top_3 = at_risk[:3]
            response = f"🚨 Clientes en Alto Riesgo de Churn:\n\n"
            for i, customer in enumerate(top_3, 1):
                response += f"{i}. Cliente #{customer['customer_id']}\n"
                response += f"   • Probabilidad de churn: {customer['churn_probability']*100:.1f}%\n"
                response += f"   • Balance: ${customer['balance']:,.2f}\n"
                response += f"   • Nivel de riesgo: {customer['risk_level']}\n\n"
            response += f"📈 Total identificados: {len(at_risk)} clientes en riesgo\n"
            response += f"💰 Recomendación: Priorizar retención de clientes con balance >$100k"
            return response
        else:
            return "No se encontraron clientes en riesgo en este momento."
    
    return "Entiendo tu consulta. ¿Podrías ser más específico sobre qué aspecto del churn te gustaría analizar?"

ChurnChatSystem._create_structured_response = _create_structured_response

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 Iniciando servidor FastAPI")
    print("="*70)
    print("\n💡 Ejemplos de consultas en /chat:")
    print("   • ¿Cuántos clientes están en riesgo?")
    print("   • ¿Cuál es la tasa de churn actual?")
    print("   • Muéstrame los 10 clientes con mayor riesgo")
    print("   • ¿Qué clientes de alto valor debo priorizar?")
    print("   • ¿Cuál es el impacto económico del churn?")
    print("\n" + "="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)