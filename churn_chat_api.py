from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json
from datetime import datetime

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
        
        # 2. Cargar modelo LLM para conversación
        try:
            llm_model_path = Path("trained_model")
            if llm_model_path.exists():
                print("🤖 Cargando LLM para conversación...")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_path,
                    torch_dtype=torch.float32
                )
                self.llm_model.eval()
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                print("✅ LLM cargado")
            else:
                print("⚠️  LLM no encontrado. Intentando descargar modelo base...")
                model_id = "meta-llama/Llama-3.2-1B-Instruct"
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32
                )
                self.llm_model.eval()
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                print("✅ LLM base descargado y cargado")
        except Exception as e:
            print(f"⚠️  Error al cargar LLM: {e}")
            print("⚠️  La API funcionará sin capacidades de chat LLM avanzadas")
            print("⚠️  Solo respuestas estructuradas estarán disponibles")
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
        if any(word in query_lower for word in ["cuántos", "cantidad", "lista", "clientes en riesgo", "top", "dame", "muestra", "quiero ver"]):
            intent["type"] = "analysis"
            intent["requires_analysis"] = True

        if any(word in query_lower for word in ["tasa", "porcentaje", "estadística", "métrica", "promedio"]):
            intent["requires_statistics"] = True

        if any(word in query_lower for word in ["predice", "predicción", "probabilidad"]):
            intent["requires_prediction"] = True

        if any(word in query_lower for word in ["alto valor", "premium", "balance alto", "mayor"]):
            intent["high_value"] = True
        
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
        """Genera respuesta usando el LLM con contexto"""
        if self.llm_model is None:
            return "Lo siento, el modelo de lenguaje no está disponible."
        
        # Construir prompt con contexto
        prompt = self._build_prompt(query, context)
        
        # Generar respuesta
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.llm_tokenizer.pad_token_id
            )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpiar el prompt de la respuesta
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Construye un prompt informado con contexto"""
        prompt_parts = [
            "Eres un asistente experto en análisis de churn (fuga de clientes).",
            "Tu empresa tiene un problema serio:",
            "- Tasa de churn anual: 25%",
            "- Clientes perdidos/mes: 2,500",
            "- Enfoque: clientes de alto valor (Balance > $100,000)",
            "- Costo de retención = 1/5 del costo de adquisición",
            "\nInformación actual:"
        ]
        
        if "statistics" in context:
            stats = context["statistics"]
            prompt_parts.append(f"- Total clientes: {stats.get('total_customers', 'N/A')}")
            prompt_parts.append(f"- Tasa de churn: {stats.get('churn_rate', 0)*100:.1f}%")
            if "monthly_churned" in stats:
                prompt_parts.append(f"- Clientes perdidos/mes: {stats['monthly_churned']}")
        
        if "at_risk_customers" in context:
            at_risk = context["at_risk_customers"]
            prompt_parts.append(f"\n- Clientes en riesgo identificados: {len(at_risk)}")
            if at_risk:
                prompt_parts.append(f"- Mayor riesgo: {at_risk[0].get('churn_probability', 0)*100:.1f}% de probabilidad")
        
        if "prediction" in context:
            pred = context["prediction"]
            prompt_parts.append(f"\nPredicción realizada:")
            prompt_parts.append(f"- Probabilidad de churn: {pred.get('churn_probability', 0)*100:.1f}%")
            prompt_parts.append(f"- Nivel de riesgo: {pred.get('risk_level', 'N/A')}")
        
        prompt_parts.append(f"\nPregunta del usuario: {query}")
        prompt_parts.append("\nRespuesta concisa y accionable:")
        
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
        
        # Generar respuesta con LLM
        response_text = chat_system.generate_llm_response(query, context)
        
        # Si la respuesta del LLM es muy corta o no informativa, crear una respuesta estructurada
        if len(response_text) < 20 or "no está disponible" in response_text.lower():
            response_text = chat_system._create_structured_response(query, context)
        
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