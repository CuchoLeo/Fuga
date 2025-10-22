import requests
import json
from typing import Dict, Any
import time

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

API_BASE_URL = "http://localhost:8000"

class ChurnAPITester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        
    def print_section(self, title: str):
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def print_response(self, response: Dict[Any, Any]):
        print(json.dumps(response, indent=2, ensure_ascii=False))
    
    # ========================================================================
    # 1. TEST DE CONEXIÓN
    # ========================================================================
    
    def test_health(self):
        """Verificar que la API esté funcionando"""
        self.print_section("🏥 TEST: Health Check")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Estado: {data['status']}")
            print(f"✅ Modelo de churn cargado: {data['churn_model_loaded']}")
            print(f"✅ LLM cargado: {data['llm_loaded']}")
            print(f"✅ Base de datos cargada: {data['database_loaded']}")
            
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # ========================================================================
    # 2. TEST DE CHAT
    # ========================================================================
    
    def test_chat(self, message: str):
        """Enviar mensaje al chat"""
        self.print_section(f"💬 TEST: Chat - '{message}'")
        
        try:
            payload = {
                "message": message,
                "conversation_history": []
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"\n📝 Respuesta:")
            print(data['response'])
            
            if data.get('data'):
                print(f"\n📊 Datos adicionales:")
                self.print_response(data['data'])
            
            return data
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    # ========================================================================
    # 3. TEST DE PREDICCIÓN
    # ========================================================================
    
    def test_prediction(self):
        """Predecir churn para clientes de ejemplo"""
        self.print_section("🔮 TEST: Predicción de Churn")
        
        # Clientes de ejemplo
        customers = [
            {
                "CreditScore": 600,
                "Geography": "Spain",
                "Gender": "Male",
                "Age": 45,
                "Tenure": 2,
                "Balance": 150000,  # Alto valor
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 0,  # Inactivo - señal de riesgo
                "EstimatedSalary": 80000
            },
            {
                "CreditScore": 750,
                "Geography": "France",
                "Gender": "Female",
                "Age": 35,
                "Tenure": 8,
                "Balance": 200000,  # Alto valor
                "NumOfProducts": 3,
                "HasCrCard": 1,
                "IsActiveMember": 1,  # Activo - menor riesgo
                "EstimatedSalary": 120000
            },
            {
                "CreditScore": 500,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 55,
                "Tenure": 1,
                "Balance": 50000,  # Valor medio
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 50000
            }
        ]
        
        try:
            payload = {"customers": customers}
            
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            print(f"\n📊 Resumen:")
            print(f"• Total analizado: {data['summary']['total_analyzed']}")
            print(f"• Alto riesgo: {data['summary']['high_risk']}")
            print(f"• Probabilidad promedio: {data['summary']['average_churn_probability']*100:.1f}%")
            print(f"• Recomendación: {data['summary']['recommendation']}")
            
            print(f"\n🎯 Predicciones individuales:")
            for i, pred in enumerate(data['predictions'], 1):
                customer = pred['customer_data']
                prediction = pred['prediction']
                
                print(f"\n--- Cliente {i} ---")
                print(f"Balance: ${customer['Balance']:,.2f}")
                print(f"Edad: {customer['Age']} años")
                print(f"Activo: {'Sí' if customer['IsActiveMember'] else 'No'}")
                print(f"➡️  Probabilidad de churn: {prediction['churn_probability']*100:.1f}%")
                print(f"➡️  Nivel de riesgo: {prediction['risk_level']}")
                print(f"➡️  Prioridad de retención: {prediction['retention_priority']}")
            
            return data
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    # ========================================================================
    # 4. TEST DE ESTADÍSTICAS
    # ========================================================================
    
    def test_statistics(self, high_value_only: bool = False):
        """Obtener estadísticas generales"""
        filter_text = " (Alto Valor)" if high_value_only else ""
        self.print_section(f"📊 TEST: Estadísticas{filter_text}")
        
        try:
            params = {"high_value_only": high_value_only}
            response = requests.get(
                f"{self.base_url}/statistics",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            
            print(f"\n📈 Métricas:")
            print(f"• Total de clientes: {data.get('total_customers', 'N/A'):,}")
            print(f"• Clientes que se fueron: {data.get('churned_customers', 'N/A'):,}")
            print(f"• Tasa de churn: {data.get('churn_rate', 0)*100:.2f}%")
            print(f"• Balance promedio: ${data.get('avg_balance', 0):,.2f}")
            print(f"• Edad promedio: {data.get('avg_age', 0):.1f} años")
            
            if 'monthly_churned' in data:
                print(f"• Pérdidas mensuales: {data['monthly_churned']:,} clientes")
            
            if 'business_impact' in data:
                impact = data['business_impact']
                print(f"\n💼 Impacto de Negocio:")
                print(f"• Pérdida mensual de clientes: {impact['monthly_customer_loss']:,}")
                print(f"• Tasa anual de churn: {impact['annual_churn_rate']*100:.0f}%")
                print(f"• Ratio costo retención: {impact['retention_cost_ratio']*100:.0f}% del costo de adquisición")
            
            return data
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    # ========================================================================
    # 5. TEST DE CLIENTES EN RIESGO
    # ========================================================================
    
    def test_at_risk(self, limit: int = 10, high_value_only: bool = False):
        """Obtener clientes en riesgo"""
        filter_text = " (Alto Valor)" if high_value_only else ""
        self.print_section(f"🚨 TEST: Clientes en Riesgo{filter_text}")
        
        try:
            params = {
                "limit": limit,
                "high_value_only": high_value_only
            }
            response = requests.get(
                f"{self.base_url}/at-risk",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            
            print(f"\n⚠️  Total en riesgo: {data['total_at_risk']}")
            print(f"📋 Recomendación: {data['recommendation']}")
            
            if data['customers']:
                print(f"\n🎯 Top {len(data['customers'])} clientes en riesgo:")
                for i, customer in enumerate(data['customers'][:10], 1):
                    print(f"\n{i}. Cliente #{customer['customer_id']}")
                    print(f"   💰 Balance: ${customer['balance']:,.2f}")
                    print(f"   📊 Probabilidad de churn: {customer['churn_probability']*100:.1f}%")
                    print(f"   ⚡ Nivel de riesgo: {customer['risk_level']}")
                    print(f"   👤 Edad: {customer['age']} años | Antigüedad: {customer['tenure']} años")
                    print(f"   {'✅' if customer['is_active'] else '❌'} {'Activo' if customer['is_active'] else 'Inactivo'}")
            
            print(f"\n💡 Estrategias de retención sugeridas:")
            for strategy in data['retention_strategies']:
                print(f"   • {strategy}")
            
            return data
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    # ========================================================================
    # 6. TESTS DE CHAT ESPECÍFICOS
    # ========================================================================
    
    def test_chat_scenarios(self):
        """Probar diferentes escenarios de conversación"""
        self.print_section("🎭 TEST: Escenarios de Conversación")
        
        questions = [
            "¿Cuántos clientes están en riesgo?",
            "¿Cuál es la tasa de churn actual?",
            "Muéstrame los 10 clientes con mayor riesgo de fuga",
            "¿Qué clientes de alto valor debo priorizar?",
            "¿Cuál es el impacto económico del churn?",
            "¿Cómo puedo reducir la fuga de clientes?"
        ]
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n--- Pregunta {i}/{len(questions)} ---")
            print(f"❓ {question}")
            
            result = self.test_chat(question)
            results.append(result)
            
            # Pequeña pausa entre requests
            time.sleep(1)
        
        return results
    
    # ========================================================================
    # 7. TEST COMPLETO
    # ========================================================================
    
    def run_all_tests(self):
        """Ejecutar todos los tests"""
        print("\n" + "🚀"*35)
        print("  INICIANDO SUITE COMPLETA DE TESTS")
        print("🚀"*35)
        
        # 1. Health check
        if not self.test_health():
            print("\n❌ La API no está disponible. Asegúrate de ejecutar:")
            print("   python churn_chat_api.py")
            return
        
        time.sleep(1)
        
        # 2. Estadísticas generales
        self.test_statistics(high_value_only=False)
        time.sleep(1)
        
        # 3. Estadísticas de alto valor
        self.test_statistics(high_value_only=True)
        time.sleep(1)
        
        # 4. Predicciones
        self.test_prediction()
        time.sleep(1)
        
        # 5. Clientes en riesgo
        self.test_at_risk(limit=5, high_value_only=False)
        time.sleep(1)
        
        # 6. Clientes de alto valor en riesgo
        self.test_at_risk(limit=5, high_value_only=True)
        time.sleep(1)
        
        # 7. Tests de conversación
        self.test_chat_scenarios()
        
        self.print_section("✅ TESTS COMPLETADOS")
        print("\n💡 La API está funcionando correctamente!")
        print("   Puedes abrir chat_interface.html en tu navegador")
        print("   o continuar usando esta API programáticamente.\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║      🎯 SISTEMA DE PREDICCIÓN DE CHURN - TEST SUITE                 ║
    ║                                                                      ║
    ║      Suite de pruebas para validar la API de chat                   ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚙️  Configuración:")
    print(f"   API URL: {API_BASE_URL}")
    print("\n📝 Nota: Asegúrate de que la API esté corriendo:")
    print("   python churn_chat_api.py")
    print("\nPresiona Enter para continuar o Ctrl+C para cancelar...")
    input()
    
    # Crear instancia del tester
    tester = ChurnAPITester()
    
    # Ejecutar todos los tests
    tester.run_all_tests()
    
    print("\n" + "="*70)
    print("OPCIONES ADICIONALES:")
    print("="*70)
    print("""
    Para probar funcionalidades específicas, puedes usar:
    
    >>> tester = ChurnAPITester()
    >>> tester.test_health()                    # Verificar conexión
    >>> tester.test_statistics()                # Ver estadísticas
    >>> tester.test_at_risk(limit=20)          # Ver top 20 en riesgo
    >>> tester.test_chat("Tu pregunta aquí")   # Enviar mensaje
    >>> tester.test_prediction()                # Predecir clientes
    """)
    
    print("\n💡 Para usar la interfaz web:")
    print("   Abre 'chat_interface.html' en tu navegador\n")