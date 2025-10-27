#!/usr/bin/env python3
"""Script de diagnóstico para verificar predicciones"""

import requests
import json

API_URL = "http://localhost:8000"

print("="*70)
print("DIAGNÓSTICO DE PREDICCIONES")
print("="*70)

# Test 1: Cliente de ALTO RIESGO
print("\n1. Probando cliente de ALTO RIESGO...")
high_risk_customer = {
    "CreditScore": 350,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 55,
    "Tenure": 1,
    "Balance": 150000,
    "NumOfProducts": 1,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 30000
}

response = requests.post(
    f"{API_URL}/predict",
    json={"customers": [high_risk_customer]}
)

if response.status_code == 200:
    result = response.json()
    prediction = result['predictions'][0]['prediction']

    if 'error' in prediction:
        print(f"❌ ERROR: {prediction['error']}")
    else:
        prob = prediction.get('churn_probability', 0)
        print(f"✅ Probabilidad: {prob:.4f} ({prob*100:.2f}%)")
        print(f"   Nivel de riesgo: {prediction.get('risk_level', 'N/A')}")
        print(f"   ¿Churn?: {prediction.get('will_churn', False)}")
else:
    print(f"❌ Error HTTP: {response.status_code}")

# Test 2: Cliente de BAJO RIESGO
print("\n2. Probando cliente de BAJO RIESGO...")
low_risk_customer = {
    "CreditScore": 850,
    "Geography": "France",
    "Gender": "Male",
    "Age": 30,
    "Tenure": 10,
    "Balance": 50000,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 150000
}

response = requests.post(
    f"{API_URL}/predict",
    json={"customers": [low_risk_customer]}
)

if response.status_code == 200:
    result = response.json()
    prediction = result['predictions'][0]['prediction']

    if 'error' in prediction:
        print(f"❌ ERROR: {prediction['error']}")
    else:
        prob = prediction.get('churn_probability', 0)
        print(f"✅ Probabilidad: {prob:.4f} ({prob*100:.2f}%)")
        print(f"   Nivel de riesgo: {prediction.get('risk_level', 'N/A')}")
        print(f"   ¿Churn?: {prediction.get('will_churn', False)}")
else:
    print(f"❌ Error HTTP: {response.status_code}")

# Test 3: Clientes en riesgo
print("\n3. Buscando clientes en riesgo...")
response = requests.get(f"{API_URL}/at-risk?limit=20")

if response.status_code == 200:
    result = response.json()
    total = result.get('total_at_risk', 0)
    print(f"✅ Total en riesgo: {total}")

    if total > 0:
        print("\n   Top 5 clientes:")
        for i, customer in enumerate(result['customers'][:5], 1):
            print(f"   {i}. Cliente #{customer['customer_id']}")
            print(f"      Probabilidad: {customer['churn_probability']*100:.2f}%")
            print(f"      Balance: ${customer['balance']:,.2f}")
            print(f"      Riesgo: {customer['risk_level']}")
    else:
        print("   ⚠️  No se encontraron clientes en riesgo")
else:
    print(f"❌ Error HTTP: {response.status_code}")

# Test 4: Chat
print("\n4. Probando chat...")
response = requests.post(
    f"{API_URL}/chat",
    json={
        "message": "¿Cuántos clientes están en riesgo?",
        "conversation_history": []
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Respuesta del chat:")
    print(f"   {result['response']}")
else:
    print(f"❌ Error HTTP: {response.status_code}")

print("\n" + "="*70)
print("DIAGNÓSTICO COMPLETADO")
print("="*70)
