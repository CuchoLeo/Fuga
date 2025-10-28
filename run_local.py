#!/usr/bin/env python3
"""
Script para ejecutar el servidor de Churnito localmente (sin Docker)

Uso:
    python run_local.py

Requisitos:
    - Python 3.11+
    - pip install -r requirements.txt
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Verifica que los archivos necesarios existan"""
    required_files = [
        "churn_chat_api.py",
        "chat_interface.html",
        "Churn_Modelling.csv",
        "requirements.txt"
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ ERROR: Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Asegúrate de estar en el directorio correcto (Fuga/)")
        sys.exit(1)

    print("✅ Todos los archivos necesarios encontrados")

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        import fastapi
        import uvicorn
        import torch
        import transformers
        import pandas
        import sklearn
        print("✅ Dependencias principales instaladas")
    except ImportError as e:
        print(f"❌ ERROR: Falta instalar dependencias")
        print(f"   {e}")
        print("\n💡 Ejecuta: pip install -r requirements.txt")
        sys.exit(1)

def main():
    print("="*70)
    print("🤖 CHURNITO - Servidor Local")
    print("="*70)

    # Verificar archivos
    check_requirements()

    # Verificar dependencias
    check_dependencies()

    # Crear directorios si no existen
    os.makedirs("churn_model", exist_ok=True)
    os.makedirs("trained_model", exist_ok=True)
    print("✅ Directorios creados/verificados")

    print("\n" + "="*70)
    print("🚀 INICIANDO SERVIDOR...")
    print("="*70)
    print("📍 URL: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("💬 Chat: http://localhost:8000")
    print("\n⏹️  Para detener: Presiona Ctrl+C")
    print("="*70)
    print()

    # Importar y ejecutar uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "churn_chat_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Auto-reload en cambios de código
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("👋 Servidor detenido")
        print("="*70)
    except Exception as e:
        print(f"\n❌ ERROR al iniciar servidor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
