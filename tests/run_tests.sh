#!/bin/bash

# ============================================================================
# SCRIPT DE EJECUCIÓN DE PRUEBAS COMPLETAS
# Sistema de Predicción de Churn
# ============================================================================

echo "========================================================================"
echo "🧪 SISTEMA DE PRUEBAS COMPLETAS - MODELO DE PREDICCIÓN DE CHURN"
echo "========================================================================"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 no está instalado"
    exit 1
fi

echo "✓ Python3 encontrado: $(python3 --version)"
echo ""

# Verificar modelo entrenado
if [ ! -d "churn_model" ]; then
    echo "⚠️  No se encontró el modelo entrenado en 'churn_model/'"
    echo "📝 Entrenando modelo primero..."
    python3 train_churn_prediction.py

    if [ $? -ne 0 ]; then
        echo "❌ Error al entrenar el modelo"
        exit 1
    fi
    echo ""
fi

# Ejecutar pruebas
echo "========================================================================"
echo "📊 Paso 1: Ejecutando pruebas y generando métricas..."
echo "========================================================================"
python3 tests/test_models.py

if [ $? -ne 0 ]; then
    echo "❌ Error al ejecutar las pruebas"
    exit 1
fi

echo ""
echo "========================================================================"
echo "📄 Paso 2: Generando reporte HTML..."
echo "========================================================================"
python3 tests/generate_report.py

if [ $? -ne 0 ]; then
    echo "❌ Error al generar el reporte"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ PRUEBAS COMPLETADAS EXITOSAMENTE"
echo "========================================================================"
echo ""
echo "📁 Resultados disponibles en: test_results/"
echo ""
echo "📄 Archivos generados:"
ls -lh test_results/ | tail -n +2 | awk '{print "   - " $9 " (" $5 ")"}'
echo ""
echo "💡 Para ver el reporte HTML:"
echo "   open test_results/informe_completo.html  # macOS"
echo "   xdg-open test_results/informe_completo.html  # Linux"
echo "   start test_results/informe_completo.html  # Windows"
echo ""
echo "========================================================================"
