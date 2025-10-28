#!/bin/bash

# ============================================
# Script de Configuración Docker - Churn Prediction
# Ejecuta este script en tu Mac (en la carpeta Fuga)
# ============================================

set -e  # Detener en caso de error

echo "============================================"
echo "🚀 SETUP DOCKER - CHURN PREDICTION SYSTEM"
echo "============================================"
echo ""

# ============================================
# PASO 1: Verificar Docker
# ============================================
echo "📋 PASO 1: Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado"
    echo "   Instálalo desde: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "✅ Docker instalado: $(docker --version)"
echo ""

if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  docker-compose no encontrado, intentando con 'docker compose'"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi
echo "✅ Docker Compose disponible"
echo ""

# ============================================
# PASO 2: Verificar archivos necesarios
# ============================================
echo "📋 PASO 2: Verificando archivos necesarios..."
required_files=("Dockerfile" "docker-compose.yml" "requirements.txt" "churn_chat_api.py" "Churn_Modelling.csv")

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Archivo no encontrado: $file"
        exit 1
    fi
    echo "✅ $file"
done
echo ""

# ============================================
# PASO 3: Limpiar contenedores previos (opcional)
# ============================================
echo "📋 PASO 3: Limpiando contenedores previos (si existen)..."
$DOCKER_COMPOSE down 2>/dev/null || true
echo "✅ Limpieza completada"
echo ""

# ============================================
# PASO 4: Construir imagen Docker
# ============================================
echo "📋 PASO 4: Construyendo imagen Docker..."
echo "⏳ Esto puede tardar 5-10 minutos la primera vez..."
echo ""
$DOCKER_COMPOSE build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Imagen construida exitosamente"
else
    echo ""
    echo "❌ Error al construir la imagen"
    exit 1
fi
echo ""

# ============================================
# PASO 5: Iniciar contenedor
# ============================================
echo "📋 PASO 5: Iniciando contenedor..."
$DOCKER_COMPOSE up -d

if [ $? -eq 0 ]; then
    echo "✅ Contenedor iniciado"
else
    echo "❌ Error al iniciar contenedor"
    exit 1
fi
echo ""

# ============================================
# PASO 6: Esperar a que el servicio esté listo
# ============================================
echo "📋 PASO 6: Esperando a que el servicio esté listo..."
echo "⏳ Esto puede tardar 1-2 minutos..."
echo ""

max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Servicio listo!"
        break
    fi

    attempt=$((attempt + 1))
    echo "   Intento $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ El servicio no respondió a tiempo"
    echo "   Ver logs con: $DOCKER_COMPOSE logs"
    exit 1
fi
echo ""

# ============================================
# PASO 7: Verificar estado
# ============================================
echo "📋 PASO 7: Verificando estado del sistema..."
health_response=$(curl -s http://localhost:8000/health)
echo "$health_response" | python3 -m json.tool 2>/dev/null || echo "$health_response"
echo ""

# ============================================
# PASO 8: Entrenar modelo (si no existe)
# ============================================
echo "📋 PASO 8: Verificando modelo de churn..."
if [ ! -d "churn_model" ]; then
    echo "⚠️  Modelo no encontrado. Entrenando dentro del contenedor..."
    echo "⏳ Esto tardará 5-15 minutos..."
    echo ""
    $DOCKER_COMPOSE exec -T churn-api python train_churn_prediction.py

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Modelo entrenado exitosamente"
    else
        echo ""
        echo "❌ Error al entrenar modelo"
        echo "   Puedes entrenarlo manualmente con:"
        echo "   $DOCKER_COMPOSE exec churn-api python train_churn_prediction.py"
    fi
else
    echo "✅ Modelo ya existe"
fi
echo ""

# ============================================
# RESUMEN
# ============================================
echo "============================================"
echo "✅ CONFIGURACIÓN COMPLETADA"
echo "============================================"
echo ""
echo "🌐 API disponible en: http://localhost:8000"
echo "📚 Documentación: http://localhost:8000/docs"
echo "💓 Health check: http://localhost:8000/health"
echo ""
echo "📋 Comandos útiles:"
echo "   Ver logs:     $DOCKER_COMPOSE logs -f"
echo "   Detener:      $DOCKER_COMPOSE stop"
echo "   Reiniciar:    $DOCKER_COMPOSE restart"
echo "   Eliminar:     $DOCKER_COMPOSE down"
echo ""
echo "🧪 Prueba el sistema:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:8000/statistics"
echo ""
echo "🎉 ¡Sistema listo para usar!"
echo "============================================"
