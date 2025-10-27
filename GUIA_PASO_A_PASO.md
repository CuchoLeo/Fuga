# 🚀 GUÍA PASO A PASO - Docker para Churn Prediction System

## PASO 1: Verificar Requisitos Previos

### 1.1 Verificar Docker
```bash
docker --version
# Debe mostrar: Docker version 20.10.x o superior
```

### 1.2 Verificar Docker Compose
```bash
docker-compose --version
# Debe mostrar: Docker Compose version 2.x.x o superior
```

### 1.3 Verificar Dataset
```bash
cd /home/user/Fuga
ls -lh Churn_Modelling.csv
# Debe mostrar el archivo CSV con ~600KB
```

**Si Docker no está instalado:**
- Linux: `sudo apt-get update && sudo apt-get install docker.io docker-compose`
- Mac/Windows: Instala Docker Desktop desde https://www.docker.com/products/docker-desktop

---

## PASO 2: Preparar el Entorno

### 2.1 Navegar al directorio del proyecto
```bash
cd /home/user/Fuga
```

### 2.2 Verificar archivos necesarios
```bash
ls -la
# Debes ver:
# - Dockerfile
# - docker-compose.yml
# - requirements.txt
# - .dockerignore
# - churn_chat_api.py
# - train_churn_prediction.py
# - Churn_Modelling.csv
```

### 2.3 (Opcional) Limpiar contenedores previos
```bash
docker-compose down
docker system prune -f
```

---

## PASO 3: Construir la Imagen Docker

### 3.1 Construir la imagen
```bash
docker-compose build
```

**Tiempo estimado:** 5-10 minutos la primera vez

**Salida esperada:**
```
[+] Building 245.3s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => [1/6] FROM docker.io/library/python:3.11-slim
 => [2/6] RUN apt-get update && apt-get install...
 => [3/6] WORKDIR /app
 => [4/6] COPY requirements.txt .
 => [5/6] RUN pip install --no-cache-dir -r requirements.txt
 => [6/6] COPY *.py *.csv *.html ./
 => exporting to image
Successfully tagged fuga-churn-api:latest
```

### 3.2 Verificar que la imagen se creó
```bash
docker images | grep churn
# Debe mostrar: fuga-churn-api con tag latest
```

---

## PASO 4: Entrenar el Modelo (Primera vez)

### 4.1 Opción A: Entrenar ANTES de iniciar el contenedor (Recomendado)
```bash
# Entrenar localmente primero
python train_churn_prediction.py
```

Esto creará la carpeta `churn_model/` que será montada en el contenedor.

### 4.2 Opción B: Entrenar DENTRO del contenedor
```bash
# Iniciar contenedor temporalmente
docker-compose up -d

# Entrenar dentro del contenedor
docker-compose exec churn-api python train_churn_prediction.py

# Esperar 5-15 minutos...
```

**Salida esperada del entrenamiento:**
```
📊 Cargando dataset desde: Churn_Modelling.csv
✓ Dataset cargado: 10000 registros, 14 columnas
💰 Clientes alto valor (Balance > $100k): 1234 (12.3%)
🚀 INICIANDO ENTRENAMIENTO...
Epoch 1/3: 100%|████████████| 250/250 [02:15<00:00]
Epoch 2/3: 100%|████████████| 250/250 [02:12<00:00]
Epoch 3/3: 100%|████████████| 250/250 [02:14<00:00]
✅ Entrenamiento completado!
💾 Modelo guardado en: churn_model/
```

---

## PASO 5: Iniciar el Contenedor

### 5.1 Iniciar con Docker Compose
```bash
docker-compose up -d
```

**Explicación de flags:**
- `-d`: Modo detached (segundo plano)
- Sin `-d`: Verás los logs en tiempo real (útil para depuración)

### 5.2 Verificar que está corriendo
```bash
docker-compose ps
```

**Salida esperada:**
```
NAME                     COMMAND                  STATUS        PORTS
churn-prediction-api     "python churn_chat_a…"   Up 30 sec     0.0.0.0:8000->8000/tcp
```

### 5.3 Ver logs del contenedor
```bash
docker-compose logs -f
```

**Salida esperada:**
```
churn-api  | ======================================================================
churn-api  | 🚀 INICIANDO SISTEMA DE CHAT DE PREDICCIÓN DE CHURN
churn-api  | ======================================================================
churn-api  | 🔄 Cargando modelos...
churn-api  | 📦 Cargando modelo de predicción de churn...
churn-api  | ✅ Modelo de churn cargado
churn-api  | ⚠️  LLM no encontrado. Usando modelo base...
churn-api  | 📊 Cargando base de datos de clientes...
churn-api  | ✅ 10000 clientes cargados
churn-api  | ✅ Todos los modelos cargados correctamente
churn-api  | ======================================================================
churn-api  | ✅ Sistema listo para recibir consultas
churn-api  | ======================================================================
churn-api  | INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Presiona `Ctrl+C` para salir de los logs (el contenedor seguirá corriendo).

---

## PASO 6: Verificar que Funciona

### 6.1 Health Check
```bash
curl http://localhost:8000/health
```

**Respuesta esperada:**
```json
{
  "status": "healthy",
  "churn_model_loaded": true,
  "llm_loaded": true,
  "database_loaded": true,
  "timestamp": "2025-10-27T10:30:00.123456"
}
```

### 6.2 Información de la API
```bash
curl http://localhost:8000/
```

**Respuesta esperada:**
```json
{
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
```

### 6.3 Obtener Estadísticas
```bash
curl http://localhost:8000/statistics
```

### 6.4 Probar el Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cuántos clientes están en riesgo?",
    "conversation_history": []
  }'
```

---

## PASO 7: Usar la Documentación Interactiva

### 7.1 Swagger UI
Abre en tu navegador:
```
http://localhost:8000/docs
```

Aquí puedes:
- Ver todos los endpoints disponibles
- Probar la API directamente desde el navegador
- Ver ejemplos de request/response
- Ver los modelos de datos

### 7.2 ReDoc (Documentación alternativa)
```
http://localhost:8000/redoc
```

Documentación más detallada y legible.

---

## PASO 8: Usar la Interfaz Web (HTML)

### 8.1 Opción A: Abrir directamente
```bash
# En tu navegador, abre:
file:///home/user/Fuga/chat_interface.html
```

### 8.2 Opción B: Con servidor HTTP
```bash
# En otra terminal
python3 -m http.server 8080

# Luego abre en el navegador:
http://localhost:8080/chat_interface.html
```

---

## PASO 9: Pruebas Automatizadas

### 9.1 Ejecutar script de pruebas
```bash
python test_churn_api.py
```

### 9.2 Ejecutar pruebas dentro del contenedor
```bash
docker-compose exec churn-api python test_churn_api.py
```

---

## PASO 10: Ejemplos de Uso Real

### 10.1 Obtener clientes en riesgo
```bash
curl "http://localhost:8000/at-risk?limit=5&high_value_only=true"
```

### 10.2 Predecir churn para un cliente específico
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 10.3 Chat en lenguaje natural
```bash
# Ejemplo 1: Estadísticas
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuál es la tasa de churn actual?"}'

# Ejemplo 2: Análisis
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Dame los 10 clientes con mayor riesgo"}'

# Ejemplo 3: Estratégico
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuál es el impacto económico del churn?"}'
```

---

## PASO 11: Detener y Reiniciar

### 11.1 Detener el contenedor (mantiene datos)
```bash
docker-compose stop
```

### 11.2 Reiniciar el contenedor
```bash
docker-compose start
```

### 11.3 Detener y eliminar el contenedor
```bash
docker-compose down
```

### 11.4 Reiniciar completamente (reconstruir)
```bash
docker-compose down
docker-compose up --build -d
```

---

## PASO 12: Monitoreo y Debugging

### 12.1 Ver logs en tiempo real
```bash
docker-compose logs -f churn-api
```

### 12.2 Ver últimos 100 logs
```bash
docker-compose logs --tail=100 churn-api
```

### 12.3 Entrar al contenedor (bash)
```bash
docker-compose exec churn-api bash
```

Dentro del contenedor puedes:
```bash
# Ver archivos
ls -la

# Ver modelos
ls -la churn_model/

# Ver procesos
ps aux

# Salir
exit
```

### 12.4 Ver uso de recursos
```bash
docker stats churn-prediction-api
```

**Salida:**
```
CONTAINER ID   NAME                   CPU %   MEM USAGE / LIMIT   MEM %
a1b2c3d4e5f6   churn-prediction-api   5.23%   1.2GiB / 4GiB      30.00%
```

---

## PASO 13: Limpieza

### 13.1 Detener y eliminar todo
```bash
docker-compose down -v
```

### 13.2 Eliminar imagen
```bash
docker rmi fuga-churn-api
```

### 13.3 Limpiar sistema Docker completo
```bash
docker system prune -a --volumes
```

**⚠️ Advertencia:** Esto eliminará TODAS las imágenes, contenedores y volúmenes no usados.

---

## 🐛 Solución de Problemas Comunes

### Problema 1: Puerto 8000 ya en uso
```bash
# Ver qué proceso usa el puerto
lsof -i :8000

# Matar el proceso (reemplaza PID)
kill -9 <PID>

# O cambiar puerto en docker-compose.yml:
ports:
  - "8001:8000"  # Cambia 8000 a 8001
```

### Problema 2: Contenedor se detiene inmediatamente
```bash
# Ver logs de error
docker-compose logs

# Causas comunes:
# - Modelo no entrenado → Entrenar primero
# - Dataset no encontrado → Verificar Churn_Modelling.csv
# - Error en código → Ver logs para detalles
```

### Problema 3: Error "No module named X"
```bash
# Reconstruir sin caché
docker-compose build --no-cache
```

### Problema 4: Modelo muy lento
```bash
# Usar GPU (si disponible)
# Editar Dockerfile, cambiar:
# torch>=2.0.0
# a:
# torch>=2.0.0+cu118
```

### Problema 5: Memoria insuficiente
```bash
# Aumentar memoria en Docker Desktop (Mac/Windows)
# O limitar memoria:
docker run -m 4g ...
```

---

## 📊 Checklist de Verificación

Marca cada paso cuando lo completes:

- [ ] **PASO 1:** Docker y Docker Compose instalados
- [ ] **PASO 2:** Dataset Churn_Modelling.csv en el directorio
- [ ] **PASO 3:** Imagen Docker construida exitosamente
- [ ] **PASO 4:** Modelo de churn entrenado
- [ ] **PASO 5:** Contenedor iniciado y corriendo
- [ ] **PASO 6:** Health check responde OK
- [ ] **PASO 7:** Swagger docs accesible en /docs
- [ ] **PASO 8:** Interfaz web funciona
- [ ] **PASO 9:** Pruebas automatizadas pasan
- [ ] **PASO 10:** Endpoints funcionan correctamente

---

## 🚀 Comandos Rápidos de Referencia

```bash
# Construir y ejecutar (primera vez)
docker-compose up --build -d

# Ver logs
docker-compose logs -f

# Verificar salud
curl http://localhost:8000/health

# Detener
docker-compose stop

# Reiniciar
docker-compose restart

# Eliminar todo
docker-compose down -v

# Entrar al contenedor
docker-compose exec churn-api bash

# Ver estadísticas
curl http://localhost:8000/statistics

# Probar chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuántos clientes están en riesgo?"}'
```

---

## 📚 Próximos Pasos

1. **Explorar la API:** Usa Swagger UI en http://localhost:8000/docs
2. **Probar diferentes consultas:** Experimenta con el endpoint /chat
3. **Integrar con tus sistemas:** Usa la API desde tus aplicaciones
4. **Monitorear rendimiento:** Usa docker stats para ver recursos
5. **Escalar:** Considera usar Kubernetes para producción

---

## 🆘 Ayuda Adicional

- **Documentación completa:** Ver README.md y README_DOCKER.md
- **API Docs:** http://localhost:8000/docs
- **Logs:** `docker-compose logs -f`
- **Estado:** `docker-compose ps`

**¡Listo! Tu sistema de predicción de churn está funcionando en Docker! 🎉**
