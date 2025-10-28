# 🚀 Guía Completa de Instalación - Sistema de Predicción de Churn

## 📋 Requisitos Previos

**Lo único que necesitas:**
- ✅ Docker Desktop instalado y corriendo
- ✅ 8GB de RAM disponible
- ✅ 5GB de espacio en disco

**Verificar que Docker funciona:**
```bash
docker --version
docker ps
```

Si ves la versión de Docker y una tabla (aunque esté vacía), ✅ estás listo.

---

## 🎯 PASO A PASO COMPLETO

### **PASO 1: Clonar el Repositorio**

```bash
# Clonar el proyecto
git clone https://github.com/CuchoLeo/Fuga.git

# Entrar al directorio
cd Fuga

# Cambiar al branch correcto
git checkout claude/create-docker-image-011CUWiCdkyttEZPktomfqF1
```

**Verificar archivos:**
```bash
ls -la
# Debes ver: Dockerfile, docker-compose.yml, requirements.txt, etc.
```

---

### **PASO 2: Descargar el Dataset**

El sistema necesita el archivo `Churn_Modelling.csv` para funcionar.

**Opción A: Descarga Manual (Recomendada)**

1. Ve a: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
2. Haz click en "Download" (necesitas cuenta de Kaggle - es gratis)
3. Descomprime el archivo `churn-modelling.zip`
4. Copia `Churn_Modelling.csv` a la carpeta `Fuga/`

**Opción B: Con Kaggle CLI**

Si tienes Kaggle CLI configurado:
```bash
kaggle datasets download -d shrutimechlearn/churn-modelling
unzip churn-modelling.zip
```

**Verificar que está el archivo:**
```bash
ls -lh Churn_Modelling.csv
# Debe mostrar: ~670KB
```

---

### **PASO 3: Construir la Imagen Docker**

```bash
# Construir la imagen (tarda 5-10 minutos la primera vez)
docker-compose build
```

**Salida esperada:**
```
[+] Building 245.3s (18/18) FINISHED
 => exporting to image
Successfully tagged churn-prediction-api:latest
```

**Verificar que se creó la imagen:**
```bash
docker images | grep churn
# Debe mostrar: churn-prediction-api  latest
```

---

### **PASO 4: Entrenar el Modelo de Churn**

⚠️ **IMPORTANTE:** Debes entrenar el modelo ANTES de iniciar la aplicación.

```bash
# Entrenar el modelo (tarda 5-15 minutos)
docker-compose run --rm churn-api python train_churn_prediction.py
```

**Salida esperada:**
```
======================================================================
SISTEMA DE PREDICCIÓN DE CHURN - CLIENTES ALTO VALOR
======================================================================
📊 Cargando dataset desde: Churn_Modelling.csv
✓ Dataset cargado: 10000 registros, 14 columnas

💰 Clientes alto valor (Balance > $100k): 1234 (12.3%)
🚀 INICIANDO ENTRENAMIENTO...

Epoch 1/3: 100%|████████████| 250/250 [02:15<00:00]
Epoch 2/3: 100%|████████████| 250/250 [02:12<00:00]
Epoch 3/3: 100%|████████████| 250/250 [02:14<00:00]

✅ Entrenamiento completado!
💾 Modelo guardado en: churn_model/

📊 Métricas de evaluación:
- Accuracy: 0.85
- Precision: 0.82
- Recall: 0.78
- F1-Score: 0.80
```

**Verificar que se creó el modelo:**
```bash
ls -lh churn_model/
# Debe mostrar varios archivos incluyendo model.safetensors (~255MB)
```

---

### **PASO 5: Iniciar la Aplicación**

```bash
# Iniciar el contenedor en segundo plano
docker-compose up -d
```

**Salida esperada:**
```
[+] Running 2/2
 ✔ Network churn-network           Created
 ✔ Container churn-prediction-api  Started
```

**Verificar que está corriendo:**
```bash
docker-compose ps
```

Debe mostrar:
```
NAME                   STATUS
churn-prediction-api   Up XX seconds (healthy)
```

---

### **PASO 6: Verificar que Funciona**

```bash
# Health check
curl http://localhost:8000/health
```

**Respuesta esperada:**
```json
{
  "status": "healthy",
  "churn_model_loaded": true,
  "llm_loaded": false,
  "database_loaded": true,
  "timestamp": "2025-10-27T..."
}
```

✅ Si ves `"status": "healthy"` y `"churn_model_loaded": true`, ¡todo funciona!

---

### **PASO 7: Ver los Logs (Opcional)**

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Presiona Ctrl+C para salir (el contenedor sigue corriendo)
```

**Debes ver:**
```
churn-api | ✅ Modelo de churn cargado
churn-api | ✅ 10000 clientes cargados
churn-api | ✅ Sistema listo para recibir consultas
churn-api | INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 🌐 USAR LA APLICACIÓN

### **Opción 1: Swagger UI (Más Fácil)** ⭐

Abre tu navegador en:
```
http://localhost:8000/docs
```

Aquí puedes:
- ✅ Ver todos los endpoints disponibles
- ✅ Probar la API directamente desde el navegador
- ✅ Ver ejemplos de request y response
- ✅ No necesitas instalar nada más

**Endpoints principales:**
- `/health` - Verificar estado del sistema
- `/statistics` - Estadísticas generales de churn
- `/at-risk` - Obtener clientes en riesgo
- `/predict` - Predecir churn para clientes específicos
- `/chat` - Chat en lenguaje natural

---

### **Opción 2: Comandos curl**

#### **1. Obtener Estadísticas**
```bash
curl http://localhost:8000/statistics | python3 -m json.tool
```

#### **2. Obtener Clientes en Riesgo**
```bash
curl "http://localhost:8000/at-risk?limit=10" | python3 -m json.tool
```

#### **3. Predecir Churn para un Cliente**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
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
    }]
  }' | python3 -m json.tool
```

#### **4. Chat en Lenguaje Natural**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cuántos clientes están en riesgo?",
    "conversation_history": []
  }' | python3 -m json.tool
```

**Ejemplos de preguntas para el chat:**
- "¿Cuántos clientes están en riesgo?"
- "Dame los 10 clientes con mayor riesgo"
- "¿Cuál es la tasa de churn actual?"
- "¿Qué clientes de alto valor debo priorizar?"
- "Muéstrame estadísticas del churn"

---

### **Opción 3: Interfaz Web HTML**

```bash
# Abrir el archivo HTML directamente en el navegador
open chat_interface.html

# O servir con un servidor HTTP
python3 -m http.server 8080
# Luego abre: http://localhost:8080/chat_interface.html
```

---

## 🛠️ COMANDOS ÚTILES

### **Gestión del Contenedor**

```bash
# Ver estado
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f

# Detener (mantiene datos)
docker-compose stop

# Iniciar de nuevo
docker-compose start

# Reiniciar
docker-compose restart

# Detener y eliminar
docker-compose down

# Ver uso de recursos
docker stats churn-prediction-api
```

### **Entrar al Contenedor (para debugging)**

```bash
# Abrir bash dentro del contenedor
docker-compose exec churn-api bash

# Dentro puedes ejecutar:
python3
ls -la
cat churn_chat_api.py

# Salir
exit
```

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### **Problema 1: "Cannot connect to Docker daemon"**

**Causa:** Docker Desktop no está corriendo.

**Solución:**
```bash
# En Mac
open -a Docker

# Espera 1-2 minutos hasta que aparezca el ícono 🐳 en la barra superior
# Luego verifica:
docker ps
```

---

### **Problema 2: "Port 8000 already in use"**

**Causa:** Otro proceso está usando el puerto 8000.

**Solución:**
```bash
# Ver qué usa el puerto
lsof -i :8000

# Matar el proceso (reemplaza PID con el número que viste)
kill -9 <PID>

# O cambiar el puerto en docker-compose.yml:
# ports:
#   - "8001:8000"  # Cambia 8000 a 8001
```

---

### **Problema 3: "Modelo de churn no encontrado"**

**Causa:** No se entrenó el modelo antes de iniciar.

**Solución:**
```bash
# Detener el contenedor
docker-compose down

# Entrenar el modelo
docker-compose run --rm churn-api python train_churn_prediction.py

# Iniciar de nuevo
docker-compose up -d
```

---

### **Problema 4: El contenedor se detiene inmediatamente**

**Causa:** Error en la aplicación al iniciar.

**Solución:**
```bash
# Ver los logs de error
docker-compose logs --tail=50

# Busca líneas con ERROR o Exception
```

Causas comunes:
- Modelo no entrenado → Ve al Problema 3
- Dataset no encontrado → Verifica que `Churn_Modelling.csv` existe
- Puerto ocupado → Ve al Problema 2

---

### **Problema 5: "Base de datos no disponible"**

**Causa:** Falta el archivo `Churn_Modelling.csv`.

**Solución:**
```bash
# Verificar que existe
ls -lh Churn_Modelling.csv

# Si no existe, descárgalo (ver PASO 2)
```

---

### **Problema 6: Respuestas muy lentas**

**Causa:** Predicciones tardan mucho.

**Solución:** Ya optimizado en la última versión. Si sigue lento:
```bash
# Reiniciar desde cero
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 📊 VERIFICACIÓN COMPLETA

Ejecuta estos comandos para verificar que todo funciona:

```bash
echo "=== 1. Contenedor corriendo ==="
docker-compose ps

echo ""
echo "=== 2. Health check ==="
curl http://localhost:8000/health

echo ""
echo "=== 3. Estadísticas ==="
curl http://localhost:8000/statistics | python3 -m json.tool

echo ""
echo "=== 4. Clientes en riesgo ==="
curl "http://localhost:8000/at-risk?limit=5" | python3 -m json.tool

echo ""
echo "=== 5. Predicción ==="
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
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
    }]
  }' | python3 -m json.tool
```

**Si todos responden correctamente, ✅ el sistema funciona al 100%.**

---

## ✅ CHECKLIST DE INSTALACIÓN

Marca cada paso cuando lo completes:

- [ ] Docker Desktop instalado y corriendo
- [ ] Repositorio clonado
- [ ] Branch correcto (claude/create-docker-image-011CUWiCdkyttEZPktomfqF1)
- [ ] Dataset `Churn_Modelling.csv` descargado
- [ ] Imagen Docker construida (`docker-compose build`)
- [ ] Modelo entrenado (`docker-compose run --rm churn-api python train_churn_prediction.py`)
- [ ] Contenedor iniciado (`docker-compose up -d`)
- [ ] Health check responde OK
- [ ] Swagger UI accesible en http://localhost:8000/docs
- [ ] Al menos un endpoint funciona correctamente

---

## 🎓 PRÓXIMOS PASOS

Una vez que todo funcione:

1. **Explora la API:** Usa Swagger UI para probar todos los endpoints
2. **Prueba diferentes consultas:** Experimenta con el chat
3. **Revisa los logs:** `docker-compose logs -f` para entender qué hace
4. **Lee la documentación completa:** `README.md` y `README_CHURN_SYSTEM.md`
5. **Integra con tus sistemas:** Usa los endpoints desde tus aplicaciones

---

## 🆘 ¿NECESITAS AYUDA?

Si tienes problemas:

1. **Revisa los logs:** `docker-compose logs --tail=100`
2. **Verifica el estado:** `docker-compose ps`
3. **Consulta Solución de Problemas:** Sección anterior
4. **GitHub Issues:** Reporta problemas en el repositorio

---

## 📝 INFORMACIÓN DEL SISTEMA

**URLs Importantes:**
- API Base: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

**Archivos Importantes:**
- `churn_chat_api.py` - Código principal de la API
- `train_churn_prediction.py` - Script de entrenamiento
- `docker-compose.yml` - Configuración del contenedor
- `Dockerfile` - Construcción de la imagen
- `requirements.txt` - Dependencias de Python

**Puertos:**
- 8000: API FastAPI

**Volúmenes:**
- `./churn_model` - Modelo entrenado (persistente)
- `./Churn_Modelling.csv` - Dataset (solo lectura)

---

**¡Listo! Tu Sistema de Predicción de Churn está funcionando en Docker! 🎉**
