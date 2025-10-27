# 🍎 INSTRUCCIONES PARA MAC - Docker Setup

## ✅ Archivos Docker Creados

Los siguientes archivos han sido creados en tu proyecto:

- ✅ `requirements.txt` - Dependencias de Python
- ✅ `Dockerfile` - Configuración de la imagen Docker
- ✅ `.dockerignore` - Archivos a excluir de la imagen
- ✅ `docker-compose.yml` - Configuración del contenedor
- ✅ `setup-docker.sh` - Script automatizado de instalación

---

## 🚀 OPCIÓN 1: Instalación Automática (Recomendada)

Abre tu **Terminal en Mac** y ejecuta:

```bash
# 1. Ve a la carpeta del proyecto
cd /Users/cucho/ruta/a/Fuga

# 2. Asegúrate de estar en el branch correcto
git pull origin claude/create-docker-image-011CUWiCdkyttEZPktomfqF1

# 3. Ejecuta el script de setup
chmod +x setup-docker.sh
./setup-docker.sh
```

El script hará TODO automáticamente:
- ✅ Verificará Docker
- ✅ Construirá la imagen
- ✅ Iniciará el contenedor
- ✅ Entrenará el modelo
- ✅ Verificará que todo funcione

**Tiempo estimado:** 10-20 minutos

---

## ⚙️ OPCIÓN 2: Instalación Manual (Paso a Paso)

Si prefieres hacerlo manualmente, sigue estos pasos:

### **PASO 1: Obtener los archivos Docker**

```bash
# En tu terminal de Mac
cd /Users/cucho/ruta/a/Fuga

# Obtener los últimos cambios
git fetch origin
git checkout claude/create-docker-image-011CUWiCdkyttEZPktomfqF1
git pull
```

### **PASO 2: Verificar que tienes Docker**

```bash
# Verificar Docker
docker --version

# Verificar Docker Compose
docker-compose --version
# o
docker compose version
```

Si no tienes Docker instalado:
👉 Descarga Docker Desktop: https://www.docker.com/products/docker-desktop

### **PASO 3: Verificar archivos necesarios**

```bash
ls -lh Dockerfile docker-compose.yml requirements.txt Churn_Modelling.csv
```

Debes ver todos estos archivos listados.

### **PASO 4: Construir la imagen Docker**

```bash
# Construir (tardará 5-10 minutos la primera vez)
docker-compose build

# O con el comando moderno:
docker compose build
```

**Salida esperada:**
```
[+] Building 245.3s (12/12) FINISHED
 => [1/6] FROM docker.io/library/python:3.11-slim
 => [2/6] RUN apt-get update && apt-get install...
 => [3/6] WORKDIR /app
 => [4/6] COPY requirements.txt .
 => [5/6] RUN pip install --no-cache-dir -r requirements.txt
 => [6/6] COPY *.py *.csv *.html ./
 => exporting to image
Successfully tagged churn-prediction-api:latest
```

### **PASO 5: Iniciar el contenedor**

```bash
# Iniciar en segundo plano
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f
```

Presiona `Ctrl+C` para salir de los logs (el contenedor seguirá corriendo).

### **PASO 6: Verificar que funciona**

```bash
# Health check
curl http://localhost:8000/health

# Debería responder algo como:
# {
#   "status": "healthy",
#   "churn_model_loaded": false,  # (porque aún no entrenamos)
#   "llm_loaded": true,
#   "database_loaded": true
# }
```

### **PASO 7: Entrenar el modelo dentro del contenedor**

```bash
# Entrenar el modelo (tardará 5-15 minutos)
docker-compose exec churn-api python train_churn_prediction.py
```

**Salida esperada:**
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

### **PASO 8: Reiniciar el contenedor**

```bash
# Reiniciar para cargar el modelo entrenado
docker-compose restart

# Esperar 30 segundos y verificar
sleep 30
curl http://localhost:8000/health

# Ahora churn_model_loaded debe ser true
```

### **PASO 9: Probar la API**

```bash
# Obtener estadísticas
curl http://localhost:8000/statistics

# Probar el chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuántos clientes están en riesgo?"}'

# Abrir documentación interactiva en el navegador
open http://localhost:8000/docs
```

---

## 🌐 URLs Importantes

Una vez que el contenedor esté corriendo:

| Servicio | URL |
|----------|-----|
| API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |
| Statistics | http://localhost:8000/statistics |

---

## 📋 Comandos Útiles

```bash
# Ver estado de contenedores
docker-compose ps

# Ver logs
docker-compose logs -f

# Detener el contenedor
docker-compose stop

# Iniciar el contenedor
docker-compose start

# Reiniciar el contenedor
docker-compose restart

# Entrar al contenedor (bash interactivo)
docker-compose exec churn-api bash

# Ver uso de recursos
docker stats churn-prediction-api

# Detener y eliminar todo
docker-compose down

# Eliminar incluyendo volúmenes
docker-compose down -v
```

---

## 🐛 Solución de Problemas

### Problema 1: Puerto 8000 ya en uso

```bash
# Ver qué usa el puerto
lsof -i :8000

# Matar el proceso (reemplaza PID con el número real)
kill -9 <PID>

# O cambiar el puerto en docker-compose.yml:
# ports:
#   - "8001:8000"
```

### Problema 2: Docker no responde

```bash
# Reiniciar Docker Desktop en Mac
# Menú > Docker Desktop > Restart

# Verificar Docker
docker info
```

### Problema 3: Contenedor se detiene inmediatamente

```bash
# Ver logs de error
docker-compose logs

# Ver logs específicos del último arranque
docker logs churn-prediction-api --tail 100
```

### Problema 4: Error de permisos

```bash
# Si hay error con volúmenes, dar permisos
chmod -R 755 churn_model/
chmod -R 755 trained_model/
```

### Problema 5: Build muy lento

```bash
# Limpiar caché de Docker
docker system prune -a

# Reconstruir sin caché
docker-compose build --no-cache
```

---

## 🎯 Verificación Final

Ejecuta estos comandos para verificar que todo funciona:

```bash
# 1. Contenedor corriendo
docker-compose ps
# Debería mostrar: churn-prediction-api   Up

# 2. Health check OK
curl http://localhost:8000/health
# Debería responder: {"status": "healthy", ...}

# 3. Estadísticas disponibles
curl http://localhost:8000/statistics
# Debería mostrar datos del churn

# 4. Modelo cargado
curl http://localhost:8000/health | grep churn_model_loaded
# Debería mostrar: "churn_model_loaded": true
```

---

## ✅ Checklist de Validación

Marca cada item cuando lo completes:

- [ ] Docker Desktop instalado en Mac
- [ ] Archivos Docker descargados (git pull)
- [ ] Imagen Docker construida (`docker-compose build`)
- [ ] Contenedor iniciado (`docker-compose up -d`)
- [ ] Health check responde OK
- [ ] Modelo entrenado dentro del contenedor
- [ ] Contenedor reiniciado después del entrenamiento
- [ ] Modelo cargado correctamente (churn_model_loaded: true)
- [ ] API responde en http://localhost:8000
- [ ] Swagger docs accesible en /docs
- [ ] Chat funciona correctamente

---

## 🎓 Próximos Pasos

Una vez que todo funcione:

1. **Explorar la API:** http://localhost:8000/docs
2. **Probar el chat:** Usa curl o la interfaz HTML
3. **Integrar con tus sistemas:** Usa los endpoints en tus aplicaciones
4. **Monitorear:** `docker stats` para ver uso de recursos

---

## 📞 Ayuda Adicional

Si tienes problemas:

1. Revisa los logs: `docker-compose logs -f`
2. Verifica el estado: `docker-compose ps`
3. Revisa health check: `curl http://localhost:8000/health`
4. Consulta README_DOCKER.md para más detalles

---

**¡Listo! Tu sistema de predicción de churn está funcionando en Docker! 🎉**
