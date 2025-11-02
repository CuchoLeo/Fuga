# Despliegue con Docker

Este documento describe cómo ejecutar el Sistema de Predicción de Churn usando Docker.

## Requisitos Previos

- Docker instalado (v20.10 o superior)
- Docker Compose instalado (v2.0 o superior)
- Al menos 4GB de RAM disponible
- Dataset `Churn_Modelling.csv` en el directorio raíz

## Opciones de Despliegue

### Opción 1: Docker Compose (Recomendado)

La forma más sencilla de ejecutar la aplicación:

```bash
# Construir y ejecutar
docker-compose up --build

# Ejecutar en segundo plano
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

### Opción 2: Docker directo

Si prefieres usar Docker sin Compose:

```bash
# Construir la imagen
docker build -t churn-prediction-api .

# Ejecutar el contenedor
docker run -d \
  --name churn-api \
  -p 8000:8000 \
  -v $(pwd)/churn_model:/app/churn_model \
  -v $(pwd)/trained_model:/app/trained_model \
  -v $(pwd)/Churn_Modelling.csv:/app/Churn_Modelling.csv \
  churn-prediction-api

# Ver logs
docker logs -f churn-api

# Detener
docker stop churn-api
docker rm churn-api
```

## Entrenar Modelos Dentro del Contenedor

Si necesitas entrenar los modelos dentro del contenedor:

```bash
# Entrenar modelo de churn
docker-compose exec churn-api python train_churn_prediction.py

# O con Docker directo
docker exec -it churn-api python train_churn_prediction.py
```

## Verificar el Estado

Una vez que el contenedor esté ejecutándose:

```bash
# Verificar health check
curl http://localhost:8000/health

# Obtener estadísticas
curl http://localhost:8000/statistics

# Probar el chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuántos clientes están en riesgo?"}'
```

## Acceder a la Aplicación

- API: http://localhost:8000
- Documentación interactiva (Swagger): http://localhost:8000/docs
- Documentación alternativa (ReDoc): http://localhost:8000/redoc
- Health check: http://localhost:8000/health

## Interfaz Web

Para usar la interfaz web HTML:

1. Abre `chat_interface.html` en tu navegador
2. O sirve la página con un servidor HTTP simple:
   ```bash
   python -m http.server 8080
   ```
3. Accede a http://localhost:8080/chat_interface.html

## Solución de Problemas

### Contenedor no inicia

```bash
# Ver logs completos
docker-compose logs

# Verificar si el puerto está en uso
lsof -i :8000
```

### Modelos no encontrados

Si los modelos no están entrenados, el contenedor iniciará pero advertirá:
```
⚠️  Modelo de churn no encontrado. Ejecuta train_churn_prediction.py primero
```

Solución:
```bash
# Entrenar dentro del contenedor
docker-compose exec churn-api python train_churn_prediction.py
```

### Memoria insuficiente

Si el contenedor se queda sin memoria:

```bash
# Aumentar memoria asignada a Docker (en Docker Desktop)
# O ejecutar con límite de memoria específico
docker run -m 4g ...
```

### Dataset no encontrado

Asegúrate de que `Churn_Modelling.csv` existe en el directorio raíz antes de construir/ejecutar.

## Configuración Avanzada

### Variables de Entorno

Puedes personalizar la configuración editando `docker-compose.yml`:

```yaml
environment:
  - API_HOST=0.0.0.0
  - API_PORT=8000
  - CHURN_MODEL_PATH=/app/churn_model
  - CSV_PATH=/app/Churn_Modelling.csv
  - HIGH_VALUE_THRESHOLD=100000
```

### Volúmenes Persistentes

Los modelos entrenados se guardan en volúmenes que persisten entre reinicios:

```yaml
volumes:
  - ./churn_model:/app/churn_model
  - ./trained_model:/app/trained_model
```

## Comandos Útiles

```bash
# Reconstruir imagen sin caché
docker-compose build --no-cache

# Ver uso de recursos
docker stats churn-prediction-api

# Entrar al contenedor
docker-compose exec churn-api bash

# Eliminar todo (contenedor, volúmenes, imágenes)
docker-compose down -v
docker rmi churn-prediction-api
```

## Despliegue en Producción

Para producción, considera:

1. **Usar Gunicorn con múltiples workers:**
   ```dockerfile
   CMD ["gunicorn", "churn_chat_api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
   ```

2. **Configurar límites de recursos:**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
       reservations:
         cpus: '1'
         memory: 2G
   ```

3. **Usar secretos para configuración sensible:**
   ```yaml
   secrets:
     - api_key
   ```

4. **Implementar logging estructurado:**
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

## Performance

Tiempos esperados:
- Primera carga (descarga modelos): 2-5 minutos
- Startup normal: 30-60 segundos
- Respuesta API: 200-500ms por predicción

## Soporte

Para problemas con Docker:
1. Verifica los logs: `docker-compose logs`
2. Verifica el health check: `curl http://localhost:8000/health`
3. Revisa los requisitos de sistema
4. Consulta el README principal para más detalles
