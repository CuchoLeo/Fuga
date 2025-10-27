# ⚡ Quick Start - 5 Minutos

## 🎯 Inicio Rápido

Si solo quieres probar el sistema **AHORA MISMO**, sigue estos pasos:

### **1. Prerequisitos**
```bash
# Verificar Docker
docker --version
docker ps
```

### **2. Clonar y Preparar**
```bash
git clone https://github.com/CuchoLeo/Fuga.git
cd Fuga
git checkout claude/create-docker-image-011CUWiCdkyttEZPktomfqF1
```

### **3. Descargar Dataset**
Descarga manualmente desde:
👉 https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

Coloca `Churn_Modelling.csv` en la carpeta `Fuga/`

### **4. Construir y Entrenar**
```bash
# Construir imagen (5-10 min)
docker-compose build

# Entrenar modelo (5-15 min)
docker-compose run --rm churn-api python train_churn_prediction.py

# Iniciar aplicación
docker-compose up -d
```

### **5. ¡Usar!**

Abre en tu navegador:
```
http://localhost:8000/docs
```

O prueba con:
```bash
curl http://localhost:8000/health
```

---

## 📚 Documentación Completa

- **Instalación Detallada:** [INSTALACION_DOCKER.md](INSTALACION_DOCKER.md)
- **Solución de Problemas:** Ver INSTALACION_DOCKER.md sección "Solución de Problemas"
- **Documentación Técnica:** [README_CHURN_SYSTEM.md](README_CHURN_SYSTEM.md)

---

## 🎓 Ejemplos de Uso

### **Swagger UI (Recomendado)**
```
http://localhost:8000/docs
```

### **Obtener Clientes en Riesgo**
```bash
curl "http://localhost:8000/at-risk?limit=10" | python3 -m json.tool
```

### **Chat en Lenguaje Natural**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuántos clientes están en riesgo?"}' \
  | python3 -m json.tool
```

### **Predecir Churn**
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

---

## 🛠️ Comandos Básicos

```bash
# Ver estado
docker-compose ps

# Ver logs
docker-compose logs -f

# Reiniciar
docker-compose restart

# Detener
docker-compose down
```

---

## ❓ ¿Problemas?

**Error común:** "Modelo no encontrado"
```bash
docker-compose run --rm churn-api python train_churn_prediction.py
docker-compose restart
```

**Puerto ocupado:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Ver documentación completa:** [INSTALACION_DOCKER.md](INSTALACION_DOCKER.md)

---

**¡Eso es todo! 🚀**
