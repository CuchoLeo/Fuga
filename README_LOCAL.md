# 🚀 Ejecutar Churnito Localmente (Sin Docker)

Esta guía te permite ejecutar Churnito directamente en tu máquina sin necesidad de Docker.

---

## 📋 **REQUISITOS PREVIOS**

### 1. Python 3.11 o superior
```bash
# Verificar versión de Python
python3 --version
# Debe mostrar: Python 3.11.x o superior
```

### 2. pip (gestor de paquetes Python)
```bash
# Verificar pip
pip3 --version
```

---

## ⚡ **INICIO RÁPIDO (3 pasos)**

### **Paso 1: Instalar dependencias**
```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv

# Activar entorno virtual
# En Mac/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### **Paso 2: Ejecutar el servidor**
```bash
python run_local.py
```

### **Paso 3: Abrir en navegador**
```
http://localhost:8000
```

---

## 📦 **INSTALACIÓN DETALLADA**

### **Opción A: Usando el script automático (Recomendado)**

```bash
# 1. Navegar al directorio del proyecto
cd Fuga/

# 2. Crear entorno virtual
python3 -m venv venv

# 3. Activar entorno virtual
source venv/bin/activate  # Mac/Linux
# O en Windows: venv\Scripts\activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar servidor
python run_local.py
```

**El script verificará automáticamente:**
- ✅ Archivos necesarios
- ✅ Dependencias instaladas
- ✅ Creación de directorios

---

### **Opción B: Usando uvicorn directamente**

```bash
# 1. Activar entorno virtual (si usas uno)
source venv/bin/activate

# 2. Ejecutar servidor con uvicorn
uvicorn churn_chat_api:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 **ACCEDER A LA APLICACIÓN**

Una vez iniciado el servidor, abre tu navegador:

| URL | Descripción |
|-----|-------------|
| `http://localhost:8000` | 💬 Interfaz de chat con Churnito |
| `http://localhost:8000/docs` | 📖 Documentación interactiva (Swagger) |
| `http://localhost:8000/health` | ❤️ Estado del sistema |
| `http://localhost:8000/api` | 📊 Información de endpoints |

---

## 🧪 **PRIMERA PRUEBA**

1. Abre `http://localhost:8000` en tu navegador
2. Deberías ver la interfaz de chat con tema oscuro
3. Escribe: **"Hola Churnito, ¿cómo te llamas?"**
4. Churnito debería responder presentándose

**Otras consultas de prueba:**
```
Muéstrame los 10 clientes con mayor riesgo de fuga
¿Cuál es la tasa de churn actual?
¿Cuántos clientes de alto valor tenemos?
Dame estrategias para reducir el churn
```

---

## 📂 **ESTRUCTURA DE ARCHIVOS**

```
Fuga/
├── run_local.py              ← Script para ejecutar localmente
├── churn_chat_api.py         ← Aplicación principal (FastAPI)
├── chat_interface.html       ← Interfaz web del chat
├── train_churn_prediction.py ← Entrenamiento del modelo
├── Churn_Modelling.csv       ← Datos de entrenamiento
├── requirements.txt          ← Dependencias Python
├── churn_model/              ← Modelo entrenado (se crea automáticamente)
└── trained_model/            ← LLM descargado (se crea automáticamente)
```

---

## ⚙️ **CONFIGURACIÓN AVANZADA**

### **Cambiar puerto:**
```bash
# Editar run_local.py, línea ~77:
uvicorn.run(
    "churn_chat_api:app",
    host="0.0.0.0",
    port=8080,  # ← Cambiar aquí
    reload=True
)
```

### **Desactivar auto-reload:**
```bash
# Para producción, cambia reload=True a reload=False
uvicorn.run(
    "churn_chat_api:app",
    host="0.0.0.0",
    port=8000,
    reload=False  # ← Cambiar aquí
)
```

---

## 🛑 **DETENER EL SERVIDOR**

Presiona `Ctrl+C` en la terminal donde está corriendo el servidor.

```bash
# Salida esperada:
^C
======================================================================
👋 Servidor detenido
======================================================================
```

---

## 🐛 **SOLUCIÓN DE PROBLEMAS**

### **Error: "No module named 'fastapi'"**
```bash
# Solución: Instalar dependencias
pip install -r requirements.txt
```

### **Error: "Address already in use"**
```bash
# El puerto 8000 ya está en uso por otro proceso
# Solución 1: Detener el otro proceso
lsof -ti:8000 | xargs kill -9

# Solución 2: Cambiar el puerto en run_local.py
```

### **Error: "churn_model not found"**
```bash
# Solución: Entrenar el modelo primero
python train_churn_prediction.py
```

### **LLM tarda mucho en cargar (primera vez)**
```
✅ NORMAL - La primera vez descarga ~3GB del modelo Qwen2.5
⏱️ Tiempo estimado: 5-10 minutos (dependiendo de tu conexión)
💾 Se guarda en ./trained_model/ para usos futuros
```

### **Error: "ModuleNotFoundError: No module named 'torch'"**
```bash
# Instalar PyTorch manualmente
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 🆚 **DOCKER vs LOCAL**

| Aspecto | Docker | Local |
|---------|--------|-------|
| **Instalación** | Solo Docker Desktop | Python + dependencias |
| **Aislamiento** | ✅ Completo | ❌ Usa Python del sistema |
| **Velocidad inicial** | Lenta (build imagen) | Rápida |
| **Desarrollo** | Reconstruir imagen | ✅ Auto-reload instantáneo |
| **Portabilidad** | ✅ Funciona igual en cualquier OS | Depende del entorno |
| **Uso recomendado** | Producción, distribución | Desarrollo, pruebas rápidas |

---

## 📝 **NOTAS IMPORTANTES**

1. **Entorno virtual recomendado**: Usa `venv` para evitar conflictos con otras instalaciones de Python
2. **Primera ejecución**: El LLM se descargará (~3GB), esto puede tardar varios minutos
3. **Modelo de churn**: Si no existe en `churn_model/`, debes ejecutar `train_churn_prediction.py` primero
4. **Memoria RAM**: El LLM requiere ~4-6GB de RAM disponible
5. **Auto-reload**: El modo desarrollo recarga automáticamente cuando cambias el código

---

## 🎯 **PRÓXIMOS PASOS**

Una vez que el servidor esté corriendo:

1. ✅ Prueba la interfaz web en `http://localhost:8000`
2. ✅ Explora la documentación en `/docs`
3. ✅ Conversa con Churnito y prueba diferentes consultas
4. ✅ Revisa los logs en la terminal para debug

---

## 🤝 **AYUDA**

Si tienes problemas:
1. Verifica que Python 3.11+ esté instalado
2. Asegúrate de estar en el directorio correcto (`Fuga/`)
3. Revisa que todas las dependencias estén instaladas
4. Consulta la sección de "Solución de problemas" arriba

---

**¡Disfruta conversando con Churnito! 🤖**
