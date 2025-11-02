# 🧪 Sistema de Pruebas Completas - Modelo de Predicción de Churn

Este conjunto de herramientas permite evaluar exhaustivamente el modelo de predicción de churn y generar reportes profesionales para el informe final.

---

## 📋 Contenido

1. [Archivos Incluidos](#archivos-incluidos)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Uso Rápido](#uso-rápido)
5. [Resultados Generados](#resultados-generados)
6. [Interpretación de Métricas](#interpretación-de-métricas)
7. [Personalización](#personalización)

---

## 📁 Archivos Incluidos

```
Fuga/
├── test_models.py          # Script principal de evaluación
├── generate_report.py      # Generador de reporte HTML
├── run_tests.sh           # Script de ejecución automatizada
└── README_TESTS.md        # Esta documentación
```

### `test_models.py`
Script principal que:
- Carga el modelo entrenado
- Genera predicciones en datos de test
- Calcula métricas exhaustivas
- Crea visualizaciones profesionales
- Analiza rendimiento por segmentos
- Guarda todos los resultados en JSON

### `generate_report.py`
Generador de reporte HTML que:
- Lee los resultados de `test_models.py`
- Crea un reporte HTML interactivo
- Embebe todas las visualizaciones
- Incluye recomendaciones automáticas
- Formato profesional listo para presentar

### `run_tests.sh`
Script bash que ejecuta todo el pipeline:
- Verifica requisitos
- Ejecuta las pruebas
- Genera el reporte
- Muestra resumen de resultados

---

## ⚙️ Requisitos

### Python 3.10+
```bash
python3 --version  # Debe ser >= 3.10
```

### Dependencias
Todas están en `requirements.txt`:
- transformers
- torch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Modelo Entrenado
Debe existir el directorio `churn_model/` con el modelo entrenado.

Si no existe:
```bash
python3 train_churn_prediction.py
```

---

## 🚀 Instalación

```bash
# 1. Clonar o navegar al directorio
cd Fuga/

# 2. Activar entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# O: venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## ⚡ Uso Rápido

### Opción 1: Script Automatizado (Recomendado)

```bash
# Ejecutar todo el pipeline
chmod +x run_tests.sh
./run_tests.sh
```

Esto:
1. ✅ Verifica requisitos
2. ✅ Entrena modelo si no existe
3. ✅ Ejecuta todas las pruebas
4. ✅ Genera reporte HTML
5. ✅ Muestra resumen

### Opción 2: Manual (Paso a Paso)

```bash
# 1. Ejecutar pruebas
python3 test_models.py

# 2. Generar reporte HTML
python3 generate_report.py

# 3. Abrir reporte
open test_results/informe_completo.html  # macOS
```

---

## 📊 Resultados Generados

Todos los resultados se guardan en `test_results/`:

```
test_results/
├── informe_completo.html           # 🌟 Reporte principal (abrir en navegador)
├── metrics.json                    # Métricas en formato JSON
├── classification_report.json      # Reporte detallado de clasificación
├── test_summary.json              # Resumen ejecutivo
├── threshold_analysis.json        # Análisis de umbrales
├── segments_analysis.json         # Análisis por segmentos
├── error_examples.csv             # Ejemplos de errores
├── confusion_matrix.png           # Matriz de confusión
├── roc_curve.png                  # Curva ROC
├── precision_recall_curve.png     # Curva Precision-Recall
├── probability_distribution.png   # Distribución de probabilidades
├── metrics_summary.png            # Resumen visual de métricas
└── threshold_analysis.png         # Visualización de umbrales
```

### 🌟 Archivo Principal

**`informe_completo.html`** - Reporte HTML interactivo con:
- ✅ Resumen ejecutivo con métricas clave
- ✅ Matriz de confusión interactiva
- ✅ Curvas ROC y Precision-Recall
- ✅ Análisis de umbrales
- ✅ Recomendaciones automáticas
- ✅ Conclusiones y próximos pasos
- ✅ Diseño profesional responsive

---

## 📈 Interpretación de Métricas

### Métricas Principales

| Métrica | Descripción | Valor Ideal |
|---------|-------------|-------------|
| **Accuracy** | % de predicciones correctas (total) | > 0.80 |
| **Precision** | De los que predecimos CHURN, % correctos | > 0.70 |
| **Recall** | De los que hacen CHURN, % detectados | > 0.70 |
| **F1-Score** | Balance entre Precision y Recall | > 0.70 |
| **ROC-AUC** | Capacidad de discriminación | > 0.80 |

### Matriz de Confusión

```
                    Predicción
                 No Churn  |  Churn
Real ────────────────────────────────
No Churn │   TN (✅)   │   FP (❌)
         │  Correcto   │   Error
─────────┼─────────────┼───────────
Churn    │   FN (❌)   │   TP (✅)
         │   Error     │  Correcto
```

- **TN (True Negative)**: Cliente no hizo churn y lo predijimos correctamente ✅
- **FP (False Positive)**: Cliente no hizo churn pero predijimos que sí ❌ (Costo: campaña innecesaria)
- **FN (False Negative)**: Cliente hizo churn pero no lo detectamos ❌ (Costo: cliente perdido)
- **TP (True Positive)**: Cliente hizo churn y lo detectamos ✅ (Éxito: oportunidad de retención)

### Trade-offs

**Aumentar Recall (detectar más churners):**
- ✅ Capturamos más clientes en riesgo
- ❌ Más falsos positivos (campañas innecesarias)
- 💡 Usar si: El costo de perder un cliente > costo de campaña

**Aumentar Precision (evitar falsos positivos):**
- ✅ Menos campañas innecesarias
- ❌ Perdemos algunos clientes en riesgo
- 💡 Usar si: El costo de campaña es alto

---

## 🎯 Personalización

### Cambiar Umbral de Decisión

Edita `test_models.py` línea ~365:

```python
# Cambiar umbrales a probar
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

### Agregar Más Visualizaciones

Agrega en `test_models.py` después de la línea 300:

```python
# Tu código de visualización
fig, ax = plt.subplots(figsize=(10, 6))
# ... tu gráfico
plt.savefig(RESULTS_DIR / 'mi_grafico.png', dpi=300)
```

Luego actualiza `generate_report.py` para incluirlo en el HTML.

### Personalizar Reporte HTML

Edita `generate_report.py`:
- **Colores**: Modifica la sección `<style>` (línea ~80)
- **Secciones**: Agrega nuevas secciones HTML (después línea ~400)
- **Logo**: Agrega tu logo en base64

---

## 📝 Ejemplos de Uso

### Evaluación Completa

```bash
# Ejecutar todo
./run_tests.sh

# Ver reporte
open test_results/informe_completo.html
```

### Solo Métricas (Sin Reporte)

```bash
python3 test_models.py
cat test_results/metrics.json
```

### Solo Reporte (Actualizar Diseño)

```bash
# Editar generate_report.py
# Luego regenerar
python3 generate_report.py
```

### Exportar Métricas

```bash
# Métricas en JSON
cat test_results/metrics.json | jq '.'

# Ejemplos de errores en CSV
open test_results/error_examples.csv
```

---

## 🐛 Solución de Problemas

### Error: "No se encontró el modelo"

```bash
# Solución: Entrenar modelo primero
python3 train_churn_prediction.py
```

### Error: "ModuleNotFoundError"

```bash
# Solución: Instalar dependencias
pip install -r requirements.txt
```

### Reporte HTML no se ve bien

```bash
# Solución: Usar navegador moderno (Chrome, Firefox, Safari)
# Evitar Internet Explorer
```

### Imágenes no aparecen en el reporte

```bash
# Verificar que existen
ls test_results/*.png

# Si faltan, regenerar
python3 test_models.py
```

---

## 💡 Tips Profesionales

### Para el Informe Final

1. **Captura de pantalla**: Usa las visualizaciones PNG para slides
2. **Métricas JSON**: Importa a Excel/LaTeX para tablas
3. **Reporte HTML**: Comparte link o PDF del navegador
4. **Análisis de errores**: Usa `error_examples.csv` para casos específicos

### Formato PDF

```bash
# Desde el navegador:
# 1. Abrir informe_completo.html
# 2. Ctrl+P / Cmd+P
# 3. "Guardar como PDF"
# 4. Configurar márgenes a "Ninguno"
```

### Automatización

```bash
# Agregar a cron para evaluación periódica
0 0 * * * cd /path/to/Fuga && ./run_tests.sh
```

---

## 📚 Referencias

- [Documentación scikit-learn - Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ROC Curve Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)

---

## 🤝 Soporte

Si tienes problemas:

1. Verifica requisitos (Python 3.10+, dependencias)
2. Revisa logs de error
3. Consulta sección "Solución de Problemas"
4. Verifica que `churn_model/` existe

---

## ✅ Checklist para Informe Final

- [ ] Ejecutar `./run_tests.sh`
- [ ] Revisar `test_results/informe_completo.html`
- [ ] Exportar reporte a PDF
- [ ] Incluir visualizaciones PNG en slides
- [ ] Copiar métricas JSON a documentación
- [ ] Analizar ejemplos de errores
- [ ] Documentar interpretación de resultados
- [ ] Incluir recomendaciones del reporte

---

**¡Listo para generar tu informe profesional! 🚀**

Para cualquier duda, consulta la documentación o revisa los comentarios en `test_models.py` y `generate_report.py`.
