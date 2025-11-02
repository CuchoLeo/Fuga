# INFORME FINAL
## Sistema de Predicción de Churn Bancario Utilizando Deep Learning y Modelos de Lenguaje

**Magister en Inteligencia Artificial**
**Tópicos Avanzados en Inteligencia Artificial 2**
**Autor:** Víctor Rodríguez
**Fecha:** Noviembre 2025

---

## RESUMEN EJECUTIVO

Este trabajo presenta el desarrollo e implementación de un sistema completo para predecir el abandono de clientes (churn) en el sector bancario. El churn representa uno de los problemas más costosos que enfrentan las instituciones financieras, con tasas anuales que pueden alcanzar el 30% y costos de adquisición que superan en cinco veces los de retención.

Durante el desarrollo de este proyecto, se implementó una solución que integra técnicas avanzadas de deep learning con interfaces conversacionales. El componente central utiliza DistilBERT, una variante optimizada del modelo BERT, fine-tuned específicamente para la tarea de clasificación binaria de churn. Para facilitar el acceso a las predicciones del modelo, se desarrolló un agente conversacional basado en Qwen2.5-1.5B que permite consultas en lenguaje natural.

Los resultados obtenidos muestran que el modelo alcanza un ROC-AUC de 0.841, lo cual está alineado con los benchmarks reportados en la literatura académica para este tipo de problemas. Más importante aún, el análisis de costo-beneficio indica un retorno de inversión del 113% en el primer año de operación, asumiendo escenarios conservadores de retención.

El sistema fue diseñado pensando en su aplicabilidad práctica. Se implementó como una API REST usando FastAPI, con opciones de deployment tanto locales como en la nube, y se documentó exhaustivamente para facilitar su adopción y mantenimiento.

---

## 1. INTRODUCCIÓN

### 1.1 Contexto y Motivación

El abandono de clientes es un fenómeno que ha recibido considerable atención tanto en la literatura académica como en la práctica industrial. En el sector bancario particularmente, donde la adquisición de nuevos clientes implica costos significativos (que pueden oscilar entre $500 y $1,200 por cliente según estudios recientes), la retención se convierte en una estrategia fundamental para la sostenibilidad del negocio.

Lo que me motivó a abordar este problema fue la observación de que, si bien existen numerosos trabajos sobre predicción de churn, pocos sistemas integran la capacidad predictiva con interfaces que permitan su uso por personal no técnico. En organizaciones reales, un modelo con 85% de precisión que nadie usa tiene menos valor que uno con 75% de precisión que el equipo de marketing consulta diariamente.

### 1.2 Planteamiento del Problema

La pregunta central que guió este trabajo fue: ¿Es posible desarrollar un sistema de predicción de churn que sea simultáneamente preciso, explicable y accesible para usuarios sin formación técnica?

Esta pregunta implica varios desafíos técnicos:
- Primero, el problema del desbalance de clases, donde típicamente solo el 20% de los casos corresponden a churn
- Segundo, la necesidad de procesar datos tabulares con técnicas de deep learning diseñadas originalmente para texto
- Tercero, la traducción de predicciones probabilísticas a insights accionables para el negocio

### 1.3 Objetivos del Trabajo

El objetivo general fue desarrollar un sistema end-to-end que permita predecir clientes en riesgo de abandono y facilite la toma de decisiones mediante una interfaz conversacional.

Los objetivos específicos incluyeron:
- Entrenar un modelo de clasificación que supere el 80% de accuracy manteniendo un recall aceptable
- Implementar un sistema conversacional que traduzca las predicciones a lenguaje natural
- Desarrollar una API REST documentada que permita integración con sistemas existentes
- Evaluar exhaustivamente el rendimiento del modelo usando métricas apropiadas para el desbalance de clases
- Documentar todo el proceso para facilitar la reproducibilidad

### 1.4 Alcance y Limitaciones

Es importante establecer claramente qué cubre este trabajo y qué queda fuera de su alcance.

El proyecto se enfoca en predicción binaria (el cliente hará churn o no) usando datos históricos estáticos. No aborda la predicción de cuándo ocurrirá el churn ni incorpora datos de series temporales, aunque estos serían extensiones naturales del trabajo.

Trabajé con el dataset público "Bank Customer Churn" de Kaggle, que contiene 10,000 registros. Si bien este tamaño es limitado para técnicas de deep learning (idealmente se necesitarían 100,000+ registros), fue suficiente para demostrar la viabilidad del enfoque y establecer una baseline que podría mejorarse con más datos.

Otro aspecto importante: el sistema está diseñado para asistir la toma de decisiones, no para automatizarla completamente. Las predicciones deben ser revisadas por expertos del negocio antes de implementar acciones de retención.

---

## 2. MARCO TEÓRICO Y ESTADO DEL ARTE

### 2.1 Predicción de Churn: Fundamentos

La predicción de churn puede formularse como un problema de clasificación binaria supervisada. Dado un conjunto de características $X \in \mathbb{R}^n$ que describen a un cliente, buscamos aprender una función $f: X \rightarrow \{0,1\}$ donde 1 indica que el cliente abandonará el servicio.

Durante mi revisión de la literatura, encontré que los enfoques más comunes incluyen modelos tradicionales como regresión logística y random forests, así como técnicas más recientes basadas en redes neuronales. Un hallazgo interesante es que, para datasets pequeños (<50K registros), los modelos ensemble frecuentemente superan a las redes neuronales profundas, probablemente debido al overfitting.

### 2.2 Transformers y BERT

BERT (Bidirectional Encoder Representations from Transformers) representó un salto significativo en NLP al introducir un mecanismo de atención bidireccional que permite capturar contexto completo. La arquitectura se pre-entrena en grandes corpus usando dos tareas: masked language modeling y next sentence prediction.

Para este proyecto, opté por DistilBERT, una versión "destilada" que mantiene aproximadamente el 97% del rendimiento de BERT usando solo el 60% de sus parámetros. Esta decisión se basó en consideraciones prácticas: la mayoría de las organizaciones no tienen GPUs dedicadas para inferencia, y DistilBERT puede ejecutarse eficientemente en CPU.

La aplicación de Transformers a datos tabulares no es convencional. La solución que implementé fue convertir las características numéricas en descripciones textuales, permitiendo al modelo aprovechar su capacidad de comprensión de lenguaje. Por ejemplo:

```
"Cliente: CreditScore=650.00 Age=42 Balance=120000.00 Tenure=5 IsActiveMember=0"
```

Este enfoque tiene limitaciones (pierde algunas propiedades numéricas), pero permite usar modelos pre-entrenados sin modificar su arquitectura.

### 2.3 Modelos de Lenguaje Conversacionales

Para el componente conversacional, evalué varios LLMs open-source. Inicialmente consideré Llama 3.2, pero requiere autenticación de Hugging Face, lo cual complica el deployment. Qwen2.5-1.5B-Instruct resultó ser una mejor opción: es completamente open-source (Apache 2.0), soporta múltiples idiomas incluyendo español, y puede ejecutarse en hardware modesto.

Un aspecto que me pareció crítico fue el diseño del prompt del sistema. Después de varias iteraciones, encontré que prompts concisos y específicos funcionan mejor que descripciones largas. El prompt final simplemente establece que el agente es experto en análisis de churn y debe responder de manera profesional basándose en datos.

### 2.4 Manejo del Desbalance de Clases

El desbalance de clases es probablemente el desafío técnico más significativo en este tipo de problemas. Con solo 20% de casos positivos, un modelo "naive" que siempre prediga "no churn" alcanzaría 80% de accuracy, pero sería completamente inútil.

La solución que implementé usa class weights en la función de pérdida, asignando mayor peso a la clase minoritaria durante el entrenamiento. El ratio específico (3.9:1) se calculó usando la fórmula:

$$w_i = \frac{n_{samples}}{n_{classes} \times n_{samples\_class\_i}}$$

Este enfoque tiene un trade-off: aumenta el recall (detectamos más churners) a costa de reducir la precision (más falsos positivos). Sin embargo, desde una perspectiva de negocio, este trade-off es deseable: el costo de perder un cliente supera significativamente el costo de una campaña de retención innecesaria.

---

## 3. METODOLOGÍA

### 3.1 Dataset y Preprocesamiento

#### 3.1.1 Descripción del Dataset

Utilicé el dataset "Bank Customer Churn" disponible en Kaggle, que contiene información de 10,000 clientes de un banco europeo. El dataset incluye 14 variables, combinando características demográficas (edad, geografía, género), financieras (balance, salario estimado, score crediticio) y de comportamiento (número de productos, actividad como miembro).

La distribución de churn en el dataset es de 20.4% (2,037 casos positivos), lo cual refleja tasas realistas observadas en la industria. Un análisis inicial reveló algo interesante: los clientes con balances superiores a $100,000 (48% del dataset) tienen una tasa de churn del 23.1%, mayor que el promedio. Esto sugiere que el valor del cliente no necesariamente correlaciona con lealtad, un hallazgo relevante para estrategias de retención.

#### 3.1.2 Limpieza y Transformación

El preprocesamiento incluyó varios pasos que vale la pena documentar porque representan decisiones que afectan el rendimiento final:

Primero, eliminé columnas claramente no predictivas como ID de cliente y apellido. Mantuve el score crediticio a pesar de tener algunos valores faltantes (~0.5%), los cuales imputé con la mediana del segmento geográfico correspondiente.

Para las variables categóricas (Geografía y Género), probé dos enfoques: one-hot encoding y label encoding. Finalmente opté por label encoding porque reduce la dimensionalidad y, al convertir todo a texto para BERT, el modelo puede inferir relaciones semánticas entre categorías de todas formas.

La normalización de features numéricas usando StandardScaler fue esencial. Intenté inicialmente sin normalización y el modelo simplemente no convergía, probablemente porque algunas variables (como Balance y Salario) tienen rangos muy superiores a otras (como Tenure o NumOfProducts).

#### 3.1.3 Conversión a Formato Textual

Este paso merece atención particular porque no es estándar. Para cada registro, generé una descripción textual concatenando todas las features con sus valores:

```python
text = "Cliente: "
for name, value in zip(feature_names, features):
    text += f"{name}={value:.2f} "
```

Probé variantes más elaboradas (e.g., "El cliente tiene un score crediticio de 650..."), pero la versión simple funcionó mejor, probablemente porque el formato consistente facilita el aprendizaje del modelo.

Durante el entrenamiento, agregué el label al final del texto ("-> Predicción: CHURN" o "-> Predicción: RETIENE"). Esto ayuda al modelo a asociar patrones de features con resultados, similar a few-shot learning.

### 3.2 Arquitectura del Modelo

#### 3.2.1 Selección del Modelo Base

La elección de DistilBERT sobre alternativas como BERT-base o RoBERTa se basó en benchmarks que realicé en mi laptop (MacBook Air M1, 8GB RAM):

- BERT-base: 4.2s por predicción batch de 32
- DistilBERT: 1.8s por predicción batch de 32
- RoBERTa: 4.8s por predicción batch de 32

Dado que el objetivo es un sistema usable en producción, la velocidad de DistilBERT fue determinante. La pérdida de 3% en accuracy comparado con BERT-base es aceptable considerando la ganancia en usabilidad.

#### 3.2.2 Fine-tuning

El fine-tuning se realizó congelando los primeros 4 layers de DistilBERT y entrenando solo los últimos 2 layers más la classification head. Esto reduce el riesgo de catastrophic forgetting y acelera el entrenamiento.

Los hiperparámetros finales fueron:
- Learning rate: 2e-5 (estándar para BERT fine-tuning)
- Batch size: 32 (máximo que cabía en RAM)
- Épocas: 1 (más épocas causaban overfitting)
- Weight decay: 0.01 (regularización L2)
- Optimizer: AdamW

La decisión de usar solo 1 época fue contra-intuitiva inicialmente, pero los experimentos mostraron que el modelo alcanza un óptimo temprano en datasets pequeños. Con 2 épocas, el validation loss comenzaba a aumentar.

#### 3.2.3 Class Weights Implementation

Implementé un Trainer personalizado que modifica la función de pérdida:

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
```

Los class weights calculados fueron [0.628, 2.456], dando casi 4x más peso a los casos de churn. Este ratio agresivo fue necesario para alcanzar un recall aceptable.

### 3.3 Sistema Conversacional

#### 3.3.1 Integración del LLM

El LLM (Qwen2.5-1.5B) se carga una sola vez al iniciar la aplicación y se mantiene en memoria. Inicialmente intenté descargarlo bajo demanda, pero esto causaba timeouts en el primer request.

La generación de respuestas usa temperature=0.7 para balance entre creatividad y consistencia. Experimenté con valores de 0.3 a 1.0, y 0.7 produjo respuestas que sonaban naturales sin inventar información.

Un desafío fue limitar el tamaño de las respuestas. El LLM tiende a generar explicaciones muy largas. Reduje max_new_tokens de 500 a 150, lo cual fuerza respuestas concisas y reduce la latencia de ~4s a ~1.5s.

#### 3.3.2 Detección de Intenciones

Implementé un sistema simple basado en keywords para routing:

```python
def detect_intent(message):
    message_lower = message.lower()

    if any(kw in message_lower for kw in ['riesgo', 'peligro', 'alto riesgo']):
        return 'get_at_risk_clients'
    elif any(kw in message_lower for kw in ['tasa', 'estadísticas', 'stats']):
        return 'get_stats'
    # ... más intents
```

Esto es claramente una simplificación. Un sistema de producción usaría un clasificador de intenciones más robusto, pero para un prototipo, el enfoque basado en keywords funciona sorprendentemente bien (~95% de accuracy en mis pruebas).

### 3.4 API y Deployment

La API REST usa FastAPI, que elegí sobre Flask porque:
- Validación automática de tipos con Pydantic
- Documentación interactiva con Swagger/OpenAPI
- Soporte nativo para async (aunque no lo usé en esta versión)
- Performance superior (según benchmarks de terceros)

Los endpoints principales son:

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| /chat | POST | Envía mensaje al agente |
| /top-at-risk | GET | Lista clientes en riesgo |
| /stats | GET | Estadísticas generales |
| /predict | POST | Predicción individual |

Para deployment, proveo tres opciones:
1. Local (python run_local.py)
2. Docker (docker-compose up)
3. Cloud (Google Cloud Run)

La opción de Docker fue la más trabajosa de configurar. Tuve que resolver issues con la descarga del LLM dentro del container (timeout por tamaño de modelo) y problemas de permisos para escribir el cache de Hugging Face.

---

## 4. RESULTADOS Y EVALUACIÓN

### 4.1 Métricas de Clasificación

Los resultados se obtuvieron evaluando el modelo en un conjunto de test de 2,000 casos (20% del dataset total), estratificado para mantener la proporción de churn.

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Accuracy | 0.812 | 81.2% de predicciones correctas |
| Precision | 0.531 | De los marcados como "churn", 53% realmente lo hicieron |
| Recall | 0.649 | De los que hicieron churn, detectamos 65% |
| F1-Score | 0.584 | Media armónica de precision y recall |
| ROC-AUC | 0.841 | Capacidad de discriminación entre clases |

El ROC-AUC de 0.841 es el resultado más importante porque es robusto al desbalance de clases. Este valor está en línea con estudios académicos similares que reportan valores entre 0.80-0.85 para este problema.

La precision de 53% puede parecer baja, pero hay que contextualizarla. En un escenario de negocio donde el costo de perder un cliente ($5,000 LTV) es mucho mayor que el costo de una campaña de retención innecesaria ($500), un modelo que sacrifica precision por recall es óptimo.

### 4.2 Análisis de la Matriz de Confusión

La matriz de confusión muestra la distribución de predicciones:

```
                Predicción
              No Churn  Churn
Real      ┌─────────────────────
No Churn  │  1360      233
Churn     │   143      264
```

Analizando estos números:

- **True Negatives (1360)**: Clientes que no hicieron churn y predijimos correctamente. Este es el caso más común y el modelo lo maneja bien.

- **False Positives (233)**: Clientes que NO iban a hacer churn pero los marcamos en riesgo. Esto representa el 14.6% de los no-churners. El costo es una campaña innecesaria por cliente, estimado en $500.

- **False Negatives (143)**: Clientes que SÍ hicieron churn pero no los detectamos. Este es el error más costoso: perdemos el cliente completo ($5,000 LTV). Representa el 35% de los churners - todavía hay margen de mejora aquí.

- **True Positives (264)**: Clientes en riesgo que detectamos correctamente. Estos son nuestras oportunidades de retención.

### 4.3 Curvas de Evaluación

La curva ROC muestra el trade-off entre True Positive Rate y False Positive Rate a diferentes umbrales:

[Descripción: La curva se aleja significativamente de la diagonal (random baseline), con AUC=0.841. El punto óptimo (maximiza distancia a diagonal) está aproximadamente en threshold=0.45]

La curva Precision-Recall es particularmente informativa para datasets desbalanceados. Muestra que:
- A threshold bajo (0.3): Recall alto (~86%) pero Precision baja (~36%)
- A threshold alto (0.7): Precision mejor (~68%) pero Recall bajo (~53%)
- Threshold actual (0.5): Balance razonable

Para producción, recomendaría threshold=0.4, que aumenta recall a 73% con precision de 47%. El trade-off es favorable dado el análisis de costos.

### 4.4 Análisis por Segmentos

Evalué el modelo en dos segmentos específicos:

**Clientes de Alto Valor (Balance > $100K)**
- Tamaño: 1,193 casos en test
- Accuracy: 77.2% (menor que el promedio)
- Tasa de churn real: 23.1%

La accuracy menor en este segmento sugiere que los clientes de alto valor son más difíciles de predecir. Esto podría deberse a que tienen comportamientos más diversos o a que el dataset tiene menos ejemplos de este tipo.

**Clientes Jóvenes (Age < mediana)**
- Tamaño: 1,018 casos
- Accuracy: 89.1% (mayor que el promedio)
- Tasa de churn: 8.4%

Los clientes jóvenes son más predecibles y tienen menor tasa de churn, lo cual es consistente con literatura que indica que clientes más jóvenes tienden a ser más leales.

### 4.5 Análisis de Costos y ROI

Asumiendo un banco mediano con 100,000 clientes y aplicando el modelo:

**Escenario sin modelo:**
- Churners totales: 20,400 (20.4% tasa base)
- Pérdida total: 20,400 × $5,000 = $102M

**Escenario con modelo (threshold 0.5):**
- Churners detectados: 407 × 0.649 = 264
- Clientes contactados: 497 (264 TP + 233 FP)
- Costo campañas: 497 × $500 = $248,500
- Asumiendo 40% tasa éxito retención: 106 clientes salvados
- Valor salvado: 106 × $5,000 = $530,000
- **Beneficio neto: $530K - $248K = $281,500**
- **ROI: 113%**

Este análisis asume tasas conservadoras. En la práctica, campañas bien dirigidas pueden alcanzar 60%+ de éxito en retención, lo cual mejoraría significativamente el ROI.

---

## 5. DISCUSIÓN

### 5.1 Comparación con Enfoques Alternativos

Durante el desarrollo, implementé varios modelos baseline para validar que DistilBERT aportaba valor:

| Modelo | Accuracy | ROC-AUC | F1 | Tiempo Entrenamiento |
|--------|----------|---------|-----|---------------------|
| Logistic Regression | 0.790 | 0.760 | 0.520 | 2 segundos |
| Random Forest | 0.810 | 0.820 | 0.560 | 45 segundos |
| **DistilBERT** | **0.812** | **0.841** | **0.584** | **5 minutos** |

DistilBERT muestra mejora modesta en accuracy (+0.2%) pero significativa en ROC-AUC (+2.1% vs Random Forest). El tiempo de entrenamiento es considerablemente mayor, pero es aceptable para reentrenamientos mensuales.

Un hallazgo importante: en datasets >50K, esperaría que el gap se amplíe a favor de DistilBERT. Con 10K registros, estamos en el límite donde deep learning comienza a ser competitivo con métodos tradicionales.

### 5.2 Impacto del Preprocesamiento

Realicé un estudio ablativo sobre componentes del preprocesamiento:

| Configuración | ROC-AUC | Δ vs Completo |
|---------------|---------|---------------|
| Completo | 0.841 | baseline |
| Sin normalización | 0.623 | -0.218 |
| Sin class weights | 0.798 | -0.043 |
| Sin conversión a texto | N/A | N/A |

La normalización es crítica (drop del 26% sin ella). Los class weights agregan 4.3% de performance - moderado pero valioso. La conversión a texto es necesaria para usar DistilBERT, por lo que no pude evaluarla independientemente.

### 5.3 Limitaciones del Trabajo

Es importante reconocer limitaciones explícitamente:

**Tamaño del dataset**: Con solo 10K registros, hay riesgo de overfitting. Los resultados en test (2K casos) son estadísticamente significativos pero idealmente se validarían con más datos.

**Features estáticas**: El modelo no considera evolución temporal. Un cliente cuyo balance cayó 50% en el último mes tiene mucho mayor riesgo, pero esa información no está disponible en este dataset.

**Generalización geográfica**: El dataset es de un banco europeo. Patrones de churn pueden diferir significativamente en otras regiones.

**Explicabilidad limitada**: Aunque el sistema conversacional ayuda, el modelo en sí es una caja negra. No puedo decir con certeza POR QUÉ predijo churn para un cliente específico.

**Precision moderada**: Con 53% de precision, casi la mitad de las alertas son falsas. Esto podría generar "alarm fatigue" si el equipo de retención recibe muchos falsos positivos.

### 5.4 Trabajo Relacionado

Mi enfoque se relaciona con varias líneas de investigación:

**Transformers para datos tabulares**: Huang et al. (2020) propusieron TabTransformer, que usa attention mechanisms específicamente diseñados para features categóricas. Mi enfoque de convertir a texto es más simple pero menos eficiente.

**Class imbalance**: El uso de class weights es estándar, pero alternativas como SMOTE (Synthetic Minority Over-sampling) podrían ser interesantes de explorar.

**Explicabilidad**: SHAP (SHapley Additive exPlanations) es el método más citado para explicar predicciones de modelos negros. No lo implementé por restricciones de tiempo, pero sería una extensión natural.

---

## 6. CONCLUSIONES

Este trabajo demostró que es posible construir un sistema de predicción de churn que combina performance técnica competitiva con accesibilidad para usuarios no técnicos. El modelo alcanzó un ROC-AUC de 0.841, comparable con resultados publicados en literatura académica, y el análisis de ROI muestra viabilidad económica clara.

Más allá de las métricas, el aspecto más valioso del proyecto fue el aprendizaje sobre el proceso completo de desarrollo de sistemas de ML. Aspectos que no se ven en papers pero son críticos en práctica:

- El preprocesamiento consume 60%+ del tiempo de desarrollo
- Los hiperparámetros "estándar" de la literatura no siempre funcionan
- La documentación y deployment son tan importantes como el modelo
- El trade-off entre performance y usabilidad es real y debe resolverse caso por caso

Si tuviera que empezar de nuevo, consideraría:
- Usar un dataset más grande (>50K) para aprovechar mejor deep learning
- Implementar ensemble de DistilBERT + XGBoost para mejor performance
- Agregar SHAP values desde el inicio para explicabilidad
- Hacer A/B testing con usuarios reales para validar utilidad de la interfaz conversacional

El código completo está disponible en GitHub (github.com/CuchoLeo/Fuga) bajo licencia MIT, con documentación detallada para facilitar reproducibilidad y extensión.

---

## 7. REFERENCIAS

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv preprint arXiv:1810.04805*.

2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

3. Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). TabTransformer: Tabular Data Modeling Using Contextual Embeddings. *arXiv preprint arXiv:2012.06678*.

4. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in neural information processing systems*, 30.

6. Zhao, Y., Li, B., Li, X., Liu, W., & Ren, S. (2019). Customer Churn Prediction Using Improved One-Class Support Vector Machines. *Advanced Data Mining and Applications: 11th International Conference*.

7. Kumar, A., & Ravi, V. (2020). Customer churn prediction in telecom using machine learning in big data platform. *Journal of Big Data*, 7(1), 1-18.

---

**NOTA:** Este trabajo fue desarrollado de manera individual como proyecto final del curso Tópicos Avanzados en Inteligencia Artificial 2. Se utilizaron herramientas de asistencia de IA durante el proceso de desarrollo y documentación, principalmente para generación de código boilerplate y estructuración de documentos.

## 8. RECOMENDACIONES Y TRABAJO FUTURO

### 8.1 Mejoras Inmediatas para Producción

Si este sistema fuera a desplegarse en un entorno de producción real, hay varias mejoras que implementaría en los próximos 3-6 meses.

Lo primero sería ajustar el threshold de decisión. Actualmente está en 0.5 (default), pero el análisis de costos sugiere que 0.4 sería más óptimo. Esto aumentaría el recall del 65% al 73%, detectando 33 churners adicionales por cada 2,000 clientes a cambio de solo 50 falsos positivos más. El trade-off es claramente favorable.

También integraría el sistema con el CRM existente. Durante el desarrollo, la API está diseñada para esto, pero necesitaría trabajar con el equipo de IT para mapear correctamente los campos y manejar la autenticación. La idea sería que cada mañana el equipo de retención reciba automáticamente una lista priorizada de clientes a contactar.

Un tercer aspecto crítico es el reentrenamiento. Los patrones de churn cambian con el tiempo. Recomendaría reentrenar mensualmente con los últimos 12 meses de datos, manteniendo un registro de métricas de drift para detectar cuando el modelo comienza a degradarse.

### 8.2 Extensiones a Mediano Plazo

Hay varias extensiones que mejorarían significativamente el sistema pero requieren más esfuerzo de desarrollo.

**Features temporales**: Actualmente el modelo ve un snapshot estático del cliente. Agregar tendencias (cómo ha evolucionado el balance, frecuencia de login, etc.) probablemente aumentaría el ROC-AUC a 0.87-0.90. La implementación requeriría datos históricos que no tengo actualmente.

**Predicción multi-horizonte**: En lugar de predecir solo si el cliente hará churn, predecir la probabilidad a 30, 60 y 90 días. Esto permitiría estrategias de retención diferenciadas (contacto urgente vs. engagement gradual).

**Sistema de recomendaciones**: El modelo actual dice QUÉ clientes están en riesgo, pero no QUÉ hacer. Un sistema que sugiera acciones específicas ("Ofrecer tarjeta gold reduce churn en 23% para este perfil") sería mucho más valioso. Requeriría datos históricos de intervenciones y sus resultados.

### 8.3 Investigación Futura

Desde una perspectiva académica, hay varias preguntas interesantes que podrían explorarse:

¿Puede un modelo multimodal que combine datos tabulares, texto de interacciones con soporte, y análisis de sentimiento mejorar las predicciones? Mi hipótesis es que sí, especialmente si se captura frustración del cliente en tickets de soporte, pero la implementación sería compleja.

¿Graph Neural Networks capturan mejor las relaciones entre clientes? Si asumimos que clientes similares tienden a tener comportamientos similares, representar la base de clientes como un grafo podría revelar patrones que modelos tradicionales pierden.

Por último, ¿Reinforcement Learning para optimización de estrategias de retención? En lugar de predecir churn pasivamente, un agente que aprenda qué acciones maximizan retención considerando restricciones de presupuesto sería el siguiente nivel.

---

## 9. LECCIONES APRENDIDAS

### 9.1 Técnicas

Algunas lecciones técnicas que me llevé de este proyecto:

El preprocesamiento importa MUCHO más de lo que esperaba. Inicialmente asumí que DistilBERT manejaría automáticamente variaciones en escala de features, pero sin normalización el modelo simplemente no convergía. Este tipo de detalles no aparecen en papers pero son críticos en la práctica.

Los class weights son una herramienta poderosa pero requieren tuning cuidadoso. Intenté ratios de 2:1, 3:1, 4:1 y 5:1. El óptimo (3.9:1) da un balance razonable, pero 5:1 generaba demasiados falsos positivos.

Un época de entrenamiento es suficiente para datasets pequeños. Esta fue contra-intuitiva - en deep learning normalmente se entrena por 10-100 épocas. Pero con solo 8K ejemplos de entrenamiento, el modelo memorizaba después de la primera época.

### 9.2 Ingeniería de Software

Más allá de machine learning, aprendí mucho sobre desarrollo de sistemas de software:

La documentación debe escribirse simultáneamente con el código, no después. Intenté dejarla para el final y me di cuenta de que había olvidado por qué tomé ciertas decisiones. Documentar mientras desarrollo fue mucho más efectivo.

Docker es complicado pero vale la pena. Me tomó 2 días resolver problemas de permisos y timeouts al descargar el LLM dentro del container, pero una vez funcionando, el deployment se volvió trivial.

FastAPI es excepcional. La validación automática de tipos me salvó de muchos bugs, y la documentación con Swagger es tan buena que no necesité escribir documentación adicional de la API.

### 9.3 Producto y Negocio

Quizás lo más valioso fueron lecciones sobre cómo desarrollar productos de ML que realmente se usen:

La interfaz conversacional fue la mejor decisión del proyecto. Inicialmente pensé en solo proveer una API REST, pero agregar el chat hizo el sistema accesible para no-técnicos. La diferencia entre "haz un POST a /predict con estos campos JSON" y "escribe tu pregunta en español" es enorme.

El ROI debe ser claro desde el inicio. Pasé tiempo calculando costos y beneficios no porque sea crítico para el informe académico, sino porque en una implementación real, si no puedes justificar el ROI, el proyecto no se aprueba.

Los falsos negativos son más costosos que los falsos positivos en este dominio. Esta es una decisión de negocio, no técnica. En otros contextos (e.g., detección de spam), podría ser al revés.

---

## 10. ANEXOS

### 10.1 Especificaciones Técnicas

**Hardware utilizado para desarrollo:**
- MacBook Air M1, 8GB RAM
- Sin GPU (todo en CPU)
- Almacenamiento: ~5GB para modelos y datos

**Software y versiones:**
- Python 3.10
- PyTorch 2.0.1
- Transformers 4.57.1
- FastAPI 0.104.0
- Scikit-learn 1.3.0

**Tiempos de ejecución:**
- Entrenamiento completo: ~5 minutos
- Inferencia (batch de 32): ~1.8 segundos
- Cold start de la API: ~15 segundos (carga de LLM)
- Query al agente conversacional: ~1.5 segundos

### 10.2 Estructura del Repositorio

```
Fuga/
├── train_churn_prediction.py      # Entrenamiento del modelo (427 líneas)
├── churn_chat_api.py               # API REST + LLM (565 líneas)
├── run_local.py                    # Script de ejecución (101 líneas)
├── chat_interface.html             # Interfaz web
├── Churn_Modelling.csv            # Dataset (no incluido en Git)
├── requirements.txt                # Dependencias
├── Dockerfile                      # Container Docker
├── docker-compose.yml             # Orquestación
│
├── tests/                          # Suite de evaluación
│   ├── test_models.py             # Tests exhaustivos (572 líneas)
│   ├── generate_report.py         # Generador de reportes
│   └── run_tests.sh               # Automatización
│
├── churn_model/                    # Modelo entrenado
│   ├── model.safetensors          # Pesos (268 MB)
│   ├── config.json
│   ├── tokenizer files
│   └── preprocessing_artifacts.pkl
│
└── Documentación/
    ├── DOCUMENTACION_CODIGO.md    # Explicación línea por línea
    ├── DOCUMENTACION_MODELOS.md   # Decisiones técnicas
    ├── DESPLIEGUE_GCP.md          # Guía cloud
    └── README.md                   # Documentación principal
```

### 10.3 Comandos de Ejecución

**Entrenamiento:**
```bash
python train_churn_prediction.py
# Output: churn_model/ con todos los artefactos
```

**Ejecución local:**
```bash
python run_local.py
# Servidor en http://localhost:8000
```

**Tests:**
```bash
cd tests
./run_tests.sh
# Genera test_results/ con visualizaciones y métricas
```

**Docker:**
```bash
docker-compose up --build
# Servidor en http://localhost:8000
```

### 10.4 Ejemplos de Uso de la API

**Predicción individual:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 42,
    "Tenure": 5,
    "Balance": 125000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 95000
  }'

# Response:
{
  "churn_probability": 0.73,
  "prediction": "CHURN",
  "confidence": "HIGH"
}
```

**Chat conversacional:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Muéstrame los 5 clientes con mayor riesgo"
  }'
```

### 10.5 Resultados Completos de Evaluación

Todos los resultados están disponibles en `test_results/`:
- `informe_completo.html`: Reporte interactivo con todas las visualizaciones
- `metrics.json`: Métricas en formato estructurado
- `confusion_matrix.png`: Visualización de la matriz de confusión
- `roc_curve.png`: Curva ROC
- `precision_recall_curve.png`: Curva PR
- `threshold_analysis.json`: Performance a diferentes umbrales

---

## 11. IMPACTO Y APLICABILIDAD

### 11.1 Transferibilidad a Otros Dominios

Aunque este trabajo se enfoca en churn bancario, la arquitectura es aplicable a otros problemas similares:

**Telecomunicaciones:** El churn en telcos tiene patrones similares. Solo requeriría reentrenar con features específicas del dominio (minutos consumidos, datos, llamadas al soporte).

**SaaS y Suscripciones:** Empresas como Netflix o Spotify enfrentan el mismo problema. Las features serían diferentes (tiempo de uso, contenido consumido), pero la arquitectura se mantiene.

**Retail:** Predecir clientes que dejarán de comprar. Requeriría features transaccionales (recencia, frecuencia, valor monetario - modelo RFM).

La clave es que la metodología (Transformer para clasificación + LLM para interpretación) es agnóstica al dominio específico.

### 11.2 Contribución a la Democratización de IA

Un aspecto que me parece importante destacar es cómo este proyecto contribuye a hacer IA más accesible:

**Código completamente open-source:** Todo está en GitHub bajo licencia MIT. Cualquier organización puede usarlo sin costo.

**Documentación exhaustiva:** ~20,000 palabras de documentación. No solo explico QUÉ hace el código, sino POR QUÉ tomé cada decisión.

**Opciones de deployment flexibles:** Local (para testing), Docker (para consistencia), Cloud (para producción). No todos tienen los mismos recursos.

**Sin requerimientos de GPU:** El sistema completo corre en una laptop estándar. Esto es crítico para organizaciones pequeñas.

### 11.3 Consideraciones Éticas

Finalmente, es importante considerar implicaciones éticas de sistemas como este:

**Sesgo algorítmico:** El modelo podría perpetuar sesgos presentes en datos históricos. Por ejemplo, si históricamente ciertos grupos demográficos recibieron peor servicio y por eso tienen mayor churn, el modelo podría penalizar a esos grupos. Revisé las tasas de churn por género y geografía y no encontré sesgos evidentes, pero un análisis más profundo sería apropiado.

**Privacidad:** El sistema procesa datos sensibles de clientes. En una implementación real, debe cumplir con regulaciones como GDPR. La arquitectura permite deployment on-premise, lo cual ayuda a mantener datos bajo control de la organización.

**Transparencia:** Los clientes tienen derecho a saber si están siendo evaluados por un algoritmo. Las organizaciones que implementen esto deberían ser transparentes sobre su uso.

**Automatización vs. Asistencia:** El sistema está diseñado para asistir decisiones humanas, no reemplazarlas. Las predicciones deben ser revisadas por expertos antes de tomar acciones.

---

## 12. CONCLUSIÓN FINAL

Este proyecto demuestra que es viable desarrollar sistemas de predicción de churn que combinen performance técnica sólida con accesibilidad práctica. El modelo alcanzó un ROC-AUC de 0.841, comparable con resultados publicados en conferencias académicas, mientras que el análisis de ROI indica viabilidad económica clara con retorno del 113% en el primer año.

Más allá de las métricas, lo que considero más valioso es haber abordado el problema de manera integral. No solo entrené un modelo, sino que construí un sistema completo que puede desplegarse, mantenerse y usarse en condiciones reales. Esta perspectiva end-to-end es crítica para que proyectos de ML generen valor real.

Las lecciones aprendidas durante el desarrollo serán aplicables a futuros proyectos. En particular, la importancia del preprocesamiento cuidadoso, el diseño de interfaces accesibles, y la consideración de trade-offs de negocio desde las etapas tempranas del desarrollo.

Si tuviera que resumir una lección central: los modelos de ML son solo una pieza del sistema. El deployment, la documentación, la interfaz de usuario, y la integración con procesos de negocio existentes son igualmente importantes para el éxito del proyecto.

El código está disponible públicamente en GitHub (github.com/CuchoLeo/Fuga) con documentación exhaustiva. Espero que sirva como referencia útil para otros estudiantes e investigadores trabajando en problemas similares.

---

**Agradecimientos:** Agradezco la asesoría del profesor [nombre] y los comentarios de compañeros durante el desarrollo de este trabajo. También reconozco el uso de herramientas de asistencia de IA (Claude Code de Anthropic) para generación de código boilerplate, estructuración de documentos, y debugging. El diseño, implementación, evaluación y análisis representan trabajo original del autor.

---

**Fecha de finalización:** Noviembre 2025  
**Palabras:** ~6,500  
**Código:** ~2,800 líneas Python  
**Documentación:** ~20,000 palabras totales  

---

*"The best model is the one that actually gets used."* - Anónimo
