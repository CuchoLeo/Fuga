# Cuando un Chatbot Predice Quién Abandonará tu Banco: Historia de un Sistema de IA que Combina Deep Learning con Conversación Natural

**Por Víctor Rodríguez**
*Magister en Inteligencia Artificial*

---

## Resumen

El abandono de clientes cuesta a los bancos millones cada año. Este artículo describe el desarrollo de "Churnito", un sistema que usa DistilBERT para predecir qué clientes están en riesgo de irse, combinado con un agente conversacional que permite consultas en español. El resultado: 84% de precisión en detectar riesgo y un ROI del 113%. Pero más importante que las métricas es cómo hacer que equipos no técnicos realmente usen el sistema.

---

## El Problema: Cuando Tus Clientes Se Van

Hace unos meses me topé con una estadística que me hizo replantear todo mi proyecto de tesis: un banco mediano pierde aproximadamente $100 millones al año por clientes que se van. No es que los bancos no sepan que esto pasa - lo saben perfectamente. El problema es que cuando se dan cuenta, el cliente ya cerró su cuenta y se fue a la competencia.

Lo interesante es que retener a un cliente existente cuesta entre $100 y $200, mientras que adquirir uno nuevo cuesta $500-$1,200. La matemática es simple: si pudieras saber CON ANTICIPACIÓN qué clientes van a irse, podrías hacer algo al respecto y ahorrarte una fortuna.

Eso es exactamente lo que me propuse hacer.

## La Idea: No Solo Predecir, Sino Conversar

Revisando papers académicos sobre predicción de churn, encontré docenas de modelos con accuracies del 80-90%. Impresionante, ¿no? El problema es que ninguno respondía la pregunta que yo me haría si fuera gerente de retención: "¿Y cómo diablos uso esto?"

La mayoría de estos sistemas requieren que hagas POST requests a una API con JSONs perfectamente formateados. Genial para desarrolladores, terrible para el equipo de marketing que solo quiere saber a quién llamar hoy.

Entonces decidí que mi sistema tenía que cumplir dos criterios:
1. Predicciones técnicamente sólidas (obvio)
2. Una interfaz tan simple como escribir: "Muéstrame los 10 clientes con mayor riesgo"

Y así nació Churnito.

## La Implementación: Dos Cerebros Son Mejor Que Uno

### El Cerebro Analítico: DistilBERT

Para las predicciones use DistilBERT, que es básicamente BERT pero más pequeño y rápido. La decisión fue práctica: la mayoría de las empresas no tienen GPUs dedicadas, y DistilBERT corre perfectamente en una laptop normal.

Aquí viene la parte rara: DistilBERT fue diseñado para procesar texto, pero los datos de clientes son números en tablas (edad, balance, productos contratados, etc.). ¿La solución? Convertir todo a texto:

```
"Cliente: CreditScore=650 Age=42 Balance=120000 Tenure=5 IsActiveMember=0"
```

Sí, es poco ortodoxo. Sí, funciona.

### El Cerebro Conversacional: Qwen2.5

Para la parte conversacional probé varios modelos. Inicialmente quería usar Llama 3.2, pero requiere autenticación de Hugging Face (barrera innecesaria). Terminé usando Qwen2.5-1.5B porque:
- Es completamente open-source
- Habla español decentemente
- Corre en CPU sin quemar mi laptop

El truco está en el prompt. Después de probar prompts elaborados de 200 palabras, descubrí que uno simple funciona mejor:

```
"Eres Churnito, experto en análisis de churn bancario.
Responde profesionalmente basándote en datos."
```

A veces menos es más.

## Los Resultados: Mejor De Lo Esperado

Entrenécon 8,000 clientes y probé con 2,000. Los números:

- **ROC-AUC: 84.1%** - Esta es la métrica que importa. Significa que el modelo puede distinguir entre un cliente que se va y uno que se queda en 84% de los casos.
- **Accuracy: 81.2%** - 8 de cada 10 predicciones son correctas.
- **Recall: 64.9%** - Detectamos casi 2 de cada 3 clientes que realmente se van.

¿Suena perfecto? No lo es. El 35% de los churners todavía se nos escapan. Pero comparado con no hacer nada (0% detectados), es un avance significativo.

### La Matriz de Confusión Traducida al Español Normal

De 2,000 predicciones:
- 1,360 clientes que NO se fueron y predijimos correctamente ✓
- 264 clientes en riesgo que detectamos ✓
- 233 falsas alarmas (campaña innecesaria, $500 c/u)
- 143 clientes que perdimos sin detectar ($5,000 c/u) ✗

Total costo de errores: $715,000 por clientes perdidos + $116,500 por campañas innecesarias = $831,500.

Total ahorro por clientes salvados: Asumiendo 40% de éxito en retención de los 264 detectados = 106 clientes × $5,000 = $530,000.

Costo del sistema: $248,500 en campañas dirigidas.

**ROI: 113%**

No está mal para un prototipo académico.

## Lo Que Aprendí (Y Lo Que No Funcionó)

### Cosas Que Funcionaron

**Class weights agresivos:** El dataset tenía 80% "no churn" y 20% "churn". Sin ajustes, el modelo simplemente predecía "no churn" para todos. Le di 4x más peso a los casos de churn y boom, comenzó a funcionar.

**Una sola época de entrenamiento:** Contra-intuitivo pero cierto. Más épocas causaban overfitting. El modelo memorizado todos los ejemplos en lugar de aprender patrones generales.

**La interfaz conversacional:** Esta fue la mejor decisión. Ver a alguien del equipo de marketing escribir "¿Cuántos clientes de alto valor están en riesgo?" y obtener una respuesta coherente en segundos fue el momento "aha" del proyecto.

### Cosas Que NO Funcionaron

**Llama models:** Requieren autenticación. Demasiada fricción para algo que debería ser plug-and-play.

**Prompts elaborados:** Pensé que descripciones largas y detalladas mejorarían las respuestas del LLM. Wrong. Prompts concisos funcionan mejor.

**500 tokens de respuesta:** El LLM escribía biblias. Reducir a 150 tokens lo forzó a ser conciso y redujo latencia de 4s a 1.5s.

## El Caso de Uso Real: Centro de Atención al Cliente

Imagina esto: un cliente llama, claramente molesto. El agente ve en su pantalla:

```
⚠️ Riesgo ALTO (87%)

Contexto:
- Cliente VIP ($145K balance)
- Sin actividad digital en 45 días
- Redujo saldo 15% último mes
- 3 quejas previas sobre fees

Acción sugerida: Eliminar fees + tarjeta premium
Probabilidad retención: 72%
```

El agente ahora tiene contexto accionable en segundos. Puede ofrecer algo específico en lugar de un genérico "¿hay algo en lo que pueda ayudarle?"

Esa es la diferencia entre un modelo académico y un sistema usable.

## Democratizando la IA (Sin Sonar Cursi)

Algo que me frustra de muchos proyectos de ML es que terminan siendo accesibles solo para empresas con presupuestos millonarios. Salesforce Einstein cuesta $2,000+ al mes. AWS SageMaker, $1,500+.

Churnito cuesta $50/mes en Google Cloud Run (tier gratuito es suficiente para comenzar).

Todo el código está en GitHub bajo licencia MIT. La documentación completa tiene 20,000+ palabras explicando cada decisión. No solo QUÉ hace el código, sino POR QUÉ.

¿Por qué? Porque si solo unos pocos pueden usar IA avanzada, estamos desperdiciando su potencial.

## Limitaciones (Sí, Tiene)

Sería deshonesto no mencionar los problemas:

1. **Dataset pequeño:** 10K registros es poco para deep learning. Idealmente necesitaría 100K+.

2. **Features estáticas:** No considera historial. Un cliente cuyo balance cayó 50% en un mes tiene mucho mayor riesgo, pero esa info no está disponible.

3. **Precision del 53%:** Casi la mitad de las alertas son falsas. Eso puede generar "alarm fatigue" en el equipo.

4. **Sin explicabilidad:** Puedo decir QUÉ predice el modelo pero no POR QUÉ. SHAP values serían el próximo paso.

## ¿Y Ahora Qué?

Si fuera a llevarlo a producción real (y varias personas me han preguntado sobre esto), estos serían los próximos pasos:

**Corto plazo (1-3 meses):**
- Bajar threshold a 0.4 para aumentar recall a 73%
- Integrar con CRM para automatizar alertas
- Implementar A/B testing para validar efectividad real

**Mediano plazo (6 meses):**
- Agregar features temporales (tendencias de balance, actividad)
- Reentrenamiento mensual automático
- Dashboard ejecutivo con métricas en tiempo real

**Largo plazo (1 año):**
- Sistema de recomendaciones: no solo QUÉ sino CÓMO retener
- Predicción multi-horizonte (30, 60, 90 días)
- Causal inference para identificar causas raíz

## Conclusión: Los Modelos Buenos Son Los Que Se Usan

Puedo entrenar un modelo con 95% de accuracy que nadie use, o uno con 81% que el equipo consulte cada mañana. El segundo es infinitamente más valioso.

Este proyecto me enseñó que ML en el mundo real no es solo sobre algoritmos. Es sobre:
- Interfaces que humanos reales puedan usar
- ROI que justifique la inversión
- Documentación que permita mantenimiento
- Deployment que no requiera un equipo de DevOps

Las métricas son importantes, pero la usabilidad es crítica.

Si estás considerando un proyecto similar, mi consejo: comienza con la interfaz. Pregúntale a quien lo usará cómo lo idealizaría. Luego construye el modelo para que encaje en esa visión, no al revés.

---

**Código y Documentación:**
GitHub: github.com/CuchoLeo/Fuga (MIT License)

**Para Empresas Interesadas:**
El sistema puede adaptarse a telecomunicaciones, SaaS, retail, o cualquier negocio con clientes recurrentes. Todo el código está disponible públicamente.

**Nota Final:**
Este proyecto fue desarrollado como tesis de magister. Se utilizaron herramientas de asistencia de IA (Claude Code) para generación de código boilerplate y debugging, pero el diseño, implementación, evaluación y análisis representan trabajo original.

---

*Víctor Rodríguez es estudiante de Magister en Inteligencia Artificial, especializado en NLP aplicado a problemas de negocio. Contacto: GitHub @CuchoLeo*

---

*"Build things that people actually use, not things that sound impressive in papers."* - Lección aprendida en este proyecto
