"""
SISTEMA DE PRUEBAS COMPLETAS PARA MODELOS DE PREDICCIÓN DE CHURN
Informe Final - Magister en Inteligencia Artificial

Este script realiza:
- Evaluación exhaustiva del modelo
- Generación de métricas detalladas
- Visualizaciones profesionales
- Análisis por segmentos
- Reporte HTML completo

Autor: Sistema de Predicción de Churn
Fecha: 2025
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("🧪 SISTEMA DE PRUEBAS COMPLETAS - MODELO DE PREDICCIÓN DE CHURN")
print("="*80)
print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODEL_PATH = Path("churn_model")
DATA_PATH = Path("Churn_Modelling.csv")
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Parámetros de prueba
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 256

# ============================================================================
# 1. FUNCIONES AUXILIARES
# ============================================================================

def create_text_from_features(X, feature_names, y=None):
    """Convierte features numéricas en descripciones textuales"""
    texts = []
    labels = []

    for idx in range(len(X)):
        feature_values = X[idx]
        text_parts = ["Cliente:"]

        for name, value in zip(feature_names, feature_values):
            text_parts.append(f"{name}={value:.2f}")

        text = " ".join(text_parts)

        if y is not None:
            label_text = "CHURN" if y[idx] == 1 else "RETIENE"
            text += f" -> Predicción: {label_text}"
            labels.append(y[idx])

        texts.append(text)

    return texts, labels if y is not None else texts

class ChurnDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def save_dict_to_json(data, filepath):
    """Guarda diccionario como JSON con formato legible"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON guardado: {filepath}")

# ============================================================================
# 2. CARGA DE DATOS Y MODELO
# ============================================================================

print("\n📊 Paso 1: Cargando datos y modelo...")
print("-" * 80)

# Verificar archivos necesarios
if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset: {DATA_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo entrenado: {MODEL_PATH}")

# Cargar datos
df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")

# Filtrar clientes de alto valor
high_value_mask = df['Balance'] > 100000
df_high_value = df[high_value_mask].copy()
print(f"✓ Clientes alto valor: {len(df_high_value)} ({len(df_high_value)/len(df)*100:.1f}%)")

# Preprocesar datos
cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

# Codificar variables categóricas
label_encoders = {}
for col in ['Geography', 'Gender']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Separar features y target
X = df.drop('Exited', axis=1)
y = df['Exited']
feature_names = X.columns.tolist()

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# Cargar modelo y tokenizer
print("\n🤖 Cargando modelo entrenado...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("✓ Modelo cargado y en modo evaluación")

# Preparar datos de test
test_texts, test_labels = create_text_from_features(X_test, feature_names, y_test)
test_encodings = tokenizer(
    test_texts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

# ============================================================================
# 3. GENERACIÓN DE PREDICCIONES
# ============================================================================

print("\n🔮 Paso 2: Generando predicciones...")
print("-" * 80)

predictions = []
probabilities = []
true_labels = []

with torch.no_grad():
    for i in range(len(test_labels)):
        inputs = {
            'input_ids': test_encodings['input_ids'][i:i+1],
            'attention_mask': test_encodings['attention_mask'][i:i+1]
        }

        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

        predictions.append(pred)
        probabilities.append(probs[0][1].item())  # Probabilidad de churn
        true_labels.append(test_labels[i])

predictions = np.array(predictions)
probabilities = np.array(probabilities)
true_labels = np.array(true_labels)

print(f"✓ {len(predictions)} predicciones generadas")

# ============================================================================
# 4. CÁLCULO DE MÉTRICAS
# ============================================================================

print("\n📊 Paso 3: Calculando métricas de evaluación...")
print("-" * 80)

# Métricas básicas
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='binary', zero_division=0
)

# Métricas adicionales
roc_auc = roc_auc_score(true_labels, probabilities)
avg_precision = average_precision_score(true_labels, probabilities)

# Matriz de confusión
cm = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = cm.ravel()

# Métricas derivadas
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

# Guardar métricas
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
    "specificity": float(specificity),
    "npv": float(npv),
    "fpr": float(fpr),
    "fnr": float(fnr),
    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
}

print("\n✅ MÉTRICAS PRINCIPALES:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   Avg Precision: {avg_precision:.4f}")

print("\n📊 MATRIZ DE CONFUSIÓN:")
print(f"   TN: {tn:4d} | FP: {fp:4d}")
print(f"   FN: {fn:4d} | TP: {tp:4d}")

save_dict_to_json(metrics, RESULTS_DIR / "metrics.json")

# ============================================================================
# 5. VISUALIZACIONES
# ============================================================================

print("\n📈 Paso 4: Generando visualizaciones...")
print("-" * 80)

# 5.1 Matriz de Confusión
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Predicción', fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Matriz de confusión guardada")

# 5.2 Curva ROC
fpr_curve, tpr_curve, _ = roc_curve(true_labels, probabilities)
fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Curva ROC guardada")

# 5.3 Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(true_labels, probabilities)
fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2,
         label=f'PR curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Curva Precision-Recall', fontsize=16, fontweight='bold')
plt.legend(loc="lower left", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Curva Precision-Recall guardada")

# 5.4 Distribución de Probabilidades
fig, ax = plt.subplots(figsize=(12, 6))
plt.hist(probabilities[true_labels == 0], bins=50, alpha=0.6, label='No Churn', color='blue')
plt.hist(probabilities[true_labels == 1], bins=50, alpha=0.6, label='Churn', color='red')
plt.xlabel('Probabilidad de Churn', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Distribución de Probabilidades Predichas', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Distribución de probabilidades guardada")

# 5.5 Métricas comparativas
metrics_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Valor': [accuracy, precision, recall, f1, roc_auc]
})

fig, ax = plt.subplots(figsize=(10, 6))
bars = plt.barh(metrics_df['Métrica'], metrics_df['Valor'], color='steelblue')
plt.xlabel('Score', fontsize=12)
plt.title('Resumen de Métricas de Evaluación', fontsize=16, fontweight='bold')
plt.xlim([0, 1])
for i, (bar, val) in enumerate(zip(bars, metrics_df['Valor'])):
    plt.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'metrics_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Resumen de métricas guardado")

# ============================================================================
# 6. ANÁLISIS POR UMBRALES
# ============================================================================

print("\n🎯 Paso 5: Análisis por umbrales de decisión...")
print("-" * 80)

thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_analysis = []

for threshold in thresholds_to_test:
    preds_thresh = (probabilities >= threshold).astype(int)
    acc = accuracy_score(true_labels, preds_thresh)
    prec, rec, f1_t, _ = precision_recall_fscore_support(
        true_labels, preds_thresh, average='binary', zero_division=0
    )

    threshold_analysis.append({
        'threshold': threshold,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1_t)
    })

threshold_df = pd.DataFrame(threshold_analysis)
print("\n📊 Análisis por Umbrales:")
print(threshold_df.to_string(index=False))

# Visualizar
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(thresholds_to_test))
width = 0.2

plt.bar(x - 1.5*width, threshold_df['accuracy'], width, label='Accuracy', color='skyblue')
plt.bar(x - 0.5*width, threshold_df['precision'], width, label='Precision', color='lightgreen')
plt.bar(x + 0.5*width, threshold_df['recall'], width, label='Recall', color='salmon')
plt.bar(x + 1.5*width, threshold_df['f1_score'], width, label='F1-Score', color='gold')

plt.xlabel('Umbral de Decisión', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Métricas vs Umbral de Decisión', fontsize=16, fontweight='bold')
plt.xticks(x, [f'{t:.1f}' for t in thresholds_to_test])
plt.legend(fontsize=11)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Análisis de umbrales guardado")

save_dict_to_json(threshold_analysis, RESULTS_DIR / "threshold_analysis.json")

# ============================================================================
# 7. ANÁLISIS POR SEGMENTOS
# ============================================================================

print("\n🔍 Paso 6: Análisis por segmentos de clientes...")
print("-" * 80)

# Crear DataFrame con resultados
results_df = pd.DataFrame({
    'true_label': true_labels,
    'prediction': predictions,
    'probability': probabilities
})

# Agregar features originales
X_test_df = pd.DataFrame(X_test, columns=feature_names)
results_df = pd.concat([results_df, X_test_df], axis=1)

# Análisis por segmentos
segments_analysis = {}

# Por Balance (Alto valor vs resto)
high_value_test = results_df['Balance'] > 0  # Ya están normalizados
if high_value_test.sum() > 0:
    hv_accuracy = accuracy_score(
        results_df[high_value_test]['true_label'],
        results_df[high_value_test]['prediction']
    )
    segments_analysis['high_value_clients'] = {
        'count': int(high_value_test.sum()),
        'accuracy': float(hv_accuracy),
        'churn_rate': float(results_df[high_value_test]['true_label'].mean())
    }

# Por edad (usando percentiles)
age_median = results_df['Age'].median()
young = results_df['Age'] <= age_median
if young.sum() > 0:
    young_accuracy = accuracy_score(
        results_df[young]['true_label'],
        results_df[young]['prediction']
    )
    segments_analysis['young_clients'] = {
        'count': int(young.sum()),
        'accuracy': float(young_accuracy),
        'churn_rate': float(results_df[young]['true_label'].mean())
    }

print("\n📊 Análisis por Segmentos:")
for segment, data in segments_analysis.items():
    print(f"\n{segment}:")
    print(f"  Count: {data['count']}")
    print(f"  Accuracy: {data['accuracy']:.4f}")
    print(f"  Churn Rate: {data['churn_rate']:.4f}")

save_dict_to_json(segments_analysis, RESULTS_DIR / "segments_analysis.json")

# ============================================================================
# 8. ANÁLISIS DE ERRORES
# ============================================================================

print("\n❌ Paso 7: Análisis de errores...")
print("-" * 80)

# Falsos Positivos
fp_mask = (predictions == 1) & (true_labels == 0)
fp_df = results_df[fp_mask].copy()
fp_df['error_type'] = 'False Positive'

# Falsos Negativos
fn_mask = (predictions == 0) & (true_labels == 1)
fn_df = results_df[fn_mask].copy()
fn_df['error_type'] = 'False Negative'

print(f"Falsos Positivos: {len(fp_df)} ({len(fp_df)/len(results_df)*100:.2f}%)")
print(f"Falsos Negativos: {len(fn_df)} ({len(fn_df)/len(results_df)*100:.2f}%)")

# Guardar ejemplos de errores
errors_df = pd.concat([fp_df.head(10), fn_df.head(10)])
errors_df.to_csv(RESULTS_DIR / 'error_examples.csv', index=False)
print("✓ Ejemplos de errores guardados")

# ============================================================================
# 9. REPORTE CLASIFICACIÓN COMPLETO
# ============================================================================

print("\n📄 Paso 8: Generando reporte de clasificación...")
print("-" * 80)

class_report = classification_report(
    true_labels,
    predictions,
    target_names=['No Churn', 'Churn'],
    output_dict=True
)

print("\n" + classification_report(
    true_labels,
    predictions,
    target_names=['No Churn', 'Churn']
))

save_dict_to_json(class_report, RESULTS_DIR / "classification_report.json")

# ============================================================================
# 10. RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("✅ PRUEBAS COMPLETADAS - RESUMEN FINAL")
print("="*80)

summary = {
    "timestamp": datetime.now().isoformat(),
    "model_path": str(MODEL_PATH),
    "test_samples": len(test_labels),
    "key_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc)
    },
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    },
    "recommendations": []
}

# Generar recomendaciones
if recall < 0.6:
    summary["recommendations"].append("⚠️ Recall bajo: Considerar ajustar umbral o reentrenar con más peso en clase minoritaria")
if precision < 0.6:
    summary["recommendations"].append("⚠️ Precision baja: Muchos falsos positivos. Revisar features o aumentar umbral")
if f1 > 0.7:
    summary["recommendations"].append("✅ F1-Score bueno: Modelo balanceado entre precision y recall")
if roc_auc > 0.8:
    summary["recommendations"].append("✅ Excelente capacidad de discriminación (ROC-AUC > 0.8)")

save_dict_to_json(summary, RESULTS_DIR / "test_summary.json")

print("\n📊 ARCHIVOS GENERADOS:")
print(f"   Directorio: {RESULTS_DIR}/")
for f in sorted(RESULTS_DIR.glob('*')):
    print(f"   ✓ {f.name}")

print("\n💡 RECOMENDACIONES:")
for rec in summary["recommendations"]:
    print(f"   {rec}")

print("\n" + "="*80)
print("🎯 PRUEBAS FINALIZADAS CON ÉXITO")
print("="*80)
print(f"\n📁 Resultados guardados en: {RESULTS_DIR.absolute()}")
print("\nPróximos pasos:")
print("   1. Revisar visualizaciones generadas")
print("   2. Analizar ejemplos de errores")
print("   3. Considerar ajustes según recomendaciones")
print("   4. Incluir resultados en informe final")
print("="*80)
