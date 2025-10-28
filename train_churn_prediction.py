import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import shutil

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("="*70)
print("SISTEMA DE PREDICCIÓN DE CHURN - CLIENTES ALTO VALOR")
print("="*70)

# ============================================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# ============================================================================

def load_and_preprocess_data(csv_path):
    """
    Carga el dataset de Kaggle: Bank Customer Churn
    Esperado: Churn_Modelling.csv
    """
    print(f"\n📊 Cargando dataset desde: {csv_path}")
    
    # Leer CSV
    df = pd.read_csv(csv_path)
    print(f"✓ Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
    
    # Mostrar información del dataset
    print(f"\nColumnas disponibles: {list(df.columns)}")
    print(f"\nDistribución de Churn:")
    print(df['Exited'].value_counts())
    print(f"Tasa de Churn: {df['Exited'].mean()*100:.2f}%")
    
    # Filtrar clientes de alto valor (>$100,000 en Balance)
    if 'Balance' in df.columns:
        high_value_mask = df['Balance'] > 100000
        print(f"\n💰 Clientes alto valor (Balance > $100k): {high_value_mask.sum()} ({high_value_mask.mean()*100:.1f}%)")
        print(f"Churn rate alto valor: {df[high_value_mask]['Exited'].mean()*100:.2f}%")
    
    # Eliminar columnas no relevantes
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    # Codificar variables categóricas
    label_encoders = {}
    categorical_cols = ['Geography', 'Gender']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Separar features y target
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Normalizar features numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convertir a texto para el modelo de lenguaje
    # Creamos descripciones textuales de cada cliente
    feature_names = X.columns.tolist()
    
    print(f"\n✓ Preprocesamiento completado")
    print(f"Features: {len(feature_names)}")
    
    return X_scaled, y.values, feature_names, scaler, label_encoders

def create_text_from_features(X, feature_names, y=None):
    """
    Convierte features numéricas en descripciones textuales
    para entrenar el modelo de lenguaje
    """
    texts = []
    labels = []
    
    for idx in range(len(X)):
        # Crear descripción del cliente
        feature_values = X[idx]
        text_parts = ["Cliente:"]
        
        for name, value in zip(feature_names, feature_values):
            text_parts.append(f"{name}={value:.2f}")
        
        text = " ".join(text_parts)
        
        # Si tenemos labels, agregar la predicción esperada
        if y is not None:
            label_text = "CHURN" if y[idx] == 1 else "RETIENE"
            text += f" -> Predicción: {label_text}"
            labels.append(y[idx])
        
        texts.append(text)
    
    return texts, labels if y is not None else texts

# ============================================================================
# 2. CONFIGURACIÓN DEL MODELO
# ============================================================================

# Ruta del dataset (ajustar según tu ubicación)
DATA_PATH = "Churn_Modelling.csv"

# Verificar si existe el archivo
if not Path(DATA_PATH).exists():
    print(f"\n⚠️  ADVERTENCIA: No se encontró el archivo {DATA_PATH}")
    print("\nPara descargar el dataset:")
    print("1. Ve a: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling")
    print("2. Descarga 'Churn_Modelling.csv'")
    print("3. Colócalo en el mismo directorio que este script")
    print("\nO usa Kaggle API:")
    print("   kaggle datasets download -d shrutimechlearn/churn-modelling")
    exit(1)

# Cargar y procesar datos
X, y, feature_names, scaler, label_encoders = load_and_preprocess_data(DATA_PATH)

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📚 División de datos:")
print(f"Training: {len(X_train)} muestras")
print(f"Testing: {len(X_test)} muestras")

# Convertir a texto
train_texts, train_labels = create_text_from_features(X_train, feature_names, y_train)
test_texts, test_labels = create_text_from_features(X_test, feature_names, y_test)

# Load the model and tokenizer para clasificación
model_id = "distilbert-base-uncased"  # Modelo más ligero para clasificación
print(f"\n🤖 Cargando modelo base: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=2,  # Binario: Churn / No Churn
    torch_dtype=torch.float32
)

# Set the padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# 3. PREPARACIÓN DEL DATASET
# ============================================================================

max_length = 256  # Aumentado para acomodar features

print(f"\n⚙️  Tokenizando {len(train_texts)} ejemplos de entrenamiento...")
train_encodings = tokenizer(
    train_texts,
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

print(f"⚙️  Tokenizando {len(test_texts)} ejemplos de prueba...")
test_encodings = tokenizer(
    test_texts,
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

class ChurnDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

train_dataset = ChurnDataset(train_encodings, train_labels)
test_dataset = ChurnDataset(test_encodings, test_labels)

# ============================================================================
# 4. CONFIGURACIÓN DE ENTRENAMIENTO
# ============================================================================

output_dir = Path("churn_model").resolve()
checkpoint_dir = Path("checkpoint_churn").resolve()

# Limpiar archivos antiguos
if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)

print(f"\n📁 Directorio de salida: {output_dir}")
print(f"🖥️  CUDA disponible: {torch.cuda.is_available()}")

# Training arguments optimizados para churn prediction
training_args = TrainingArguments(
    output_dir=str(checkpoint_dir),
    num_train_epochs=1,  # Reducido a 1 época para entrenamiento rápido
    per_device_train_batch_size=32,  # Aumentado para procesar más rápido
    per_device_eval_batch_size=32,  # Aumentado para procesar más rápido
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,  # Menos logging = más rápido
    use_cpu=True,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

print("\n" + "="*70)
print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO DE PREDICCIÓN DE CHURN")
print("="*70)

# Función para calcular métricas
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ============================================================================
# 5. ENTRENAMIENTO
# ============================================================================

try:
    train_result = trainer.train()
    print("\n✅ Entrenamiento completado!")
    
    # Evaluar en test set
    print("\n📊 Evaluando modelo en conjunto de prueba...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*70)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*70)
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
except Exception as e:
    print(f"\n❌ Error durante el entrenamiento: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 6. GUARDAR MODELO
# ============================================================================

print("\n" + "="*70)
print("💾 GUARDANDO MODELO Y ARTEFACTOS")
print("="*70)

try:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    print(f"\n📦 Guardando modelo en: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Guardar scaler y encoders
    import pickle
    
    artifacts = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names
    }
    
    with open(output_dir / 'preprocessing_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print("✅ Artefactos de preprocesamiento guardados")
    
    # Verificar archivos
    print("\n" + "="*70)
    print("ARCHIVOS GUARDADOS:")
    print("="*70)
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / (1024*1024)
        print(f"✓ {f.name}: {size:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ MODELO DE PREDICCIÓN DE CHURN ENTRENADO Y GUARDADO")
    print("="*70)
    print(f"\n💡 Para usar el modelo:")
    print(f"   1. Cargar desde: {output_dir}")
    print(f"   2. Aplicar a clientes de alto valor (Balance > $100k)")
    print(f"   3. Priorizar retención de clientes con alta probabilidad de churn")
    print(f"\n💰 ROI Estimado:")
    print(f"   Costo retención = 1/5 del costo de adquisición")
    print(f"   Clientes salvados/mes: reducción de 2,500 fugas actuales")
    
except Exception as e:
    print(f"\n❌ Error guardando modelo: {e}")
    import traceback
    traceback.print_exc()

# Limpiar checkpoint directory
if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)
    print("\n🧹 Archivos temporales eliminados")

print("\n" + "="*70)
print("🎯 PRÓXIMOS PASOS RECOMENDADOS:")
print("="*70)
print("1. Validar predicciones en clientes de alto valor")
print("2. Implementar sistema de alertas tempranas")
print("3. Integrar con CRM para acciones de retención")
print("4. Monitorear métricas: Recall (capturar churners) y Precision (evitar falsos positivos)")
print("="*70)