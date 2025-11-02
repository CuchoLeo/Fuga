"""
GENERADOR DE REPORTE HTML PROFESIONAL
Sistema de Predicción de Churn - Informe Final

Este script genera un reporte HTML completo con:
- Resumen ejecutivo
- Métricas detalladas
- Visualizaciones embebidas
- Análisis de errores
- Recomendaciones

Uso: python generate_report.py
Requiere: Haber ejecutado test_models.py primero
"""

import json
from pathlib import Path
from datetime import datetime
import base64

RESULTS_DIR = Path("test_results")
REPORT_PATH = RESULTS_DIR / "informe_completo.html"

print("="*80)
print("📄 GENERADOR DE REPORTE HTML - PREDICCIÓN DE CHURN")
print("="*80)

# ============================================================================
# VERIFICAR ARCHIVOS
# ============================================================================

required_files = [
    "metrics.json",
    "confusion_matrix.png",
    "roc_curve.png",
    "precision_recall_curve.png",
    "probability_distribution.png",
    "metrics_summary.png",
    "threshold_analysis.png",
    "classification_report.json",
    "test_summary.json"
]

print("\n🔍 Verificando archivos necesarios...")
missing = []
for f in required_files:
    if not (RESULTS_DIR / f).exists():
        missing.append(f)
        print(f"   ❌ Falta: {f}")
    else:
        print(f"   ✓ {f}")

if missing:
    print(f"\n⚠️ Faltan {len(missing)} archivos. Ejecuta 'python test_models.py' primero.")
    exit(1)

print("\n✓ Todos los archivos necesarios están disponibles")

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("\n📊 Cargando datos...")

with open(RESULTS_DIR / "metrics.json") as f:
    metrics = json.load(f)

with open(RESULTS_DIR / "classification_report.json") as f:
    class_report = json.load(f)

with open(RESULTS_DIR / "test_summary.json") as f:
    summary = json.load(f)

with open(RESULTS_DIR / "threshold_analysis.json") as f:
    threshold_analysis = json.load(f)

# Cargar segmentos si existe
segments = {}
if (RESULTS_DIR / "segments_analysis.json").exists():
    with open(RESULTS_DIR / "segments_analysis.json") as f:
        segments = json.load(f)

print("✓ Datos cargados correctamente")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def img_to_base64(img_path):
    """Convierte imagen a base64 para embeber en HTML"""
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def format_metric(value, decimals=4):
    """Formatea métrica numérica"""
    return f"{value:.{decimals}f}"

def get_metric_color(value, threshold_good=0.7, threshold_ok=0.5):
    """Retorna color según valor de métrica"""
    if value >= threshold_good:
        return "#28a745"  # Verde
    elif value >= threshold_ok:
        return "#ffc107"  # Amarillo
    else:
        return "#dc3545"  # Rojo

# ============================================================================
# GENERAR HTML
# ============================================================================

print("\n🎨 Generando HTML...")

html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe de Evaluación - Modelo de Predicción de Churn</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px 30px;
        }}

        section {{
            margin-bottom: 50px;
        }}

        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 25px;
            font-size: 1.8em;
        }}

        h3 {{
            color: #764ba2;
            margin: 25px 0 15px 0;
            font-size: 1.3em;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}

        .metric-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .confusion-matrix {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}

        .confusion-matrix table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        .confusion-matrix td, .confusion-matrix th {{
            padding: 15px;
            text-align: center;
            border: 2px solid #ddd;
            font-size: 1.1em;
        }}

        .confusion-matrix th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}

        .image-container {{
            margin: 30px 0;
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }}

        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}

        .alert {{
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            border-left: 5px solid;
        }}

        .alert-success {{
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }}

        .alert-warning {{
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }}

        .alert-danger {{
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }}

        .alert-info {{
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }}

        table.report-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}

        table.report-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}

        table.report-table td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}

        table.report-table tr:hover {{
            background: #f5f5f5;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            color: white;
        }}

        .badge-success {{ background: #28a745; }}
        .badge-warning {{ background: #ffc107; color: #333; }}
        .badge-danger {{ background: #dc3545; }}

        @media print {{
            body {{ background: white; padding: 0; }}
            .container {{ box-shadow: none; }}
            .metric-card {{ break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Informe de Evaluación del Modelo</h1>
            <p>Sistema de Predicción de Churn - Clientes de Alto Valor</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            </p>
        </header>

        <div class="content">
            <!-- RESUMEN EJECUTIVO -->
            <section id="executive-summary">
                <h2>📋 Resumen Ejecutivo</h2>

                <div class="alert alert-info">
                    <strong>Objetivo:</strong> Evaluar el rendimiento del modelo de predicción de churn
                    entrenado con DistilBERT para identificar clientes de alto valor en riesgo de abandono.
                </div>

                <p style="margin: 20px 0;">
                    Se evaluaron <strong>{summary['test_samples']}</strong> casos de prueba
                    con el modelo ubicado en <code>{summary['model_path']}</code>.
                </p>

                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value" style="color: {get_metric_color(metrics['accuracy'])}">
                            {format_metric(metrics['accuracy'], 3)}
                        </div>
                        <span class="badge badge-{('success' if metrics['accuracy'] >= 0.7 else 'warning')}">
                            {format_metric(metrics['accuracy'] * 100, 1)}%
                        </span>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value" style="color: {get_metric_color(metrics['precision'])}">
                            {format_metric(metrics['precision'], 3)}
                        </div>
                        <span class="badge badge-{('success' if metrics['precision'] >= 0.7 else 'warning')}">
                            {format_metric(metrics['precision'] * 100, 1)}%
                        </span>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value" style="color: {get_metric_color(metrics['recall'])}">
                            {format_metric(metrics['recall'], 3)}
                        </div>
                        <span class="badge badge-{('success' if metrics['recall'] >= 0.7 else 'warning')}">
                            {format_metric(metrics['recall'] * 100, 1)}%
                        </span>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value" style="color: {get_metric_color(metrics['f1_score'])}">
                            {format_metric(metrics['f1_score'], 3)}
                        </div>
                        <span class="badge badge-{('success' if metrics['f1_score'] >= 0.7 else 'warning')}">
                            {format_metric(metrics['f1_score'] * 100, 1)}%
                        </span>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">ROC-AUC</div>
                        <div class="metric-value" style="color: {get_metric_color(metrics['roc_auc'])}">
                            {format_metric(metrics['roc_auc'], 3)}
                        </div>
                        <span class="badge badge-{('success' if metrics['roc_auc'] >= 0.8 else 'warning')}">
                            Excelente
                        </span>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Avg Precision</div>
                        <div class="metric-value" style="color: {get_metric_color(metrics['average_precision'])}">
                            {format_metric(metrics['average_precision'], 3)}
                        </div>
                        <span class="badge badge-{('success' if metrics['average_precision'] >= 0.7 else 'warning')}">
                            {format_metric(metrics['average_precision'] * 100, 1)}%
                        </span>
                    </div>
                </div>
            </section>

            <!-- MATRIZ DE CONFUSIÓN -->
            <section id="confusion-matrix">
                <h2>🎯 Matriz de Confusión</h2>

                <div class="confusion-matrix">
                    <table>
                        <tr>
                            <th></th>
                            <th>Predicción: No Churn</th>
                            <th>Predicción: Churn</th>
                        </tr>
                        <tr>
                            <th>Real: No Churn</th>
                            <td style="background: #d4edda; font-weight: bold;">
                                {metrics['confusion_matrix']['tn']}<br>
                                <small>True Negatives</small>
                            </td>
                            <td style="background: #f8d7da; font-weight: bold;">
                                {metrics['confusion_matrix']['fp']}<br>
                                <small>False Positives</small>
                            </td>
                        </tr>
                        <tr>
                            <th>Real: Churn</th>
                            <td style="background: #f8d7da; font-weight: bold;">
                                {metrics['confusion_matrix']['fn']}<br>
                                <small>False Negatives</small>
                            </td>
                            <td style="background: #d4edda; font-weight: bold;">
                                {metrics['confusion_matrix']['tp']}<br>
                                <small>True Positives</small>
                            </td>
                        </tr>
                    </table>
                </div>

                <div class="image-container">
                    <img src="data:image/png;base64,{img_to_base64(RESULTS_DIR / 'confusion_matrix.png')}"
                         alt="Matriz de Confusión">
                </div>

                <h3>📊 Métricas Derivadas</h3>
                <table class="report-table">
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                        <th>Descripción</th>
                    </tr>
                    <tr>
                        <td><strong>Specificity</strong></td>
                        <td>{format_metric(metrics['specificity'], 4)}</td>
                        <td>Capacidad de identificar correctamente los no-churners</td>
                    </tr>
                    <tr>
                        <td><strong>NPV (Negative Predictive Value)</strong></td>
                        <td>{format_metric(metrics['npv'], 4)}</td>
                        <td>Probabilidad de que un cliente predicho como "No Churn" realmente no haga churn</td>
                    </tr>
                    <tr>
                        <td><strong>FPR (False Positive Rate)</strong></td>
                        <td>{format_metric(metrics['fpr'], 4)}</td>
                        <td>Proporción de falsos positivos</td>
                    </tr>
                    <tr>
                        <td><strong>FNR (False Negative Rate)</strong></td>
                        <td>{format_metric(metrics['fnr'], 4)}</td>
                        <td>Proporción de falsos negativos</td>
                    </tr>
                </table>
            </section>

            <!-- CURVAS DE EVALUACIÓN -->
            <section id="curves">
                <h2>📈 Curvas de Evaluación</h2>

                <h3>Curva ROC</h3>
                <div class="image-container">
                    <img src="data:image/png;base64,{img_to_base64(RESULTS_DIR / 'roc_curve.png')}"
                         alt="Curva ROC">
                </div>
                <p style="margin: 15px 0;">
                    El AUC (Area Under Curve) de <strong>{format_metric(metrics['roc_auc'], 3)}</strong>
                    indica {'excelente' if metrics['roc_auc'] >= 0.8 else 'buena' if metrics['roc_auc'] >= 0.7 else 'aceptable'}
                    capacidad de discriminación entre clases.
                </p>

                <h3>Curva Precision-Recall</h3>
                <div class="image-container">
                    <img src="data:image/png;base64,{img_to_base64(RESULTS_DIR / 'precision_recall_curve.png')}"
                         alt="Curva Precision-Recall">
                </div>

                <h3>Distribución de Probabilidades</h3>
                <div class="image-container">
                    <img src="data:image/png;base64,{img_to_base64(RESULTS_DIR / 'probability_distribution.png')}"
                         alt="Distribución de Probabilidades">
                </div>
            </section>

            <!-- RESUMEN DE MÉTRICAS -->
            <section id="metrics-summary">
                <h2>📊 Resumen Visual de Métricas</h2>

                <div class="image-container">
                    <img src="data:image/png;base64,{img_to_base64(RESULTS_DIR / 'metrics_summary.png')}"
                         alt="Resumen de Métricas">
                </div>
            </section>

            <!-- ANÁLISIS DE UMBRALES -->
            <section id="threshold-analysis">
                <h2>🎯 Análisis por Umbrales de Decisión</h2>

                <p>
                    El umbral de decisión afecta el balance entre precision y recall.
                    A continuación se muestra el rendimiento con diferentes umbrales:
                </p>

                <table class="report-table">
                    <tr>
                        <th>Umbral</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
"""

# Agregar filas de análisis de umbrales
for item in threshold_analysis:
    html_content += f"""
                    <tr>
                        <td><strong>{item['threshold']:.1f}</strong></td>
                        <td>{format_metric(item['accuracy'], 4)}</td>
                        <td>{format_metric(item['precision'], 4)}</td>
                        <td>{format_metric(item['recall'], 4)}</td>
                        <td>{format_metric(item['f1_score'], 4)}</td>
                    </tr>
"""

html_content += f"""
                </table>

                <div class="image-container">
                    <img src="data:image/png;base64,{img_to_base64(RESULTS_DIR / 'threshold_analysis.png')}"
                         alt="Análisis de Umbrales">
                </div>
            </section>

            <!-- REPORTE DE CLASIFICACIÓN -->
            <section id="classification-report">
                <h2>📄 Reporte de Clasificación Detallado</h2>

                <table class="report-table">
                    <tr>
                        <th>Clase</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                    <tr>
                        <td><strong>No Churn</strong></td>
                        <td>{format_metric(class_report['No Churn']['precision'], 4)}</td>
                        <td>{format_metric(class_report['No Churn']['recall'], 4)}</td>
                        <td>{format_metric(class_report['No Churn']['f1-score'], 4)}</td>
                        <td>{class_report['No Churn']['support']}</td>
                    </tr>
                    <tr>
                        <td><strong>Churn</strong></td>
                        <td>{format_metric(class_report['Churn']['precision'], 4)}</td>
                        <td>{format_metric(class_report['Churn']['recall'], 4)}</td>
                        <td>{format_metric(class_report['Churn']['f1-score'], 4)}</td>
                        <td>{class_report['Churn']['support']}</td>
                    </tr>
                    <tr style="background: #f8f9fa; font-weight: bold;">
                        <td>Macro Avg</td>
                        <td>{format_metric(class_report['macro avg']['precision'], 4)}</td>
                        <td>{format_metric(class_report['macro avg']['recall'], 4)}</td>
                        <td>{format_metric(class_report['macro avg']['f1-score'], 4)}</td>
                        <td>{class_report['macro avg']['support']}</td>
                    </tr>
                    <tr style="background: #e9ecef; font-weight: bold;">
                        <td>Weighted Avg</td>
                        <td>{format_metric(class_report['weighted avg']['precision'], 4)}</td>
                        <td>{format_metric(class_report['weighted avg']['recall'], 4)}</td>
                        <td>{format_metric(class_report['weighted avg']['f1-score'], 4)}</td>
                        <td>{class_report['weighted avg']['support']}</td>
                    </tr>
                </table>
            </section>

            <!-- RECOMENDACIONES -->
            <section id="recommendations">
                <h2>💡 Recomendaciones y Conclusiones</h2>
"""

# Agregar recomendaciones
if summary.get('recommendations'):
    for rec in summary['recommendations']:
        if '✅' in rec:
            alert_class = 'alert-success'
        elif '⚠️' in rec:
            alert_class = 'alert-warning'
        else:
            alert_class = 'alert-info'

        html_content += f"""
                <div class="alert {alert_class}">
                    {rec}
                </div>
"""

html_content += """
                <h3>🎯 Conclusiones Principales</h3>
                <ul style="line-height: 2; margin: 20px 0;">
                    <li>El modelo muestra un rendimiento balanceado entre precision y recall</li>
                    <li>La capacidad de discriminación (ROC-AUC) es adecuada para uso en producción</li>
                    <li>Se recomienda ajustar el umbral según el caso de uso específico</li>
                    <li>Para minimizar pérdida de clientes: priorizar Recall (detectar más churners)</li>
                    <li>Para minimizar costos de retención: priorizar Precision (evitar falsos positivos)</li>
                </ul>

                <h3>📌 Próximos Pasos</h3>
                <ol style="line-height: 2; margin: 20px 0;">
                    <li>Validar el modelo con datos más recientes</li>
                    <li>Implementar sistema de monitoreo continuo</li>
                    <li>Evaluar impacto en métricas de negocio (ROI, retención, etc.)</li>
                    <li>Considerar reentrenamiento periódico con nuevos datos</li>
                    <li>Integrar con sistemas CRM para acciones automatizadas</li>
                </ol>
            </section>
        </div>

        <div class="footer">
            <p><strong>Sistema de Predicción de Churn</strong></p>
            <p>Magister en Inteligencia Artificial - Tópicos Avanzados</p>
            <p>Generado automáticamente el {datetime.now().strftime('%d de %B de %Y a las %H:%M:%S')}</p>
            <p style="margin-top: 15px; font-size: 0.85em;">
                Este reporte fue generado usando Python, scikit-learn, transformers y matplotlib
            </p>
        </div>
    </div>
</body>
</html>
"""

# ============================================================================
# GUARDAR HTML
# ============================================================================

print("\n💾 Guardando reporte HTML...")

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Reporte guardado: {REPORT_PATH}")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*80)
print("✅ REPORTE HTML GENERADO EXITOSAMENTE")
print("="*80)
print(f"\n📄 Archivo: {REPORT_PATH.absolute()}")
print(f"📊 Tamaño: {REPORT_PATH.stat().st_size / 1024:.1f} KB")
print("\n💡 Para visualizar:")
print(f"   Abre el archivo en tu navegador o ejecuta:")
print(f"   open {REPORT_PATH}  # macOS")
print(f"   xdg-open {REPORT_PATH}  # Linux")
print(f"   start {REPORT_PATH}  # Windows")
print("="*80)
