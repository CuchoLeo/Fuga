"""
GENERADOR DE HTML PARA ARTÍCULO DE REVISTA
Convierte ARTICULO_REVISTA.md a HTML con diseño profesional de revista tech

Uso: python generate_article_html.py
"""

import markdown
import base64
from pathlib import Path
from datetime import datetime

print("="*80)
print("📰 GENERADOR DE HTML - ARTÍCULO DE REVISTA TECH")
print("="*80)

# Archivos
MARKDOWN_FILE = Path("ARTICULO_REVISTA.md")
HTML_FILE = Path("ARTICULO_REVISTA.html")

# Verificar archivo
if not MARKDOWN_FILE.exists():
    print(f"\n❌ Error: No se encontró {MARKDOWN_FILE}")
    exit(1)

print(f"\n✓ Archivo Markdown encontrado: {MARKDOWN_FILE}")

# Leer markdown
with open(MARKDOWN_FILE, 'r', encoding='utf-8') as f:
    md_content = f.read()

print("✓ Contenido leído")

# Convertir a HTML
print("\n🔄 Convirtiendo a HTML...")

md = markdown.Markdown(extensions=[
    'extra',
    'codehilite',
    'toc',
    'tables',
    'fenced_code'
])

body_html = md.convert(md_content)

# Intentar cargar visualizaciones
def load_image_base64(image_path):
    """Carga imagen y la convierte a base64"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# Cargar imágenes de test_results si existen
roc_img = load_image_base64(Path("test_results/roc_curve.png"))
confusion_img = load_image_base64(Path("test_results/confusion_matrix.png"))
metrics_img = load_image_base64(Path("test_results/metrics_summary.png"))

# Template HTML estilo revista tech
html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Churnito: Sistema inteligente de predicción de churn con Deep Learning y LLMs">
    <meta name="keywords" content="AI, Machine Learning, Churn Prediction, DistilBERT, LLM, Banking">
    <meta name="author" content="Víctor Rodríguez">
    <title>Churnito: IA para Predicción de Churn Bancario | Revista Tech</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Fira+Code:wght@400;500&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --primary: #2563eb;
            --secondary: #7c3aed;
            --accent: #f59e0b;
            --dark: #0f172a;
            --light: #f8fafc;
            --gray: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.7;
            color: var(--dark);
            background: var(--light);
            font-size: 17px;
        }}

        /* Header Hero */
        .hero {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 80px 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .hero::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)" /></svg>');
            opacity: 0.3;
        }}

        .hero-content {{
            max-width: 900px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }}

        .hero h1 {{
            font-size: 3.5em;
            font-weight: 900;
            margin-bottom: 20px;
            line-height: 1.1;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        .hero .subtitle {{
            font-size: 1.4em;
            font-weight: 300;
            margin-bottom: 30px;
            opacity: 0.95;
        }}

        .hero .meta {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 30px;
            font-size: 0.95em;
        }}

        .hero .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Container */
        .container {{
            max-width: 800px;
            margin: -50px auto 0;
            background: white;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            border-radius: 20px 20px 0 0;
            padding: 60px 60px 40px;
            position: relative;
            z-index: 10;
        }}

        /* Typography */
        h1, h2, h3, h4 {{
            font-weight: 700;
            line-height: 1.3;
            color: var(--dark);
        }}

        h2 {{
            font-size: 2.2em;
            margin: 60px 0 25px;
            padding-bottom: 15px;
            border-bottom: 4px solid var(--primary);
            position: relative;
        }}

        h2::before {{
            content: '';
            position: absolute;
            left: 0;
            bottom: -4px;
            width: 60px;
            height: 4px;
            background: var(--accent);
        }}

        h3 {{
            font-size: 1.6em;
            margin: 40px 0 20px;
            color: var(--primary);
        }}

        h4 {{
            font-size: 1.2em;
            margin: 30px 0 15px;
            color: var(--secondary);
        }}

        p {{
            margin: 20px 0;
            text-align: justify;
        }}

        /* Lead paragraph */
        .container > p:first-of-type {{
            font-size: 1.2em;
            font-weight: 400;
            color: var(--gray);
            border-left: 4px solid var(--accent);
            padding-left: 20px;
            margin: 30px 0;
        }}

        /* Code */
        code {{
            background: #f1f5f9;
            padding: 3px 8px;
            border-radius: 5px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
            color: var(--danger);
            font-weight: 500;
        }}

        pre {{
            background: var(--dark);
            color: #e2e8f0;
            padding: 25px;
            border-radius: 12px;
            overflow-x: auto;
            margin: 30px 0;
            line-height: 1.6;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 30px 0;
            font-size: 0.95em;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-radius: 10px;
            overflow: hidden;
        }}

        table thead {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
        }}

        table th {{
            padding: 15px 20px;
            text-align: left;
            font-weight: 600;
        }}

        table td {{
            padding: 15px 20px;
            border-bottom: 1px solid #e5e7eb;
        }}

        table tbody tr {{
            background: white;
            transition: background 0.2s;
        }}

        table tbody tr:nth-of-type(even) {{
            background: #f9fafb;
        }}

        table tbody tr:hover {{
            background: #eff6ff;
        }}

        /* Blockquotes */
        blockquote {{
            border-left: 5px solid var(--accent);
            margin: 30px 0;
            padding: 20px 25px;
            background: #fffbeb;
            border-radius: 0 8px 8px 0;
            font-style: italic;
        }}

        /* Lists */
        ul, ol {{
            margin: 20px 0;
            padding-left: 30px;
        }}

        li {{
            margin: 12px 0;
        }}

        /* Links */
        a {{
            color: var(--primary);
            text-decoration: none;
            border-bottom: 2px solid transparent;
            transition: border-color 0.2s;
        }}

        a:hover {{
            border-bottom-color: var(--primary);
        }}

        /* Badges */
        strong {{
            font-weight: 600;
            color: var(--dark);
        }}

        /* Highlight boxes */
        .highlight {{
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-left: 5px solid var(--primary);
            padding: 25px;
            margin: 30px 0;
            border-radius: 0 10px 10px 0;
        }}

        .warning {{
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-left: 5px solid var(--warning);
        }}

        .success {{
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-left: 5px solid var(--success);
        }}

        /* Images */
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin: 30px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}

        /* Stats grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

        .stat-card {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
        }}

        .stat-value {{
            font-size: 2.5em;
            font-weight: 900;
            display: block;
            margin: 10px 0;
        }}

        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Footer */
        .article-footer {{
            margin-top: 80px;
            padding-top: 40px;
            border-top: 3px solid var(--light);
            text-align: center;
            color: var(--gray);
        }}

        .author-box {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 30px;
            border-radius: 12px;
            margin: 40px 0;
            border: 2px solid #e2e8f0;
        }}

        .author-box h3 {{
            color: var(--primary);
            margin-top: 0;
        }}

        /* Print button */
        .print-button {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: var(--primary);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
            z-index: 1000;
        }}

        .print-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(37, 99, 235, 0.5);
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .hero h1 {{
                font-size: 2.2em;
            }}

            .container {{
                padding: 40px 25px;
                margin-top: -30px;
                border-radius: 15px 15px 0 0;
            }}

            h2 {{
                font-size: 1.8em;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Print styles */
        @media print {{
            .print-button {{
                display: none;
            }}

            body {{
                background: white;
            }}

            .container {{
                box-shadow: none;
                margin-top: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="hero">
        <div class="hero-content">
            <h1>🤖 Churnito</h1>
            <p class="subtitle">Un Sistema Inteligente de Predicción de Abandono de Clientes<br>que Combina Deep Learning y Conversación Natural</p>
            <div class="meta">
                <div class="meta-item">
                    <span>👤</span>
                    <span>Por <strong>Víctor Rodríguez</strong></span>
                </div>
                <div class="meta-item">
                    <span>📅</span>
                    <span>{datetime.now().strftime('%B %Y')}</span>
                </div>
                <div class="meta-item">
                    <span>⏱️</span>
                    <span>15 min lectura</span>
                </div>
                <div class="meta-item">
                    <span>🎯</span>
                    <span>Inteligencia Artificial</span>
                </div>
            </div>
        </div>
    </div>

    <button class="print-button" onclick="window.print()">
        🖨️ Exportar PDF
    </button>

    <div class="container">
        {body_html}

        <div class="article-footer">
            <div class="author-box">
                <h3>Sobre el Autor</h3>
                <p><strong>Víctor Rodríguez</strong> es estudiante de Magister en Inteligencia Artificial, especializado en NLP y aplicaciones empresariales de Machine Learning. Su investigación se enfoca en democratizar herramientas de IA avanzadas para empresas de cualquier tamaño.</p>
                <p style="margin-top: 15px;">
                    <strong>GitHub:</strong> <a href="https://github.com/CuchoLeo" target="_blank">@CuchoLeo</a><br>
                    <strong>Proyecto:</strong> <a href="https://github.com/CuchoLeo/Fuga" target="_blank">github.com/CuchoLeo/Fuga</a>
                </p>
            </div>

            <hr style="border: none; border-top: 2px solid #e5e7eb; margin: 40px 0;">

            <p style="color: var(--gray); font-size: 0.9em;">
                <strong>Revista de Tecnología e Innovación</strong><br>
                Artículo publicado: {datetime.now().strftime('%d de %B de %Y')}<br>
                Palabras clave: Machine Learning, Transformers, LLM, Predicción de Churn, DistilBERT
            </p>

            <p style="margin-top: 20px; font-size: 0.85em; color: var(--gray);">
                🤖 <em>Este artículo fue desarrollado con asistencia de Claude Code de Anthropic</em>
            </p>
        </div>
    </div>

    <script>
        // Smooth scroll para enlaces internos
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});

        // Highlight de código
        console.log('%c🤖 Churnito Article Loaded', 'color: #2563eb; font-size: 16px; font-weight: bold;');
        console.log('%cRepositorio: https://github.com/CuchoLeo/Fuga', 'color: #64748b;');
    </script>
</body>
</html>
"""

# Guardar HTML
with open(HTML_FILE, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"✓ HTML generado: {HTML_FILE}")

# Calcular estadísticas
size_kb = HTML_FILE.stat().st_size / 1024
with open(MARKDOWN_FILE, 'r') as f:
    word_count = len(f.read().split())

print(f"\n📊 Estadísticas:")
print(f"   Tamaño HTML: {size_kb:.1f} KB")
print(f"   Palabras: ~{word_count:,}")

print("\n" + "="*80)
print("✅ ARTÍCULO DE REVISTA GENERADO")
print("="*80)
print(f"\n📁 Archivos creados:")
print(f"   - {MARKDOWN_FILE} (fuente)")
print(f"   - {HTML_FILE} (visualización)")
print("\n💡 Próximos pasos:")
print("   1. Abrir HTML en navegador:")
print(f"      open {HTML_FILE}")
print("   2. Revisar formato y contenido")
print("   3. Exportar a PDF con Cmd+P / Ctrl+P")
print("   4. Enviar a revista para revisión")
print("\n" + "="*80)

# Intentar abrir automáticamente
import subprocess
import platform

try:
    if platform.system() == 'Darwin':  # macOS
        subprocess.run(['open', str(HTML_FILE)])
        print(f"\n🌐 Abriendo {HTML_FILE} en el navegador...")
    elif platform.system() == 'Linux':
        subprocess.run(['xdg-open', str(HTML_FILE)])
    elif platform.system() == 'Windows':
        subprocess.run(['start', str(HTML_FILE)], shell=True)
except:
    pass

print("="*80)
