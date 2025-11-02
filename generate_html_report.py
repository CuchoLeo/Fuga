"""
GENERADOR DE HTML DEL INFORME FINAL
Convierte INFORME_FINAL.md a HTML profesional listo para PDF

Uso: python generate_html_report.py
Luego: Abrir en navegador y Ctrl+P / Cmd+P → Guardar como PDF
"""

import markdown
from pathlib import Path
from datetime import datetime

print("="*80)
print("📄 GENERADOR DE HTML - INFORME FINAL")
print("="*80)

# Archivos
MARKDOWN_FILE = Path("INFORME_FINAL.md")
HTML_FILE = Path("INFORME_FINAL.html")

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

# Usar markdown con extensiones
md = markdown.Markdown(extensions=[
    'extra',
    'codehilite',
    'toc',
    'tables',
    'fenced_code'
])

body_html = md.convert(md_content)

# Template HTML profesional
html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe Final - Sistema de Predicción de Churn</title>
    <style>
        @media print {{
            body {{
                margin: 0;
                padding: 20px;
            }}
            .no-print {{
                display: none;
            }}
            .page-break {{
                page-break-after: always;
            }}
            h1, h2, h3 {{
                page-break-after: avoid;
            }}
            table {{
                page-break-inside: avoid;
            }}
        }}

        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #fff;
        }}

        h1 {{
            color: #1a1a1a;
            border-bottom: 4px solid #2563eb;
            padding-bottom: 15px;
            margin-top: 60px;
            font-size: 2.5em;
            page-break-before: always;
        }}

        h1:first-of-type {{
            margin-top: 0;
            page-break-before: avoid;
            text-align: center;
            border-bottom: none;
        }}

        h2 {{
            color: #2563eb;
            margin-top: 50px;
            margin-bottom: 25px;
            font-size: 1.8em;
            border-left: 5px solid #2563eb;
            padding-left: 15px;
        }}

        h3 {{
            color: #3b82f6;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        h4 {{
            color: #60a5fa;
            margin-top: 20px;
            font-size: 1.1em;
        }}

        p {{
            margin: 15px 0;
            text-align: justify;
        }}

        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #e11d48;
        }}

        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            page-break-inside: avoid;
        }}

        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }}

        table thead tr {{
            background: #2563eb;
            color: white;
            text-align: left;
        }}

        table th,
        table td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
        }}

        table tbody tr {{
            border-bottom: 1px solid #ddd;
        }}

        table tbody tr:nth-of-type(even) {{
            background-color: #f8fafc;
        }}

        table tbody tr:hover {{
            background-color: #e0f2fe;
        }}

        blockquote {{
            border-left: 4px solid #3b82f6;
            margin: 25px 0;
            padding: 15px 20px;
            background: #eff6ff;
            font-style: italic;
        }}

        ul, ol {{
            margin: 15px 0;
            padding-left: 30px;
        }}

        li {{
            margin: 8px 0;
        }}

        strong {{
            color: #1e40af;
            font-weight: 600;
        }}

        em {{
            color: #64748b;
        }}

        a {{
            color: #2563eb;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        hr {{
            border: none;
            border-top: 2px solid #e5e7eb;
            margin: 40px 0;
        }}

        .header {{
            text-align: center;
            margin-bottom: 60px;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 30px;
        }}

        .header h1 {{
            margin-top: 0;
            border: none;
            font-size: 2.8em;
            color: #1e40af;
        }}

        .header p {{
            color: #64748b;
            font-size: 1.1em;
            margin: 10px 0;
        }}

        .footer {{
            margin-top: 80px;
            padding-top: 30px;
            border-top: 2px solid #e5e7eb;
            text-align: center;
            color: #64748b;
            font-size: 0.9em;
        }}

        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 4px 12px rgba(37,99,235,0.3);
            z-index: 1000;
        }}

        .print-button:hover {{
            background: #1e40af;
        }}

        @media screen and (max-width: 768px) {{
            body {{
                padding: 20px 10px;
            }}

            h1 {{
                font-size: 2em;
            }}

            h2 {{
                font-size: 1.5em;
            }}

            table {{
                font-size: 0.85em;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-button no-print" onclick="window.print()">
        🖨️ Exportar a PDF
    </button>

    <div class="header">
        <h1>📊 INFORME FINAL</h1>
        <p><strong>Sistema de Predicción de Churn con Inteligencia Artificial</strong></p>
        <p>Magister en Inteligencia Artificial</p>
        <p>Tópicos Avanzados en Inteligencia Artificial 2</p>
        <p style="margin-top: 20px; color: #94a3b8;">
            Autor: Víctor Rodríguez<br>
            Fecha: {datetime.now().strftime('%d de %B de %Y')}
        </p>
    </div>

    {body_html}

    <div class="footer">
        <hr>
        <p><strong>Sistema de Predicción de Churn - Proyecto Final</strong></p>
        <p>Generado automáticamente el {datetime.now().strftime('%d de %B de %Y a las %H:%M:%S')}</p>
        <p>🤖 <em>Desarrollado con Claude Code</em></p>
    </div>

    <script>
        // Auto-generar tabla de contenidos
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Informe cargado correctamente');
        }});
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
print(f"\n📊 Tamaño del HTML: {size_kb:.1f} KB")

print("\n" + "="*80)
print("✅ INFORME HTML GENERADO")
print("="*80)
print(f"\n📁 Ubicación: {HTML_FILE.absolute()}")
print("\n💡 Pasos para generar PDF:")
print("   1. Abrir el archivo HTML en tu navegador")
print(f"      open {HTML_FILE}  # macOS")
print(f"      xdg-open {HTML_FILE}  # Linux")
print(f"      start {HTML_FILE}  # Windows")
print("\n   2. Presionar Cmd+P (Mac) o Ctrl+P (Windows/Linux)")
print("   3. Seleccionar 'Guardar como PDF'")
print("   4. Configurar:")
print("      - Márgenes: Predeterminados")
print("      - Escala: 100%")
print("      - Orientación: Vertical")
print("   5. Guardar como INFORME_FINAL.pdf")
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
        print(f"\n🌐 Abriendo {HTML_FILE} en el navegador...")
    elif platform.system() == 'Windows':
        subprocess.run(['start', str(HTML_FILE)], shell=True)
        print(f"\n🌐 Abriendo {HTML_FILE} en el navegador...")
except:
    print(f"\n💡 Abre manualmente: {HTML_FILE}")

print("="*80)
