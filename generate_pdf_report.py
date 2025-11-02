"""
GENERADOR DE PDF DEL INFORME FINAL
Convierte INFORME_FINAL.md a PDF profesional

Uso: python generate_pdf_report.py
Requiere: pandoc instalado (brew install pandoc)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("="*80)
print("📄 GENERADOR DE PDF - INFORME FINAL")
print("="*80)

# Archivos
MARKDOWN_FILE = Path("INFORME_FINAL.md")
PDF_FILE = Path("INFORME_FINAL.pdf")

# Verificar archivo markdown
if not MARKDOWN_FILE.exists():
    print(f"\n❌ Error: No se encontró {MARKDOWN_FILE}")
    sys.exit(1)

print(f"\n✓ Archivo Markdown encontrado: {MARKDOWN_FILE}")

# Verificar pandoc
try:
    result = subprocess.run(['pandoc', '--version'],
                          capture_output=True,
                          text=True,
                          check=True)
    pandoc_version = result.stdout.split('\n')[0]
    print(f"✓ Pandoc instalado: {pandoc_version}")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("\n⚠️  Pandoc no está instalado")
    print("\nPara instalar:")
    print("  macOS:   brew install pandoc")
    print("  Ubuntu:  sudo apt-get install pandoc")
    print("  Windows: choco install pandoc")
    print("\nAlternativamente, abre INFORME_FINAL.md en un editor")
    print("y exporta a PDF (VS Code, Typora, etc.)")
    sys.exit(1)

# Generar PDF
print(f"\n📝 Generando PDF: {PDF_FILE}")

try:
    # Comando pandoc con opciones
    cmd = [
        'pandoc',
        str(MARKDOWN_FILE),
        '-o', str(PDF_FILE),
        '--pdf-engine=xelatex',  # Mejor soporte Unicode
        '--toc',  # Tabla de contenidos
        '--toc-depth=3',
        '-V', 'geometry:margin=1in',
        '-V', 'fontsize=11pt',
        '-V', 'documentclass=report',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '--highlight-style=tango',
        '--number-sections'
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    print(f"✓ PDF generado exitosamente")

    # Verificar tamaño
    size_kb = PDF_FILE.stat().st_size / 1024
    print(f"\n📊 Tamaño del PDF: {size_kb:.1f} KB")

    print("\n" + "="*80)
    print("✅ INFORME FINAL GENERADO")
    print("="*80)
    print(f"\n📁 Ubicación: {PDF_FILE.absolute()}")
    print("\n💡 Para visualizar:")
    print(f"   open {PDF_FILE}  # macOS")
    print(f"   xdg-open {PDF_FILE}  # Linux")
    print(f"   start {PDF_FILE}  # Windows")
    print("="*80)

except subprocess.CalledProcessError as e:
    print(f"\n❌ Error al generar PDF:")
    print(e.stderr)
    print("\n💡 Alternativa: Usa un convertidor online o editor Markdown")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Error inesperado: {e}")
    sys.exit(1)
