#!/usr/bin/env python3
"""
Convert SVG illustrations to PNG and PDF formats.

This script converts the 4 editorial illustrations from SVG to:
- PNG (300 DPI, 1920px wide) - for PowerPoint, social media, web
- PDF (vector) - for LaTeX/Overleaf

Requirements:
    pip install cairosvg

Usage:
    python convert_illustrations.py
"""
import subprocess
import sys

# Install required package
print("Installing cairosvg...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg", "-q"])

import cairosvg
from pathlib import Path

# Define the base path (same directory as this script)
base_dir = Path(__file__).parent

# List of SVG files to convert
svg_files = [
    "illustration_01_over_caring_spotlight.svg",
    "illustration_02_pedestal_effect.svg",
    "illustration_03_scissors_and_blizzard.svg",
    "illustration_04_shared_umbrella.svg",
]

print("\n" + "="*60)
print("CONVERTING EDITORIAL ILLUSTRATIONS")
print("="*60 + "\n")

converted = 0
for svg_file in svg_files:
    svg_path = base_dir / svg_file
    if not svg_path.exists():
        print(f"❌ {svg_file} not found!")
        continue
    
    print(f"📄 Processing: {svg_file}")
    
    # Convert to PNG (high resolution for presentations)
    png_path = base_dir / svg_file.replace(".svg", ".png")
    try:
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=1920,
            dpi=300
        )
        print(f"   ✅ PNG: {png_path.name}")
    except Exception as e:
        print(f"   ❌ PNG failed: {e}")
    
    # Convert to PDF (for LaTeX/Overleaf)
    pdf_path = base_dir / svg_file.replace(".svg", ".pdf")
    try:
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        print(f"   ✅ PDF: {pdf_path.name}")
    except Exception as e:
        print(f"   ❌ PDF failed: {e}")
    
    converted += 1
    print()

print("="*60)
print(f"🎉 CONVERSION COMPLETE! ({converted}/{len(svg_files)} files)")
print("="*60)
print(f"\nFiles saved to: {base_dir}\n")
print("Format guide:")
print("  🖼️  SVG: Vector format, infinite zoom, best for web")
print("  📸 PNG: Raster format, 300 DPI, best for PowerPoint/social media")
print("  📄 PDF: Vector format, best for LaTeX/Overleaf")
print()
