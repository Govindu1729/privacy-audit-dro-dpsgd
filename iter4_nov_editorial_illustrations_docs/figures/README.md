# Editorial Illustrations

This folder contains 4 conceptual editorial illustrations designed in the style of an illustrated science story (think *The New Yorker* meets *Kurzgesagt* / *Distill*).

## 📁 Files

| Illustration | SVG | PNG | PDF | Description |
|--------------|-----|-----|-----|-------------|
| **Act I: The Over-Caring Spotlight** | `illustration_01_over_caring_spotlight.svg` | `illustration_01_over_caring_spotlight.png` | `illustration_01_over_caring_spotlight.pdf` | Why trying to make AI fair accidentally destroys privacy |
| **Act II: The Pedestal Effect** | `illustration_02_pedestal_effect.svg` | `illustration_02_pedestal_effect.png` | `illustration_02_pedestal_effect.pdf` | Why smaller groups leak more privacy (r = -0.91) |
| **Act III: Scissors & Blizzard** | `illustration_03_scissors_and_blizzard.svg` | `illustration_03_scissors_and_blizzard.png` | `illustration_03_scissors_and_blizzard.pdf` | Why naive DRO + DP-SGD stacking fails |
| **Act IV: The Shared Umbrella** | `illustration_04_shared_umbrella.svg` | `illustration_04_shared_umbrella.png` | `illustration_04_shared_umbrella.pdf` | The constructive fix: CTGR-DP-DRO |

## 🎨 Format Guide

- **SVG** (Scalable Vector Graphics): Best for web, infinite zoom, editable in vector editors like Inkscape or Figma
- **PNG** (300 DPI, 1920px wide): Best for PowerPoint, social media, print presentations
- **PDF** (Vector): Best for LaTeX/Overleaf papers

## 🔧 Generating PNG/PDF from SVG

If PNG/PDF files are missing or you want to regenerate them, run:

```bash
cd /Users/srimaan/Projects/privacy-audit/docs/figures

# Install cairosvg (requires system libcairo)
pip install cairosvg

# Run conversion script
python convert_illustrations.py
```

### Alternative Methods (if cairosvg fails)

**Option 1: Browser Method (No installation needed)**
1. Open each `.svg` file in Chrome/Safari/Firefox
2. Right-click → "Print" or Ctrl+P
3. Select "Save as PDF"
4. For PNG: Use a screenshot tool or online converter like [svgtopng.com](https://svgtopng.com)

**Option 2: Inkscape (Free, cross-platform)**
```bash
# Install Inkscape from https://inkscape.org
# Then convert:
inkscape illustration_01_over_caring_spotlight.svg --export-type=png --export-width=1920
inkscape illustration_01_over_caring_spotlight.svg --export-type=pdf
```

**Option 3: Online Converters**
- [CloudConvert SVG to PNG](https://cloudconvert.com/svg-to-png)
- [CloudConvert SVG to PDF](https://cloudconvert.com/svg-to-pdf)

## 📖 Quick Reference for Presentations

### 30-Second Explanation (No Equations!)

1. **The Spotlight Trap:** *"When we tell an AI to be fair to a tiny minority group, it shines a giant spotlight on those few people. Instead of learning general patterns, it memorizes their exact faces—making them easy targets for snooping hackers."*

2. **The Pedestal Effect:** *"Why do the smallest groups suffer the most? Because on a seesaw of justice, 3 people can only balance 10,000 people if they stand on giant stone pedestals—pushing their heads right out above the clouds."*

3. **The Scissors & The Blizzard:** *"When we add standard privacy tools on top, privacy acts like hedge shears that chop off those pedestals, followed by a snowstorm that blinds the AI so it forgets how to be fair."*

4. **The Shared Umbrella:** *"The fix is to stop turning single unusual people into lightning rods. Instead, make sure every person is kept at a safe, equal height first, and lift the whole community together on one shared platform under a privacy umbrella."*

## 🎯 Usage in Paper/Presentations

These illustrations are designed for:
- Conference posters (use PDF format for best quality)
- Slide presentations (use PNG at 300 DPI)
- Blog posts / social media (use PNG)
- LaTeX papers (use PDF with `\includegraphics`)

### Example LaTeX Usage:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/illustration_01_over_caring_spotlight.pdf}
    \caption{The Paradox of the Fairness Spotlight: When DRO forces the model to give equal importance to tiny minority groups, it memorizes individual quirks rather than learning generalizable patterns, creating privacy vulnerabilities.}
    \label{fig:spotlight}
\end{figure}
```
