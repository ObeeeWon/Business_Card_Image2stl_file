# Card→STL Generator — Fung-AI Studio

Business card image → 3D-printable STL file.  
Encapsulates all 9 iterations of refinement developed in the project.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.9+)
pip install flask pillow numpy scipy

# 2. Run
python app.py

# 3. Open browser
http://localhost:5000
```

---

## What it does

Upload two images (front face + back face of your business card),
click Generate, download the STL.

**Output spec:**
- Size: 85.6 × 54 mm (ISO standard business card)
- Thickness: 2 mm total
- Engraving depth: 0.5 mm
- Corner radius: 3 mm rounded
- Edge chamfer: 1.5 mm slope (no sharp edges)
- Resolution: 6 px/mm
- Format: Binary STL (~32 MB)

**Print settings (Bambu Lab / any FDM):**
- Layer height: 0.1 mm
- Nozzle: 0.4 mm
- Material: PLA
- Orientation: front face up, no supports needed
- After printing: flip card to read back face

---

## Algorithm v9 — key decisions

| Problem | Root cause | Fix |
|---------|-----------|-----|
| Horizontal scratches | Span-merge T-junctions | Pure per-pixel quads, zero merging |
| Letter merging | Strokes too wide for 0.4mm nozzle | Distance-field gap enforcement |
| Edge staircase | Pixel-level rounded corner mask | 8× supersampled mask |
| Jagged curves | Binary threshold | Greyscale depth map + gaussian |
| Hollow geometry | Two chamfers crossing | Flat bottom face, chamfer on top only |
| AI App reversed | Wrong mirror logic | Pure fliplr (physical flip restores) |
| Non-manifold | Various mesh issues | Guaranteed z_top > z_bot + 0.3mm |

---

## File structure

```
card_app/
  app.py          ← Flask app + STL generator (single file)
  requirements.txt
  README.md
  uploads/        ← temp (auto-created, auto-cleaned)
  outputs/        ← generated STL files (auto-created)
```

---

## Fung-AI Studio
feng@fung-ai.com · 902-979-1521 · Nova Scotia, Canada
