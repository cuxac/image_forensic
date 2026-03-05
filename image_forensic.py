#!/usr/bin/env python3
"""
PDF Image Forensics Tool
========================
Analyse les images d'un article scientifique (PDF) pour détecter
d'éventuelles manipulations.

Techniques utilisées :
  - ELA (Error Level Analysis)
  - Analyse des métadonnées EXIF
  - Détection de clonage (copy-move) via feature matching
  - Analyse statistique (histogramme, bruit)
  - Détection d'incohérences de compression JPEG

Installation des dépendances :
    pip install pypdf pymupdf Pillow opencv-python numpy scipy pillow-heif

Usage :
    python pdf_image_forensics.py article.pdf
    python pdf_image_forensics.py article.pdf --output rapport.html --ela-quality 90
"""

import argparse
import base64
import io
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib

warnings.filterwarnings("ignore")

# ── Dépendances ────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("❌  PyMuPDF manquant — installez-le : pip install pymupdf")

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter
    from scipy import ndimage, stats
except ImportError as e:
    sys.exit(f"❌  Dépendance manquante : {e}\n   pip install opencv-python numpy Pillow scipy")


# ══════════════════════════════════════════════════════════════════════════════
# Structures de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Finding:
    """Un indice de manipulation détecté sur une image."""
    technique: str
    severity: str          # "info" | "warning" | "alert"
    description: str
    score: float = 0.0     # 0–1, 1 = très suspect

@dataclass
class ImageReport:
    page: int
    index: int
    filename: str
    width: int
    height: int
    mode: str
    findings: list[Finding] = field(default_factory=list)
    ela_b64: str = ""      # image ELA encodée en base64 pour le rapport HTML
    thumb_b64: str = ""    # miniature originale

    @property
    def max_severity(self) -> str:
        order = {"alert": 3, "warning": 2, "info": 1}
        if not self.findings:
            return "ok"
        return max(self.findings, key=lambda f: order.get(f.severity, 0)).severity

    @property
    def suspicion_score(self) -> float:
        if not self.findings:
            return 0.0
        return min(1.0, sum(f.score for f in self.findings) / max(1, len(self.findings)) * 2)


# ══════════════════════════════════════════════════════════════════════════════
# Extraction des images du PDF
# ══════════════════════════════════════════════════════════════════════════════

def extract_images_from_pdf(pdf_path: str, min_size: int = 50) -> list[dict]:
    """Extrait toutes les images d'un PDF via PyMuPDF."""
    doc = fitz.open(pdf_path)
    images = []
    seen_hashes = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]

                pil_img = Image.open(io.BytesIO(img_bytes))
                w, h = pil_img.size

                if w < min_size or h < min_size:
                    continue  # ignore les icônes/logos trop petits
                img_hash = hashlib.md5(img_bytes).hexdigest()
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)
                images.append({
                    "page": page_num + 1,
                    "index": img_idx,
                    "ext": ext,
                    "bytes": img_bytes,
                    "pil": pil_img,
                    "xref": xref,
                })
            except Exception as e:
                print(f"  ⚠  Page {page_num+1}, image {img_idx} : {e}", file=sys.stderr)

    doc.close()
    return images


# ══════════════════════════════════════════════════════════════════════════════
# Techniques d'analyse
# ══════════════════════════════════════════════════════════════════════════════

def run_ela(pil_img: Image.Image, quality: int = 95) -> tuple[Image.Image, float]:
    """
    Error Level Analysis :
    Re-compresse l'image en JPEG à une qualité donnée et calcule la différence.
    Des zones très brillantes = re-compressées différemment → indice de copier-coller.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    ela_img = ImageChops.difference(pil_img, recompressed)

    # Amplifier les différences pour la visualisation
    extrema = ela_img.getextrema()
    max_diff = max(max(e) for e in extrema) or 1
    scale = 255.0 / max_diff
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)

    # Score : écart-type normalisé des différences
    arr = np.array(ela_img).astype(float)
    score = float(np.std(arr) / 128.0)
    score = min(1.0, score)

    return ela_img, score


def analyze_noise_consistency(pil_img: Image.Image) -> tuple[float, str]:
    """
    Vérifie si le niveau de bruit est homogène dans l'image.
    Un bruit incohérent entre zones peut indiquer un montage.
    """
    gray = np.array(pil_img.convert("L")).astype(float)
    # Laplacien = carte du bruit local
    laplacian = ndimage.laplace(gray)

    # Divise l'image en grille 4×4 et compare les variances locales
    h, w = laplacian.shape
    bh, bw = h // 4, w // 4
    variances = []
    for i in range(4):
        for j in range(4):
            block = laplacian[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            variances.append(float(np.var(block)))

    if not variances:
        return 0.0, "Impossible de calculer"

    cv = np.std(variances) / (np.mean(variances) + 1e-9)  # Coefficient de variation
    score = min(1.0, cv / 5.0)

    if cv < 1.5:
        desc = f"Bruit homogène (CV={cv:.2f})"
    elif cv < 3.0:
        desc = f"Légères incohérences de bruit (CV={cv:.2f})"
    else:
        desc = f"Incohérences importantes de bruit (CV={cv:.2f}) — zones probablement composites"

    return score, desc


def detect_copy_move(pil_img: Image.Image, min_matches: int = 20) -> tuple[int, str]:
    """
    Détection de clonage (copy-move) via SIFT/ORB :
    Cherche des régions identiques à l'intérieur de la même image.
    """
    gray = np.array(pil_img.convert("L"))

    # ORB est disponible sans brevet
    detector = cv2.ORB_create(nfeatures=1000)
    kp, des = detector.detectAndCompute(gray, None)

    if des is None or len(kp) < 10:
        return 0, "Pas assez de points caractéristiques pour l'analyse"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des, des, k=2)

    suspicious = 0
    for m_list in matches:
        if len(m_list) < 2:
            continue
        m, n = m_list[0], m_list[1]
        # Ignorer l'auto-correspondance (même keypoint)
        if m.queryIdx == m.trainIdx:
            continue
        # Lowe's ratio test
        if m.distance < 0.75 * n.distance:
            # Vérifier que les deux points sont suffisamment éloignés spatialement
            pt1 = np.array(kp[m.queryIdx].pt)
            pt2 = np.array(kp[m.trainIdx].pt)
            dist = np.linalg.norm(pt1 - pt2)
            if dist > 20:
                suspicious += 1

    if suspicious < min_matches:
        desc = f"Aucun clonage détecté ({suspicious} correspondances spatiales)"
    elif suspicious < 50:
        desc = f"Correspondances internes modérées ({suspicious}) — à vérifier"
    else:
        desc = f"Nombreuses correspondances internes ({suspicious}) — clonage probable"

    return suspicious, desc


def analyze_histogram(pil_img: Image.Image) -> tuple[float, str]:
    """
    Analyse l'histogramme des niveaux de gris.
    Un histogramme avec des pics ou des lacunes régulières indique
    une retouche ou une modification des courbes tonales.
    """
    gray = np.array(pil_img.convert("L"))
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float)

    # Détecte les "peignes" (valeurs nulles alternées) → indice de manipulation
    zeros = np.sum(hist == 0)
    spikes = np.sum(hist > hist.mean() + 3 * hist.std())

    score = min(1.0, (zeros / 256.0) * 2 + (spikes / 50.0))

    if zeros < 10 and spikes < 5:
        desc = f"Histogramme normal (lacunes={zeros}, pics={spikes})"
    elif zeros < 30:
        desc = f"Histogramme légèrement irrégulier (lacunes={zeros}, pics={spikes})"
    else:
        desc = f"Histogramme avec motif en peigne (lacunes={zeros}) — retouches tonales probables"

    return score, desc


def check_exif_metadata(pil_img: Image.Image) -> list[Finding]:
    """Inspecte les métadonnées EXIF pour détecter des incohérences."""
    findings = []
    try:
        exif_data = pil_img._getexif() if hasattr(pil_img, '_getexif') else None
    except Exception:
        exif_data = None

    if exif_data is None:
        findings.append(Finding(
            technique="EXIF",
            severity="info",
            description="Aucune métadonnée EXIF trouvée (normal pour images extraites de PDF).",
            score=0.0,
        ))
        return findings

    # Tags EXIF intéressants
    TAG_SOFTWARE = 305
    TAG_DATETIME = 306
    TAG_DATETIME_ORIGINAL = 36867

    software = exif_data.get(TAG_SOFTWARE, "")
    dt = exif_data.get(TAG_DATETIME, "")
    dt_orig = exif_data.get(TAG_DATETIME_ORIGINAL, "")

    if software:
        editing_keywords = ["photoshop", "gimp", "lightroom", "affinity", "capture one",
                            "illustrator", "inkscape", "paint", "pixelmator"]
        if any(k in software.lower() for k in editing_keywords):
            findings.append(Finding(
                technique="EXIF",
                severity="warning",
                description=f"Logiciel de retouche détecté : {software}",
                score=0.5,
            ))
        else:
            findings.append(Finding(
                technique="EXIF",
                severity="info",
                description=f"Logiciel : {software}",
                score=0.0,
            ))

    if dt and dt_orig and dt != dt_orig:
        findings.append(Finding(
            technique="EXIF",
            severity="warning",
            description=f"Date de modification ({dt}) ≠ date de capture ({dt_orig})",
            score=0.4,
        ))

    return findings


def analyze_image(img_data: dict, ela_quality: int = 95) -> ImageReport:
    """Analyse complète d'une image extraite du PDF."""
    pil = img_data["pil"]
    page = img_data["page"]
    idx = img_data["index"]
    ext = img_data["ext"]

    report = ImageReport(
        page=page,
        index=idx,
        filename=f"page{page}_img{idx}.{ext}",
        width=pil.width,
        height=pil.height,
        mode=pil.mode,
    )

    # ── Miniature ──────────────────────────────────────────────────────────────
    thumb = pil.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=80)
    report.thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    # ── 1. ELA ─────────────────────────────────────────────────────────────────
    try:
        ela_img, ela_score = run_ela(pil, quality=ela_quality)
        buf = io.BytesIO()
        ela_img.save(buf, format="PNG")
        report.ela_b64 = base64.b64encode(buf.getvalue()).decode()

        if ela_score < 0.15:
            sev, desc = "info", f"ELA normal (score={ela_score:.3f}) — pas d'anomalie évidente"
        elif ela_score < 0.35:
            sev, desc = "warning", f"ELA modéré (score={ela_score:.3f}) — zones potentiellement retouchées"
        else:
            sev, desc = "alert", f"ELA élevé (score={ela_score:.3f}) — forte suspicion de manipulation"

        report.findings.append(Finding("ELA", sev, desc, ela_score))
    except Exception as e:
        report.findings.append(Finding("ELA", "info", f"ELA non disponible : {e}", 0.0))

    # ── 2. Cohérence du bruit ──────────────────────────────────────────────────
    try:
        noise_score, noise_desc = analyze_noise_consistency(pil)
        sev = "alert" if noise_score > 0.6 else "warning" if noise_score > 0.3 else "info"
        report.findings.append(Finding("Bruit", sev, noise_desc, noise_score))
    except Exception as e:
        report.findings.append(Finding("Bruit", "info", f"Analyse bruit impossible : {e}", 0.0))

    # ── 3. Détection de clonage ────────────────────────────────────────────────
    if pil.width > 100 and pil.height > 100:
        try:
            n_matches, clone_desc = detect_copy_move(pil)
            score = min(1.0, n_matches / 100.0)
            sev = "alert" if n_matches >= 50 else "warning" if n_matches >= 20 else "info"
            report.findings.append(Finding("Copy-Move", sev, clone_desc, score))
        except Exception as e:
            report.findings.append(Finding("Copy-Move", "info", f"Analyse clonage impossible : {e}", 0.0))

    # ── 4. Histogramme ─────────────────────────────────────────────────────────
    try:
        hist_score, hist_desc = analyze_histogram(pil)
        sev = "alert" if hist_score > 0.5 else "warning" if hist_score > 0.2 else "info"
        report.findings.append(Finding("Histogramme", sev, hist_desc, hist_score))
    except Exception as e:
        report.findings.append(Finding("Histogramme", "info", f"Analyse histogramme impossible : {e}", 0.0))

    # ── 5. Métadonnées EXIF ────────────────────────────────────────────────────
    report.findings.extend(check_exif_metadata(pil))

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Génération du rapport HTML
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_COLOR = {
    "ok":      ("#d4edda", "#155724", "✅"),
    "info":    ("#d1ecf1", "#0c5460", "ℹ️"),
    "warning": ("#fff3cd", "#856404", "⚠️"),
    "alert":   ("#f8d7da", "#721c24", "🚨"),
}

def severity_label(s: str) -> str:
    return {"ok": "OK", "info": "Info", "warning": "Attention", "alert": "Alerte"}.get(s, s)


def generate_html_report(pdf_path: str, reports: list[ImageReport], output_path: str) -> None:
    """Génère un rapport HTML interactif."""
    alerts   = sum(1 for r in reports if r.max_severity == "alert")
    warnings = sum(1 for r in reports if r.max_severity == "warning")
    ok_count = len(reports) - alerts - warnings

    rows = ""
    for r in reports:
        bg, fg, icon = SEVERITY_COLOR.get(r.max_severity, SEVERITY_COLOR["info"])
        findings_html = "".join(
            f'<li><b>[{f.technique}]</b> {f.description} '
            f'<span style="color:#999">(score={f.score:.2f})</span></li>'
            for f in r.findings
        )
        ela_html = (
            f'<img src="data:image/png;base64,{r.ela_b64}" '
            f'style="max-width:200px;border:1px solid #ccc" title="ELA">'
        ) if r.ela_b64 else "<em>N/A</em>"

        thumb_html = (
            f'<img src="data:image/jpeg;base64,{r.thumb_b64}" '
            f'style="max-width:200px;border:1px solid #ccc" title="Original">'
        ) if r.thumb_b64 else ""

        rows += f"""
        <tr style="background:{bg};color:{fg}">
          <td style="padding:8px;font-size:1.3em">{icon}</td>
          <td style="padding:8px"><b>{r.filename}</b><br>Page {r.page} — {r.width}×{r.height}px — {r.mode}</td>
          <td style="padding:8px">{thumb_html}</td>
          <td style="padding:8px">{ela_html}</td>
          <td style="padding:8px"><ul style="margin:0;padding-left:16px">{findings_html}</ul></td>
          <td style="padding:8px;text-align:center">
            <b style="font-size:1.2em">{r.suspicion_score:.0%}</b><br>
            <span style="font-size:0.8em">{severity_label(r.max_severity)}</span>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport Forensique — {Path(pdf_path).name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
    h1 {{ color: #2c3e50; }}
    .summary {{ display:flex; gap:20px; margin-bottom:20px; flex-wrap:wrap; }}
    .card {{ padding:16px; border-radius:8px; min-width:120px; text-align:center; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background:#2c3e50; color:white; padding:10px; text-align:left; }}
    td {{ vertical-align:top; border-bottom:1px solid #eee; }}
    .legend {{ font-size:0.85em; color:#666; margin-top:16px; }}
  </style>
</head>
<body>
  <h1>🔬 Rapport d'analyse forensique d'images</h1>
  <p><b>Fichier PDF :</b> {Path(pdf_path).name}<br>
     <b>Images analysées :</b> {len(reports)}</p>

  <div class="summary">
    <div class="card" style="background:#f8d7da;color:#721c24">
      <div style="font-size:2em">{alerts}</div>🚨 Alertes
    </div>
    <div class="card" style="background:#fff3cd;color:#856404">
      <div style="font-size:2em">{warnings}</div>⚠️ Avertissements
    </div>
    <div class="card" style="background:#d4edda;color:#155724">
      <div style="font-size:2em">{ok_count}</div>✅ OK
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th></th>
        <th>Image</th>
        <th>Original</th>
        <th>ELA</th>
        <th>Analyse détaillée</th>
        <th>Score suspicion</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <div class="legend">
    <b>Légende des techniques :</b><br>
    • <b>ELA</b> (Error Level Analysis) : re-compression JPEG — zones brillantes = potentiellement ajoutées/modifiées<br>
    • <b>Bruit</b> : cohérence du bruit sur l'image — incohérences = zones composites<br>
    • <b>Copy-Move</b> : détection de régions clonées à l'intérieur de l'image<br>
    • <b>Histogramme</b> : analyse tonale — motifs en peigne = retouches des courbes<br>
    • <b>EXIF</b> : métadonnées — présence de logiciels de retouche, dates incohérentes<br>
    <br>
    <b>⚠️ Important :</b> Ces analyses sont des indices, <em>pas des preuves</em>.
    Un score élevé nécessite une investigation humaine approfondie.
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse les images d'un PDF scientifique pour détecter des manipulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", help="Chemin vers le fichier PDF")
    parser.add_argument("--output", "-o", default="rapport_forensique.html",
                        help="Fichier HTML de sortie (défaut : rapport_forensique.html)")
    parser.add_argument("--ela-quality", type=int, default=95,
                        help="Qualité JPEG pour l'ELA, 80–95 (défaut : 95)")
    parser.add_argument("--min-size", type=int, default=50,
                        help="Taille minimale (px) des images à analyser (défaut : 50)")
    parser.add_argument("--json", action="store_true",
                        help="Affiche aussi un résumé JSON sur stdout")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        sys.exit(f"❌  Fichier introuvable : {args.pdf}")

    print(f"📄  Extraction des images de : {args.pdf}")
    images = extract_images_from_pdf(args.pdf, min_size=args.min_size)

    if not images:
        sys.exit("⚠️  Aucune image trouvée dans ce PDF.")

    print(f"🖼   {len(images)} image(s) trouvée(s) — analyse en cours…\n")

    reports = []
    for i, img_data in enumerate(images, 1):
        label = f"Page {img_data['page']}, image {img_data['index']}"
        print(f"  [{i}/{len(images)}] {label} ({img_data['pil'].width}×{img_data['pil'].height})")
        report = analyze_image(img_data, ela_quality=args.ela_quality)
        reports.append(report)

        for f in report.findings:
            icon = {"info": "  ℹ ", "warning": "  ⚠ ", "alert": "  🚨"}.get(f.severity, "  • ")
            print(f"{icon} [{f.technique}] {f.description}")
        print()

    print(f"📝  Génération du rapport HTML : {args.output}")
    generate_html_report(args.pdf, reports, args.output)

    # Résumé console
    alerts   = sum(1 for r in reports if r.max_severity == "alert")
    warnings = sum(1 for r in reports if r.max_severity == "warning")
    print(f"\n{'='*50}")
    print(f"  Résultat : {len(reports)} images analysées")
    print(f"  🚨 Alertes    : {alerts}")
    print(f"  ⚠️  Avertissements : {warnings}")
    print(f"  ✅ OK         : {len(reports) - alerts - warnings}")
    print(f"{'='*50}")
    print(f"\n  → Rapport complet : {args.output}\n")

    if args.json:
        summary = [
            {
                "filename": r.filename,
                "page": r.page,
                "max_severity": r.max_severity,
                "suspicion_score": round(r.suspicion_score, 3),
                "findings": [
                    {"technique": f.technique, "severity": f.severity,
                     "description": f.description, "score": round(f.score, 3)}
                    for f in r.findings
                ],
            }
            for r in reports
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
