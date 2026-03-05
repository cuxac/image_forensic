#!/usr/bin/env python3
"""
PDF Image Forensics Tool
========================
Analyse les images d'un article scientifique (PDF) pour détecter
d'éventuelles manipulations.

Techniques :
  - ELA (Error Level Analysis)
  - Cohérence du bruit (Laplacien)
  - Copy-move intra-image (ORB)
  - Duplications inter-panneaux  ← CLEF pour détecter Fig.1D, 2E, 2F
  - Analyse de l'histogramme
  - Métadonnées EXIF
  - Détection automatique du numéro de figure dans le texte PDF

Installation :
    pip install pymupdf Pillow opencv-python numpy scipy

Usage :
    python pdf_image_forensics.py article.pdf
    python pdf_image_forensics.py article.pdf -o rapport.html --ela-quality 90
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Dépendances ────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("❌  PyMuPDF manquant — pip install pymupdf")

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance
    from scipy import ndimage
except ImportError as e:
    sys.exit(f"❌  {e}\n   pip install opencv-python numpy Pillow scipy")


# ══════════════════════════════════════════════════════════════════════════════
# Structures de données
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Finding:
    technique: str
    severity: str   # "info" | "warning" | "alert"
    description: str
    score: float = 0.0

@dataclass
class ImageReport:
    page: int
    index: int
    filename: str
    width: int
    height: int
    mode: str
    figure_label: str = ""
    findings: list = field(default_factory=list)
    ela_b64: str = ""
    thumb_b64: str = ""

    @property
    def max_severity(self):
        order = {"alert": 3, "warning": 2, "info": 1}
        if not self.findings:
            return "ok"
        return max(self.findings, key=lambda f: order.get(f.severity, 0)).severity

    @property
    def suspicion_score(self):
        if not self.findings:
            return 0.0
        return min(1.0, sum(f.score for f in self.findings) / max(1, len(self.findings)) * 2)

    @property
    def display_label(self):
        if self.figure_label:
            return f"{self.figure_label} — Page {self.page}"
        return f"Page {self.page}, image #{self.index}"


# ══════════════════════════════════════════════════════════════════════════════
# Détection des numéros de figure
# ══════════════════════════════════════════════════════════════════════════════

_FIGURE_RE = re.compile(
    r'(?:Supplementary\s+)?(?:FIGURE|Figure|Fig\.?)\s*([0-9]+[A-Za-z]?)',
    re.IGNORECASE
)

def build_figure_map(doc):
    """Retourne {page_1based: [labels]} depuis le texte de chaque page."""
    figure_map = {}
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        matches = _FIGURE_RE.findall(text)
        if matches:
            seen = []
            for m in matches:
                lbl = f"Figure {m}"
                if lbl not in seen:
                    seen.append(lbl)
            figure_map[page_num + 1] = seen
    return figure_map

def assign_figure_labels(images, figure_map):
    """
    Attribue un label complet à chaque image.
    Ex : "Figure 1 (panneaux A, B, C, E)"
    Cherche sur la page courante, puis pages adjacentes.
    """
    by_page = {}
    for img in images:
        by_page.setdefault(img["page"], []).append(img)

    for page, page_images in sorted(by_page.items()):
        labels = (figure_map.get(page)
                  or figure_map.get(page - 1)
                  or figure_map.get(page + 1)
                  or [])

        if labels:
            # Numéros de base (ex: "1" depuis "Figure 1A")
            base_nums = []
            for lbl in labels:
                m = re.match(r'Figure\s*(\d+)', lbl, re.IGNORECASE)
                if m and m.group(1) not in base_nums:
                    base_nums.append(m.group(1))
            # Lettres de panneaux
            panels = []
            for lbl in labels:
                m = re.match(r'Figure\s*\d+([A-Za-z])', lbl, re.IGNORECASE)
                if m and m.group(1).upper() not in panels:
                    panels.append(m.group(1).upper())

            fig_nums = ", ".join(f"Figure {n}" for n in base_nums)
            if panels:
                fig_label = f"{fig_nums} (panneaux {', '.join(sorted(panels))})"
            else:
                fig_label = fig_nums
        else:
            fig_label = ""

        for img in page_images:
            img["figure_label"] = fig_label


# ══════════════════════════════════════════════════════════════════════════════
# Extraction des images
# ══════════════════════════════════════════════════════════════════════════════

def extract_images_from_pdf(pdf_path, min_size=50):
    doc = fitz.open(pdf_path)
    images = []
    seen_hashes = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_hash = hashlib.md5(img_bytes).hexdigest()
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)
                pil_img = Image.open(io.BytesIO(img_bytes))
                w, h = pil_img.size
                if w < min_size or h < min_size:
                    continue
                images.append({
                    "page": page_num + 1,
                    "index": img_idx,
                    "ext": base_image["ext"],
                    "bytes": img_bytes,
                    "pil": pil_img,
                    "figure_label": "",
                })
            except Exception as e:
                print(f"  ⚠  Page {page_num+1}, img {img_idx}: {e}", file=sys.stderr)

    return images, doc


# ══════════════════════════════════════════════════════════════════════════════
# Techniques d'analyse
# ══════════════════════════════════════════════════════════════════════════════

def run_ela(pil_img, quality=95):
    """Error Level Analysis."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    ela_img = ImageChops.difference(pil_img, recompressed)
    max_diff = max(max(e) for e in ela_img.getextrema()) or 1
    ela_img = ImageEnhance.Brightness(ela_img).enhance(255.0 / max_diff)
    score = min(1.0, float(np.std(np.array(ela_img).astype(float)) / 128.0))
    return ela_img, score


def analyze_noise_consistency(pil_img):
    """Cohérence du bruit local (Laplacien, grille 4×4)."""
    gray = np.array(pil_img.convert("L")).astype(float)
    laplacian = ndimage.laplace(gray)
    h, w = laplacian.shape
    bh, bw = h // 4, w // 4
    variances = [
        float(np.var(laplacian[i*bh:(i+1)*bh, j*bw:(j+1)*bw]))
        for i in range(4) for j in range(4)
    ]
    if not variances:
        return 0.0, "Impossible de calculer"
    cv = np.std(variances) / (np.mean(variances) + 1e-9)
    score = min(1.0, cv / 5.0)
    if cv < 1.5:
        desc = f"Bruit homogène (CV={cv:.2f})"
    elif cv < 3.0:
        desc = f"Légères incohérences de bruit (CV={cv:.2f})"
    else:
        desc = f"Incohérences importantes de bruit (CV={cv:.2f}) — zones composites probables"
    return score, desc


def detect_copy_move(pil_img, min_matches=20):
    """Clonage intra-image via ORB."""
    gray = np.array(pil_img.convert("L"))
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 10:
        return 0, "Pas assez de points caractéristiques"
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    suspicious = 0
    for m_list in bf.knnMatch(des, des, k=2):
        if len(m_list) < 2:
            continue
        m, n = m_list[0], m_list[1]
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < 0.75 * n.distance:
            pt1 = np.array(kp[m.queryIdx].pt)
            pt2 = np.array(kp[m.trainIdx].pt)
            if np.linalg.norm(pt1 - pt2) > 20:
                suspicious += 1
    if suspicious < min_matches:
        desc = f"Aucun clonage détecté ({suspicious} correspondances)"
    elif suspicious < 50:
        desc = f"Correspondances modérées ({suspicious}) — à vérifier"
    else:
        desc = f"Nombreuses correspondances ({suspicious}) — clonage probable"
    return suspicious, desc


def detect_cross_panel_duplicates(pil_img):
    """
    Détecte les duplications ENTRE panneaux d'une figure composite.

    Découpe l'image en colonnes (2, 3, 4 ou 6) et compare chaque paire
    via ORB feature matching. Des panneaux montrant des conditions
    expérimentales différentes ne devraient PAS partager autant de features.

    Seuils : ≥20 = warning, ≥40 = alerte, ≥80 = alerte forte.
    """
    findings = []
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape

    if w < 400 or h < 200:
        return findings, ""

    orb = cv2.ORB_create(nfeatures=800)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_score = 0
    best_pairs = []
    best_ncols = 4

    for ncols in [2, 3, 4, 6]:
        col_w = w // ncols
        if col_w < 80:
            continue
        # Extraire les descripteurs de chaque colonne
        descriptors = []
        for c in range(ncols):
            crop = gray[:, c * col_w:(c + 1) * col_w]
            _, des = orb.detectAndCompute(crop, None)
            descriptors.append(des)
        # Comparer toutes les paires
        pairs = []
        for i in range(ncols):
            for j in range(i + 1, ncols):
                di, dj = descriptors[i], descriptors[j]
                if di is None or dj is None:
                    continue
                good = [m for m in bf.match(di, dj) if m.distance < 50]
                pairs.append((i, j, len(good)))
        if pairs:
            mx = max(p[2] for p in pairs)
            if mx > best_score:
                best_score = mx
                best_ncols = ncols
                best_pairs = [(i, j, n) for i, j, n in pairs if n >= 20]

    if not best_pairs:
        findings.append(Finding(
            "Inter-panneaux", "info",
            f"Aucune duplication inter-panneaux détectée (max={best_score} corr.)",
            score=0.0,
        ))
        return findings, ""

    best_pairs.sort(key=lambda x: -x[2])
    summary_parts = []

    for col_i, col_j, n in best_pairs[:6]:
        pi = chr(ord('A') + col_i)
        pj = chr(ord('A') + col_j)
        if n >= 80:
            sev, msg = "alert", (
                f"Inter-panneaux col.{pi} ≈ col.{pj} ({n} corr.) "
                f"— images très probablement dupliquées"
            )
        elif n >= 40:
            sev, msg = "alert", (
                f"Inter-panneaux col.{pi} ≈ col.{pj} ({n} corr.) "
                f"— similarité anormalement élevée"
            )
        else:
            sev, msg = "warning", (
                f"Inter-panneaux col.{pi} ↔ col.{pj} ({n} corr.) — à vérifier"
            )
        findings.append(Finding("Inter-panneaux", sev, msg, score=min(1.0, n / 200.0)))
        summary_parts.append(f"{pi}↔{pj}({n})")

    remaining = len(best_pairs) - 6
    if remaining > 0:
        findings.append(Finding(
            "Inter-panneaux", "alert",
            f"… et {remaining} autres paires suspectes "
            f"(config optimale : {best_ncols} colonnes)",
            score=min(1.0, best_score / 200.0),
        ))

    return findings, "Paires suspectes : " + ", ".join(summary_parts)


def analyze_histogram(pil_img):
    """Analyse de l'histogramme (peigne, pics)."""
    gray = np.array(pil_img.convert("L"))
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float)
    zeros  = int(np.sum(hist == 0))
    spikes = int(np.sum(hist > hist.mean() + 3 * hist.std()))
    score  = min(1.0, (zeros / 256.0) * 2 + (spikes / 50.0))
    if zeros < 10 and spikes < 5:
        desc = f"Histogramme normal (lacunes={zeros}, pics={spikes})"
    elif zeros < 30:
        desc = f"Histogramme légèrement irrégulier (lacunes={zeros}, pics={spikes})"
    else:
        desc = f"Histogramme avec motif en peigne (lacunes={zeros}) — retouches tonales probables"
    return score, desc


def check_exif_metadata(pil_img):
    """Inspection des métadonnées EXIF."""
    findings = []
    try:
        exif_data = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
    except Exception:
        exif_data = None

    if exif_data is None:
        findings.append(Finding("EXIF", "info",
            "Aucune métadonnée EXIF (normal pour images extraites de PDF).", 0.0))
        return findings

    software = exif_data.get(305, "")
    dt       = exif_data.get(306, "")
    dt_orig  = exif_data.get(36867, "")

    if software:
        editing_kw = ["photoshop", "gimp", "lightroom", "affinity",
                      "capture one", "illustrator", "inkscape", "paint", "pixelmator"]
        if any(k in software.lower() for k in editing_kw):
            findings.append(Finding("EXIF", "warning",
                f"Logiciel de retouche détecté : {software}", 0.5))
        else:
            findings.append(Finding("EXIF", "info", f"Logiciel : {software}", 0.0))

    if dt and dt_orig and dt != dt_orig:
        findings.append(Finding("EXIF", "warning",
            f"Date modification ({dt}) ≠ date capture ({dt_orig})", 0.4))

    return findings


def analyze_image(img_data, ela_quality=95):
    """Analyse forensique complète d'une image."""
    pil = img_data["pil"]

    report = ImageReport(
        page=img_data["page"],
        index=img_data["index"],
        filename=f"page{img_data['page']}_img{img_data['index']}.{img_data['ext']}",
        width=pil.width,
        height=pil.height,
        mode=pil.mode,
        figure_label=img_data.get("figure_label", ""),
    )

    # Miniature
    thumb = pil.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=80)
    report.thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    # 1. ELA
    try:
        ela_img, ela_score = run_ela(pil, quality=ela_quality)
        buf = io.BytesIO()
        ela_img.save(buf, format="PNG")
        report.ela_b64 = base64.b64encode(buf.getvalue()).decode()
        if ela_score < 0.15:
            sev, desc = "info", f"ELA normal (score={ela_score:.3f})"
        elif ela_score < 0.35:
            sev, desc = "warning", f"ELA modéré (score={ela_score:.3f}) — zones potentiellement retouchées"
        else:
            sev, desc = "alert", f"ELA élevé (score={ela_score:.3f}) — forte suspicion de manipulation"
        report.findings.append(Finding("ELA", sev, desc, ela_score))
    except Exception as e:
        report.findings.append(Finding("ELA", "info", f"ELA non disponible : {e}", 0.0))

    # 2. Cohérence du bruit
    try:
        ns, nd = analyze_noise_consistency(pil)
        sev = "alert" if ns > 0.6 else "warning" if ns > 0.3 else "info"
        report.findings.append(Finding("Bruit", sev, nd, ns))
    except Exception as e:
        report.findings.append(Finding("Bruit", "info", f"Bruit non calculable : {e}", 0.0))

    # 3. Copy-move intra-image
    if pil.width > 100 and pil.height > 100:
        try:
            nm, cd = detect_copy_move(pil)
            score = min(1.0, nm / 100.0)
            sev = "alert" if nm >= 50 else "warning" if nm >= 20 else "info"
            report.findings.append(Finding("Copy-Move", sev, cd, score))
        except Exception as e:
            report.findings.append(Finding("Copy-Move", "info", f"Analyse impossible : {e}", 0.0))

    # 4. Duplications inter-panneaux  ← DÉTECTE Fig.1D, 2E, 2F
    if pil.width > 400 and pil.height > 200:
        try:
            ip_findings, _ = detect_cross_panel_duplicates(pil)
            report.findings.extend(ip_findings)
        except Exception as e:
            report.findings.append(Finding("Inter-panneaux", "info",
                f"Analyse impossible : {e}", 0.0))

    # 5. Histogramme
    try:
        hs, hd = analyze_histogram(pil)
        sev = "alert" if hs > 0.5 else "warning" if hs > 0.2 else "info"
        report.findings.append(Finding("Histogramme", sev, hd, hs))
    except Exception as e:
        report.findings.append(Finding("Histogramme", "info", f"Analyse impossible : {e}", 0.0))

    # 6. EXIF
    report.findings.extend(check_exif_metadata(pil))

    return report


# ══════════════════════════════════════════════════════════════════════════════
# Rapport HTML
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_COLOR = {
    "ok":      ("#d4edda", "#155724", "✅"),
    "info":    ("#d1ecf1", "#0c5460", "ℹ️"),
    "warning": ("#fff3cd", "#856404", "⚠️"),
    "alert":   ("#f8d7da", "#721c24", "🚨"),
}

def severity_label(s):
    return {"ok": "OK", "info": "Info", "warning": "Attention", "alert": "Alerte"}.get(s, s)


def generate_html_report(pdf_path, reports, output_path):
    alerts   = sum(1 for r in reports if r.max_severity == "alert")
    warnings = sum(1 for r in reports if r.max_severity == "warning")
    ok_count = len(reports) - alerts - warnings

    rows = ""
    for r in reports:
        bg, fg, icon = SEVERITY_COLOR.get(r.max_severity, SEVERITY_COLOR["info"])

        # Badge figure
        if r.figure_label:
            fig_badge = (
                f'<span style="display:inline-block;background:#2c3e50;color:white;'
                f'padding:2px 9px;border-radius:10px;font-size:.83em;font-weight:bold;'
                f'margin-bottom:5px">{r.figure_label}</span><br>'
            )
        else:
            fig_badge = (
                f'<span style="display:inline-block;background:#999;color:white;'
                f'padding:2px 9px;border-radius:10px;font-size:.83em;margin-bottom:5px">'
                f'Figure inconnue</span><br>'
            )

        identity_html = (
            f'{fig_badge}'
            f'<b>Page {r.page}</b> · image #{r.index}<br>'
            f'<small style="color:#888">{r.filename}<br>{r.width}×{r.height}px · {r.mode}</small>'
        )

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
            f'style="max-width:200px;border:1px solid #ccc">'
        ) if r.thumb_b64 else ""

        rows += f"""
        <tr style="background:{bg};color:{fg}">
          <td style="padding:8px;font-size:1.3em;text-align:center">{icon}</td>
          <td style="padding:8px">{identity_html}</td>
          <td style="padding:8px">{thumb_html}</td>
          <td style="padding:8px">{ela_html}</td>
          <td style="padding:8px"><ul style="margin:0;padding-left:16px">{findings_html}</ul></td>
          <td style="padding:8px;text-align:center">
            <b style="font-size:1.2em">{r.suspicion_score:.0%}</b><br>
            <span style="font-size:.8em">{severity_label(r.max_severity)}</span>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport Forensique — {Path(pdf_path).name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; background:#f8f9fa; }}
    h1 {{ color: #2c3e50; }}
    .summary {{ display:flex; gap:20px; margin-bottom:20px; flex-wrap:wrap; }}
    .card {{ padding:16px 22px; border-radius:10px; min-width:130px; text-align:center;
             box-shadow:0 2px 6px rgba(0,0,0,.1); }}
    .card .num {{ font-size:2.2em; font-weight:bold; margin-bottom:4px; }}
    table {{ border-collapse:collapse; width:100%; background:white;
             border-radius:10px; overflow:hidden; box-shadow:0 2px 6px rgba(0,0,0,.1); }}
    th {{ background:#2c3e50; color:white; padding:10px; text-align:left; }}
    td {{ vertical-align:top; border-bottom:1px solid #eee; }}
    .legend {{ font-size:.85em; color:#666; margin-top:20px; background:white;
               padding:14px 18px; border-radius:8px; line-height:1.8; }}
  </style>
</head>
<body>
  <h1>🔬 Rapport d'analyse forensique d'images</h1>
  <p><b>Fichier PDF :</b> {Path(pdf_path).name} &nbsp;|&nbsp; <b>Images analysées :</b> {len(reports)}</p>

  <div class="summary">
    <div class="card" style="background:#f8d7da;color:#721c24">
      <div class="num">{alerts}</div>🚨 Alertes
    </div>
    <div class="card" style="background:#fff3cd;color:#856404">
      <div class="num">{warnings}</div>⚠️ Avertissements
    </div>
    <div class="card" style="background:#d4edda;color:#155724">
      <div class="num">{ok_count}</div>✅ OK
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th style="width:36px"></th>
        <th>Figure · Page</th>
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
    • <b>ELA</b> : zones re-compressées différemment → copier-coller intra-image possible<br>
    • <b>Bruit</b> : incohérences du bruit local → zones d'origines différentes collées ensemble<br>
    • <b>Copy-Move</b> : régions dupliquées à l'intérieur d'une même image<br>
    • <b>Inter-panneaux</b> : similarité anormale entre colonnes d'une figure composite —
      deux conditions expérimentales ne devraient pas partager autant de features visuels<br>
    • <b>Histogramme</b> : motif en peigne → retouche des courbes tonales<br>
    • <b>EXIF</b> : logiciel de retouche ou dates incohérentes dans les métadonnées<br>
    <br>
    <b>ℹ️  Numéros de figure</b> extraits automatiquement du texte du PDF
    (patterns "Figure N", "Fig. N", "FIGURE N"). Les panneaux listés (A, B, C…)
    sont ceux mentionnés dans la légende sur la même page.<br>
    <br>
    <b>⚠️ Important :</b> Ces analyses sont des <em>indices</em>, pas des preuves.
    Tout score élevé doit être vérifié par un expert.
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
        description="Analyse les images d'un PDF scientifique pour détecter des manipulations."
    )
    parser.add_argument("pdf")
    parser.add_argument("--output", "-o", default="rapport_forensique.html")
    parser.add_argument("--ela-quality", type=int, default=95)
    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        sys.exit(f"❌  Fichier introuvable : {args.pdf}")

    # 1. Extraction
    print(f"📄  Extraction des images de : {args.pdf}")
    images, doc = extract_images_from_pdf(args.pdf, min_size=args.min_size)
    if not images:
        doc.close()
        sys.exit("⚠️  Aucune image trouvée.")

    # 2. Détection des figures
    print("🔎  Détection des numéros de figure dans le texte…")
    figure_map = build_figure_map(doc)
    assign_figure_labels(images, figure_map)
    doc.close()

    labeled = sum(1 for img in images if img["figure_label"])
    print(f"    → {labeled}/{len(images)} image(s) associée(s) à un numéro de figure")
    for page, labels in sorted(figure_map.items()):
        print(f"    Page {page:3d} : {', '.join(labels)}")

    print(f"\n🖼   {len(images)} image(s) à analyser…\n")

    # 3. Analyse forensique
    reports = []
    for i, img_data in enumerate(images, 1):
        fig_str = f" [{img_data['figure_label']}]" if img_data["figure_label"] else ""
        print(f"  [{i}/{len(images)}] Page {img_data['page']}{fig_str} "
              f"— {img_data['pil'].width}×{img_data['pil'].height}px")
        report = analyze_image(img_data, ela_quality=args.ela_quality)
        reports.append(report)
        for f in report.findings:
            icon = {"info": "  ℹ ", "warning": "  ⚠ ", "alert": "  🚨"}.get(f.severity, "  • ")
            print(f"{icon} [{f.technique}] {f.description}")
        print()

    # 4. Rapport HTML
    print(f"📝  Génération du rapport : {args.output}")
    generate_html_report(args.pdf, reports, args.output)

    alerts   = sum(1 for r in reports if r.max_severity == "alert")
    warnings = sum(1 for r in reports if r.max_severity == "warning")
    print(f"\n{'='*52}")
    print(f"  {len(reports)} images analysées")
    print(f"  🚨 Alertes         : {alerts}")
    print(f"  ⚠️  Avertissements : {warnings}")
    print(f"  ✅ OK              : {len(reports) - alerts - warnings}")
    print(f"{'='*52}")
    print(f"\n  → {args.output}\n")

    if args.json:
        print(json.dumps([
            {
                "filename":      r.filename,
                "page":          r.page,
                "figure_label":  r.figure_label,
                "display_label": r.display_label,
                "max_severity":  r.max_severity,
                "suspicion_score": round(r.suspicion_score, 3),
                "findings": [
                    {"technique": f.technique, "severity": f.severity,
                     "description": f.description, "score": round(f.score, 3)}
                    for f in r.findings
                ],
            }
            for r in reports
        ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()