# PDF Image Forensics

Outil d'analyse d'images scientifiques dans des articles PDF. Détecte automatiquement les manipulations potentielles : images dupliquées entre panneaux, clonage de régions, retouches de niveaux, etc.

---

## Installation

```bash
pip install pymupdf pypdf opencv-python numpy Pillow scipy pypdfium2 pdfplumber
```

| Paquet | Rôle |
|--------|------|
| `pymupdf` | Extraction des images raster depuis le PDF |
| `pypdf` | Extraction du texte (numéros de figures) |
| `opencv-python` | Feature matching ORB (copy-move, inter-panneaux) |
| `numpy` / `Pillow` / `scipy` | Traitement d'image (ELA, bruit, histogramme) |
| `pypdfium2` | Rastérisation des pages vectorielles (graphiques, courbes) |
| `pdfplumber` | Localisation des légendes dans le texte PDF |

> `pypdfium2` remplace `pdf2image`/poppler — aucune dépendance système requise.

---

## Usage

```bash
python pdf_image_forensics.py article.pdf
python pdf_image_forensics.py article.pdf -o rapport.html
python pdf_image_forensics.py article.pdf --ela-quality 90 --min-size 100
```

### Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `--output / -o` | `rapport_forensique.html` | Fichier HTML de sortie |
| `--ela-quality` | `95` | Qualité JPEG pour l'ELA (80–95) |
| `--min-size` | `50` | Taille minimale des images en pixels |
| `--max-ratio` | `5.0` | Ratio max largeur/hauteur (filtre logos/bandeaux) |
| `--json` | — | Affiche aussi un résumé JSON sur stdout |

---

## Techniques d'analyse

### 1. ELA — Error Level Analysis
Re-compression JPEG à qualité fixe et comparaison pixel-à-pixel avec l'original. Les zones dont le niveau d'erreur diffère fortement du reste de l'image ont probablement été modifiées ou ajoutées après coup.

**Seuils :** score < 0.15 → normal · 0.15–0.35 → à vérifier · > 0.35 → alerte

### 2. Cohérence du bruit
Calcul du Laplacien sur une grille 4×4. Un bruit homogène indique une image non composite ; des variations importantes entre zones suggèrent un assemblage de régions d'origines différentes.

### 3. Copy-Move intra-image
Détection de régions dupliquées *à l'intérieur* d'une même image via ORB (Oriented FAST and Rotated BRIEF) et Lowe's ratio test. Les correspondances spatiales distantes de plus de 20 px sont comptées.

### 4. Inter-panneaux *(images raster uniquement)*
La technique clé pour les figures composites (ex. transwell assay, Western blot). L'image est découpée en colonnes (2, 3, 4 ou 6 — la configuration optimale est sélectionnée automatiquement) et chaque paire de colonnes est comparée par feature matching. Deux conditions expérimentales différentes ne devraient pas partager autant de points caractéristiques.

**Seuils :** < 20 → normal · 20–40 → à vérifier · 40–80 → alerte · > 80 → alerte forte

> ⚠️ Non appliqué aux figures vectorielles rastérisées : les axes, marges et textes communs génèrent trop de faux positifs.

### 5. Histogramme
Détection du motif « en peigne » (valeurs nulles alternées) caractéristique d'une retouche des courbes tonales ou d'une conversion de profil colorimétrique.

### 6. EXIF
Présence d'un logiciel de retouche (Photoshop, GIMP, Lightroom…) ou incohérence entre date de capture et date de modification.

---

## Détection automatique des figures

Le script extrait le texte de chaque page pour y trouver les labels de figures via une regex :

```
Figure 1 · Figure 2A · Fig. 3 · Figure S1 · Figure S10 · Supplementary Figure 2
```

Le numéro de figure est affiché dans la console et dans le rapport HTML :

```
[2/5] Page 5 [Figure 1 (panneaux A, B, C, E)] — 1980×1172px
```

Les panneaux listés dans la légende (A, B, C…) sont identifiés automatiquement. Un panneau **absent** de cette liste mais présent visuellement dans l'image est lui-même un indice.

---

## Filtrage automatique

- **Doublons** : les images identiques (même hash MD5) ne sont analysées qu'une seule fois — typiquement le logo du journal présent sur chaque page.
- **Bandeaux/logos** : les images avec un ratio largeur/hauteur > 5 (ou < 0.2) sont exclues automatiquement. Réglable via `--max-ratio`.

---

## Figures vectorielles

Les graphiques scientifiques (courbes XRD, spectres XPS, courbes I-V…) sont dessinés directement dans le flux PDF et ne contiennent pas d'image raster. Le script les détecte et les rastérise via `pypdfium2` :

1. Identification des pages avec figures mais sans image raster
2. Localisation de la légende « Figure SX. » par `pdfplumber`
3. Rendu de la zone au-dessus de la légende via `pypdfium2`
4. Application d'ELA et d'analyse du bruit sur l'image obtenue

Ces figures apparaissent avec un **badge violet ⃝v** dans le rapport HTML.

---

## Rapport HTML

Le rapport généré contient pour chaque image :

- Badge figure avec numéro (bleu = raster, violet = vectoriel)
- Numéro de page et dimensions
- Miniature originale
- Image ELA amplifiée
- Liste détaillée des résultats par technique avec score
- Score de suspicion global (0–100 %)

---

## Interprétation des résultats

> **Ces analyses sont des indices, pas des preuves.** Un score élevé justifie une vérification humaine par un expert en intégrité scientifique, et éventuellement une demande des images originales aux auteurs.

| Score global | Interprétation suggérée |
|---|---|
| 0–20 % | Aucune anomalie détectée |
| 20–50 % | Anomalies mineures, probablement dues au traitement normal des images |
| 50–80 % | Anomalies significatives, investigation recommandée |
| > 80 % | Anomalies fortes, demande d'images brutes aux auteurs justifiée |

Les faux positifs les plus courants sont les images très compressées (ELA élevé sans manipulation), les figures avec fond blanc dominant (copy-move), et les figures à deux panneaux symétriques (inter-panneaux).

---

## Exemple de sortie console

```
  Extraction des images de : article.pdf
  Détection des numéros de figure dans le texte…
    → 4/4 image(s) associée(s) à un numéro de figure
    Page   5 : Figure 1A, Figure 1B, Figure 1C, Figure 1E, Figure 1
    Page   6 : Figure 2
  Rendu de 1 page(s) vectorielle(s) via pypdfium2…
    → 1 page(s) rastérisée(s) ajoutées

  5 image(s) à analyser…

  [1/5] Page 5 [Figure 1 (panneaux A, B, C, E)] — 1980×1172px
  ℹ  [ELA] ELA normal (score=0.072)
  ℹ  [Bruit] Bruit homogène (CV=0.23)
  ℹ  [Copy-Move] Aucun clonage détecté (0 correspondances)
  🚨 [Inter-panneaux] col.A ≈ col.B (252 corr.) — images très probablement dupliquées
  🚨 [Inter-panneaux] col.D ≈ col.E (218 corr.) — images très probablement dupliquées
  🚨 [Inter-panneaux] … et 9 autres paires suspectes
```
