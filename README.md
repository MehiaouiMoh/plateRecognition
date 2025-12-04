# plateRecognition

Projet de détection de plaques d'immatriculation automobiles utilisant OpenCV.
Par : MEHIAOUI Mohamed et MERIMI Ayat

## Objectif

Ce projet vise à détecter et reconnaître les plaques d'immatriculation sur des images ou des flux vidéo en utilisant OpenCV pour la localisation et Tesseract pour l'OCR. Ce README décrit les étapes qu'on va suivre pour reproduire et développer le projet.

## Pré-requis

- `Python 3.8+`
- `pip` (gestionnaire de paquets)
- OpenCV (`opencv-python`), NumPy, et autres dépendances Python
-  (moteur externe) pour la reconnaissance de caractères

## Installation (Windows)

1. Cloner le dépôt:

```powershell
git clone <repo_url>
cd "plateRecognition"
```

2. Créer et activer un environnement virtuel:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Installer les dépendances Python (exemples):

```powershell
pip install --upgrade pip
pip install opencv-python numpy imutils pytesseract matplotlib
```

4. Installer Tesseract OCR (moteur natif):

- Télécharger l'installateur pour Windows: https://github.com/tesseract-ocr/tesseract
- Installer et ajouter le dossier d'installation (par ex. `C:\Program Files\Tesseract-OCR`) au `PATH` système.
- Vérifier l'installation:

```powershell
tesseract --version
```

5. (Optionnel) Installer des dépendances pour entraînement de modèles (PyTorch, YOLOv5/YOLOv8) si vous prévoyez d'utiliser des modèles deep learning.

## Structure suggérée du projet

- `data/` : dossiers `images/`, `annotations/` (fichiers XML/JSON)
- `src/` : scripts Python (`preprocess.py`, `detect_plate.py`, `recognize.py`, `train.py`)
- `models/` : poids des modèles (cascade, YOLO, etc.)
- `notebooks/` : notebooks d'analyse
- `requirements.txt` : dépendances du projet

## Étapes de développement

1. Collecte et annotation des données
	- Utiliser des datasets publics (ex : OpenALPR, UFPR-ALPR, CCPD) ou collecter vos propres images.
	- Annoter les boîtes englobantes de plaques avec `LabelImg` ou un outil similaire (format Pascal VOC ou COCO).

2. Prétraitement
	- Redimensionner et normaliser les images.
	- Améliorer le contraste (CLAHE), filtrage bilatéral, conversion en niveaux de gris.

3. Localisation de la plaque
	- Approche classique: filtrage, détection d'arêtes (Canny), recherche de contours et filtrage par ratio/aire.
	- Approche ML/DL: utiliser un détecteur (cascade Haar, SSD, YOLO). Entraîner sur vos annotations pour de meilleures performances.

4. Segmentation des caractères (optionnelle)
	- Après extraction de la plaque, segmenter les caractères par seuillage, opérations morphologiques et détection de contours.

5. Reconnaissance (OCR)
	- Utiliser `pytesseract` pour lire la zone de la plaque.
	- Appliquer un post-traitement: suppression des caractères non alphanumériques, corrections basées sur des expressions régulières (format des plaques locales), dictionnaires de correction.

6. Évaluation
	- Pour la localisation: utiliser mAP / IoU et métriques rappel/précision.
	- Pour la reconnaissance: utiliser taux de caractères corrects et taux de plaques correctement lues.
# plateRecognition — Checklist

Projet: détection et reconnaissance de plaques d'immatriculation (OpenCV + Tesseract).

## Checklist de projet (à cocher)

- [x] Rédiger un `README.md` initial décrivant l'objectif et le pipeline
- [ ] Préparer l'environnement de développement
	- `Python 3.8+`, `pip`, créer un environnement virtuel
- [ ] Installer les dépendances Python
	- Exemple: `opencv-python`, `numpy`, `imutils`, `pytesseract`, `matplotlib`
- [ ] Installer Tesseract OCR (moteur natif)
	- Ajouter `C:\Program Files\Tesseract-OCR` au `PATH` sur Windows
- [ ] Structurer le projet
	- `data/` (images, annotations), `src/` (scripts), `models/`, `notebooks/`
- [ ] Collecter et annoter des données
	- Utiliser `LabelImg` ou formats Pascal VOC/COCO; datasets publics: UFPR-ALPR, CCPD
- [ ] Implémenter le prétraitement des images
	- Redimensionnement, conversion en gris, CLAHE, filtrage
- [ ] Implémenter la localisation des plaques
	- Approche classique: Canny + contours + filtres géométriques
	- Optionnel: entraîner un détecteur (YOLO, SSD)
- [ ] Segmenter les caractères de la plaque (si nécessaire)
	- Seuillage, morphologie, extraction de contours
- [ ] Intégrer l'OCR (Tesseract) et post-traitement
	- Nettoyage, regex pour format local, correction heuristique
- [ ] Évaluer les performances
	- Localisation: IoU / mAP; Reconnaissance: taux de caractères/plaque
- [ ] Fournir des scripts d'exemple et instructions d'exécution
	- `src/detect_plate.py`, `src/recognize.py`, exemples de commandes
- [ ] Ajouter des tests et métriques d'évaluation
- [ ] Proposer améliorations et extensions
	- Améliorer dataset, utiliser YOLOv5/YOLOv8, interface web (Flask/Streamlit)

## Commandes rapides (exemples Windows PowerShell)

```powershell
git clone <repo_url>
cd "plateRecognition"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install opencv-python numpy imutils pytesseract matplotlib
```

Vérifier Tesseract:

```powershell
tesseract --version
```

## Exemples d'utilisation

- Détection d'une image:

```powershell
python src/detect_plate.py --image data/images/test1.jpg --output results/
```

- Reconnaissance sur une plaque extraite:

```powershell
python src/recognize.py --plate results/plate_1.png
```

## Ressources

- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- UFPR-ALPR dataset: https://web.inf.ufpr.br/vri/databases/ufpr-alpr/
- CCPD: https://github.com/detectRecog/CCPD
- YOLOv5: https://github.com/ultralytics/yolov5

---
