# Importation des modules nécessaires
from flask import Flask, render_template, Response, jsonify, request  # Flask pour le serveur web
import cv2 as cv             # OpenCV pour la capture et traitement vidéo
import imutils               # Utilitaire pour gérer les contours facilement
import pytesseract           # OCR pour reconnaître le texte (plaques)
import re                    # Expressions régulières pour filtrer le format des plaques
import time                  # Gestion du temps pour éviter les doublons
import os                    # Pour arrêter le serveur

# ---------- INITIALISATION DE FLASK ----------
app = Flask(__name__)  # Création de l'application Flask

# ---------- CONFIGURATION DE LA RECONNAISSANCE DES PLAQUES ----------
regex_plaque = re.compile(r"[A-Z]{2}-?[0-9]{3}-?[A-Z]{2}")  # Format plaque FR (ex: AB-123-CD)
dernier_texte = ""    # Stocke la dernière plaque détectée pour éviter les doublons
dernier_temps = 0     # Stocke le temps de détection pour la temporisation
delai_capture = 3     # Délai minimum en secondes avant de réenregistrer une même plaque

# Initialisation de la capture vidéo (webcam par défaut)
video = cv.VideoCapture(0)  # Remplacer par 0 pour la webcam

# Liste globale pour stocker les plaques détectées (sera renvoyée au front-end)
plaques_detectees = []

# ---------- FONCTION QUI GENERE LE FLUX VIDEO ----------
def gen_frames():
    """
    Fonction qui lit la caméra en continu, détecte les plaques,
    applique l'OCR et renvoie les images encodées pour le flux vidéo
    utilisable dans la page web.
    """
    global dernier_texte, dernier_temps, plaques_detectees  # On utilise les variables globales
    
    while True:
        # Lecture d'une frame depuis la caméra
        success, frame = video.read()
        if not success:
            break  # Si la caméra ne répond pas, on sort de la boucle

        # Redimensionnement pour un affichage cohérent
        img = cv.resize(frame, (640, 480))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Conversion en niveaux de gris pour traitement

        # ---------- PRÉTRAITEMENT DE L'IMAGE ----------
        blur = cv.GaussianBlur(gray, (5, 5), 0)    # Flou pour réduire le bruit
        sobelx = cv.Sobel(blur, cv.CV_8U, 1, 0, ksize=3)  # Détection des contours horizontaux
        _, thresh = cv.threshold(sobelx, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Seuillage

        # Morphologie pour fermer les zones et améliorer la détection
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (17, 5))
        morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        # Détection des contours dans l'image
        contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        plaque_trouvee = False  # Flag pour savoir si une plaque a été détectée

        # ---------- ANALYSE DES CONTOURS ----------
        for c in contours:
            x, y, w, h = cv.boundingRect(c)  # Rectangle englobant du contour
            ratio = w / float(h)             # Ratio largeur/hauteur

            # Filtrage basé sur le ratio et la taille pour éviter les faux positifs
            if ratio < 1.2 or ratio > 6.5: continue
            if w < 80 or h < 20: continue
            if w > 500: continue

            plaque_trouvee = True
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Dessine un rectangle rouge autour

            # Extraction de la zone candidate (possible plaque)
            Cropped = gray[y:y + h, x:x + w]

            # OCR avec Tesseract pour lire le texte sur la plaque
            config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            texte = pytesseract.image_to_string(Cropped, config=config)
            texte = texte.strip().replace(" ", "").upper()  # Nettoyage du texte

            # Ignorer si le texte est trop court ou ne correspond pas au format FR
            if len(texte) < 5: continue
            if not regex_plaque.search(texte): continue

            # ---------- ANTI-DOUBLONS ----------
            maintenant = time.time()
            if texte != dernier_texte or (maintenant - dernier_temps) > delai_capture:
                dernier_texte = texte
                dernier_temps = maintenant
                # Ajoute à la liste globale si ce n'est pas déjà présent
                if texte not in plaques_detectees:
                    plaques_detectees.append(texte)
                    # Limiter la liste à 5 dernières plaques
                    if len(plaques_detectees) > 5:
                        plaques_detectees.pop(0)

            # Affichage du texte détecté sur l'image
            cv.putText(img, texte, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # On ne garde que la première plaque valide par frame

        # Si aucune plaque détectée, message sur l'image
        if not plaque_trouvee:
            cv.putText(img, "Aucune plaque detectee", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Encodage de la frame pour le flux vidéo
        ret, buffer = cv.imencode('.jpg', img)
        frame = buffer.tobytes()
        # Renvoi de la frame en format compatible streaming MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------- ROUTES FLASK ----------

@app.route('/')
def index():
    """
    Page principale : renvoie le template HTML
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Route qui fournit le flux vidéo pour le <img> HTML
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_plaques')
def get_plaques():
    """
    Route qui renvoie les plaques détectées au format JSON
    Pour l'affichage dynamique sur la page web
    """
    return jsonify(plaques_detectees)

@app.route('/stop_server', methods=['POST'])
def stop_server():
    """
    Arrête le serveur Flask si le texte reçu est 'stop'.
    """
    data = request.get_json()  # Récupère les données JSON envoyées
    if data and data.get("action") == "stop":
        print("Arrêt du serveur demandé...")
        os._exit(0)  # Termine le serveur immédiatement
        return "Arrêt du serveur", 200
    return "Action non reconnue", 400

# ---------- LANCEMENT DE L'APPLICATION ----------
if __name__ == "__main__":
    app.run(debug=True)  # Démarre le serveur Flask en mode debug
    video.release()  # Libère la caméra à la fin
