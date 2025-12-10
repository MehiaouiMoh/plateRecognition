import cv2 as cv
import numpy as np
import imutils
import pytesseract
import re
import time

# ---------- REGEX PLAQUE FR ----------
regex_plaque = re.compile(r"[A-Z]{2}-?[0-9]{3}-?[A-Z]{2}")

# Anti-duplicate
dernier_texte = ""
dernier_temps = 0
delai_capture = 3  # secondes

video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    img = cv.resize(frame, (640, 480))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ----------- Prétraitement amélioré -----------
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv.Sobel(blur, cv.CV_8U, 1, 0, ksize=3)
    _, thresh = cv.threshold(sobelx, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (17, 5))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    plaque_trouvee = False

    for c in contours:
        x, y, w, h = cv.boundingRect(c)

        # ----------- Filtres plaque assouplis -----------
        ratio = w / float(h)
        if ratio < 1.2 or ratio > 6.5:
            continue

        if w < 80 or h < 20:        # plus permissif
            continue

        if w > 500:                 # éviter détecter tout le téléphone
            continue

        # Plaque candidate détectée → extraction
        plaque_trouvee = True
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        Cropped = gray[y:y + h, x:x + w]

        # ----------- OCR amélioré -----------
        config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        texte = pytesseract.image_to_string(Cropped, config=config)

        texte = texte.strip().replace(" ", "").upper()

        # Ignorer si très court
        if len(texte) < 5:
            continue

        # ----------- Vérification regex FR -----------
        if not regex_plaque.search(texte):
            continue

        # ----------- Anti-duplicate -----------

        maintenant = time.time()
        if texte != dernier_texte or (maintenant - dernier_temps) > delai_capture:
            print("Plaque détectée :", texte)
            dernier_texte = texte
            dernier_temps = maintenant

        cv.putText(img, texte, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2)

        # Affichage plaque extraite
        cv.imshow("Plaque", cv.resize(Cropped, (350, 120)))

        break  # on prend la 1ère plaque valable

    if not plaque_trouvee:
        cv.putText(img, "Aucune plaque detectee", (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow("Detection", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()