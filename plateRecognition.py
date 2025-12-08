import sys

import cv2 as cv
import numpy as np
import imutils
import pytesseract

## charger une video
video = cv.VideoCapture(0)

# Vérifier que la vidéo est bien chargée
if not video.isOpened():
    print("Erreur : impossible de charger la vidéo.")


# 4. Boucle pour lire toutes les images en NdG
while True:
    ret, frame = video.read()
    img_origin = cv.resize(frame, (600,400) )

    if not ret:
        break  # fin de vidéo

    img_NdG = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY)

    # 3. Détection des bords (Canny)
    edged = cv.Canny(img_NdG, 30, 200)

    contours=cv.findContours(edged.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    
    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv.drawContours(img_origin, [screenCnt], -1, (0, 0, 255), 3)

        mask = np.zeros(img_NdG.shape,np.uint8)
        new_image = cv.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv.bitwise_and(img_origin,img_origin,mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = img_NdG[topx:bottomx+1, topy:bottomy+1]

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        print("programming_fever's License Plate Recognition\n")
        print("Detected license plate Number is:",text)
        img = cv.resize(img_origin,(500,300))
        Cropped = cv.resize(Cropped,(400,200))
        cv.imshow('car',img)
        cv.imshow('Cropped',Cropped)

    # 4. Affichage
    #cv.imshow("Flux original", img_origin)
    #cv.imshow("Niveaux de gris", img_NdG)
    #cv.imshow("Bords (Canny)", edged)

    # 5. Quitter avec 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
cv.destroyAllWindows()