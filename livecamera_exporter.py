# interfaccia che "prende" la cam e la fa diventare utilizzabile
# passandola per un server e interfacciandola su html

import livecamera as lc
import cv2

# Inizializziamo il riconoscitore una sola volta
recognizer = lc.HandRecognizer()
camera = cv2.VideoCapture(0)

def TrasportatoreFrame():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("ERRORE TRASPORTATORE: Frame non rilevato")
            break

        # Processiamo il frame con l'IA
        frame = recognizer.process_frame(frame)

        # lo codifichiamo come jpg
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            print("ERRORE TRASPORTATORE: Frame non codificato")
            continue

        frame_bytes = buffer.tobytes()
        
        # Formato MJPEG per il browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Mandiamo al browser anche la lettera predetta e la confidence e da quanto tempo e il timestamp di quando quella
        # lettera e stata predetta per la prima volta da quando e stata letta la camera, in modo tale che il server possa fare il grafico del tempo 
        # e vedere quando sto mimando per più tempo una determinata lettera.

def Get_Values():
    return recognizer.get_letter(), recognizer.get_confidenza(), recognizer.get_timestamp()