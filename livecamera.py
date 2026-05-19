# programma effettivo che prevede la lettera che sto facendo con la mano
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

LETTERE = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

class HandRecognizer:
    def __init__(self, model_path='modelli/modello.npz', task_path="mediapipe/hand_landmarker.task"):
        # carica modello salvato
        modello = np.load(model_path)
        self.pesi = [modello[f'pesi_{i}'] for i in range(4)]
        self.bias = [modello[f'bias_{i}'] for i in range(4)]
        print("Modello caricato con successo")

        # config mediapipe
        configurazione_base = python.BaseOptions(model_asset_path=task_path)
        opzioni = vision.HandLandmarkerOptions(
            base_options=configurazione_base, 
            num_hands=1, 
            running_mode=vision.RunningMode.IMAGE
        )
        self.task_rilevatrice = vision.HandLandmarker.create_from_options(opzioni)
        self.frame_precedente = None

        # stato corrente (per il server)
        self.lettera = ""
        self.confidenza = 0.0
        self.timestamp = 0.0  # momento in cui la lettera corrente e stata vista per la prima volta

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        esponente = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return esponente / np.sum(esponente, axis=1, keepdims=True)

    def forward_pass(self, X):
        a = X
        for i in range(len(self.pesi) - 1):
            z = np.dot(a, self.pesi[i]) + self.bias[i]
            a = self.relu(z)

        z = np.dot(a, self.pesi[-1]) + self.bias[-1]
        a = self.softmax(z)
        return a

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        risultato = self.task_rilevatrice.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))

        frame = cv2.flip(frame, 1) # riflette l'immagine, cosi la prospettiva e giusta 
        
        if risultato.hand_landmarks:
            manina = risultato.hand_landmarks[0]
            
            polso_x = manina[0].x
            polso_y = manina[0].y
            polso_z = manina[0].z

            coordinate = []

            for punto in manina:
                coordinate.append(punto.x - polso_x)
                coordinate.append(punto.y - polso_y)
                coordinate.append(punto.z - polso_z)

            # calcolo velocita
            if self.frame_precedente is not None:
                velocita = [coordinate[k] - self.frame_precedente[k] for k in range(63)]
            else:
                velocita = [0.0] * 63
            
            self.frame_precedente = coordinate.copy()

            # predizione
            X = np.array([coordinate + velocita])
            probabilita = self.forward_pass(X)[0]
            index_lettera = np.argmax(probabilita)
            confidenza = probabilita[index_lettera] * 100

            if confidenza < 60:
                lettera = "??"
            else:
                lettera = LETTERE[index_lettera]

            # aggiorna lo stato solo se la lettera cambia (reset del timestamp)
            if lettera != self.lettera:
                self.timestamp = time.time()
            self.lettera = lettera
            self.confidenza = float(confidenza)

            cv2.putText(frame, f"{lettera}: {confidenza:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Nessuna mano rilevata", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            self.frame_precedente = None
            self.lettera = ""
            self.confidenza = 0.0
            self.timestamp = 0.0
        
        return frame

    def get_letter(self):
        return self.lettera

    def get_confidenza(self):
        return self.confidenza

    def get_timestamp(self):
        return self.timestamp

if __name__ == "__main__":
    recognizer = HandRecognizer()
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame = recognizer.process_frame(frame)
        cv2.imshow("ASL RILEVATORE", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()



        