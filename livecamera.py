# programma effettivo che prevede la lettera che sto facendo con la mano
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

LETTERE = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# carica modello salvato
modello = np.load('modelli/modello.npz')
pesi = [modello[f'pesi_{i}'] for i in range(4)]
bias = [modello[f'bias_{i}'] for i in range(4)]
print("Modello caricato con successo")

# stesse funzioni del training
# servono perche la rete ricorda nel file solo i suoi pesi, non sa fare il forward pass da sola del frame che vede in diretta
# percio per prevedere deve fare il calcolo basandosi pero sulla sua intelligenza (pesi)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    esponente = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return esponente / np.sum(esponente, axis=1, keepdims=True)

def forward_pass(X):
    attivazioni = [X]
    a = X
    for i in range(len(pesi) -1):
        z = np.dot(a, pesi[i]) + bias[i]
        a = relu(z)
        attivazioni.append(a)

    z = np.dot(a, pesi[-1]) + bias[-1]
    a = softmax(z)
    attivazioni.append(a)

    return attivazioni

# config mediapipe
configurazione_base = python.BaseOptions(model_asset_path="mediapipe/hand_landmarker.task")
opzioni = vision.HandLandmarkerOptions(
    base_options=configurazione_base, 
    num_hands=1, 
    running_mode=vision.RunningMode.IMAGE
)
task_rilevatrice = vision.HandLandmarker.create_from_options(opzioni)

# Videocamera
camera = cv2.VideoCapture(0)
frame_precedente = None

while True:
    ret, frame = camera.read()
    if not ret:
        print("Impossibile leggere il frame")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    risultato = task_rilevatrice.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))

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
        if frame_precedente is not None:
            velocita = [coordinate[k] - frame_precedente[k] for k in range(63)]
        else:
            velocita = [0.0] * 63
        
        frame_precedente = coordinate.copy()

        
        # predizione

        X = np.array([coordinate + velocita])
        attivazione = forward_pass(X)
        probabilita = attivazione[-1][0]
        index_lettera = np.argmax(probabilita)
        confidenza = probabilita[index_lettera] * 100

        # se la probabilita media e troppo bassa significa che l'utente non sta facendo nessun gesto oppure
        # la mano e solo passata nel frame, per evitare predizioni casuali, non stampo nulla
        if confidenza < 60:
            lettera = "??" 
        else:
            lettera = LETTERE[index_lettera] 

        cv2.putText(frame, f"{lettera}: {confidenza:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        cv2.putText(frame, "Nessuna mano rilevata", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        # reset del frame precedente quando non c'è nessuna mano
        frame_precedente = None
    
    cv2.imshow("ASL RILEVATORE by Cristian Somma & Biolo Derek", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


        