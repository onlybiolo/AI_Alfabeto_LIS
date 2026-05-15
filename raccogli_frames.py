import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time

lettere = [(chr(i + 65), i) for i in range(26)]

BURST = 250
dataset_path = "dati/dataset.npy"

# ---------------- MEDIAPIPE ----------------

config_base = python.BaseOptions(
    model_asset_path="mediapipe/hand_landmarker.task"
)

opzioni = vision.HandLandmarkerOptions(
    base_options=config_base,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

task_rilevatrice = vision.HandLandmarker.create_from_options(opzioni)

# pose che vuoi raccogliere (entrambe le mani)
pose = [
    ("MANO SX ALTO", "Palmo frontale"),
    ("MANO SX CENTRO", "Ruota leggermente a sinistra"),
    ("MANO SX BASSO", "Ruota leggermente a destra"),
    ("MANO SX VICINA", "Avvicina alla camera"),
    ("MANO DX ALTO", "Palmo frontale"),
    ("MANO DX CENTRO", "Ruota leggermente a destra"),
    ("MANO DX BASSO", "Ruota leggermente a sinistra"),
    ("MANO DX VICINA", "Avvicina alla camera"),
]


def raccogli_frames():

    # carico dataset esistente
    dataset_totale = []

    if os.path.exists(dataset_path):
        if os.path.getsize(dataset_path) > 0:
            dataset_totale = np.load(dataset_path).tolist()

    print(f"Frame già presenti: {len(dataset_totale)}")

    # scelta lettera
    lettera_input = input("Scegli lettera: ").upper()

    etichetta = next(
        (indice for char, indice in lettere if char == lettera_input),
        None
    )

    if etichetta is None:
        print("Lettera non valida")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Webcam non trovata")
        return

    dati_sessione = []
    frame_precedente = None 

    for posizione, rotazione in pose:

        pronto = False
        burst_attivo = False
        frame_registrati = 0

        while True:

            ret, frame = camera.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            risultato = task_rilevatrice.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=img_rgb
                )
            )

            frame = cv2.flip(frame, 1)

            # UI tua semplice
            cv2.putText(
                frame,
                f"LETTERA: {lettera_input}",
                (10, 35),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                posizione,
                (10, 80),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                rotazione,
                (10, 115),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if not burst_attivo:
                cv2.putText(
                    frame,
                    "Premi S quando sei pronto",
                    (10, 160),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    f"BURST: {frame_registrati}/{BURST}",
                    (10, 160),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # REGISTRAZIONE
            if burst_attivo and risultato.hand_landmarks:

                manina = risultato.hand_landmarks[0]

                polso_x = manina[0].x
                polso_y = manina[0].y
                polso_z = manina[0].z

                coordinate_frame = []

                for punto in manina:
                    coordinate_frame.append(
                        punto.x - polso_x
                    )
                    coordinate_frame.append(
                        punto.y - polso_y
                    )
                    coordinate_frame.append(
                        punto.z - polso_z
                    )
                
                #calcolo della velocita (diff con frame precedente)
                if frame_precedente is not None:
                    velocita = [
                        coordinate_frame[k] - frame_precedente[k]
                        for k in range(63)
                    ]
                else:
                    velocita = [0.0] * 63
                
                frame_precedente = coordinate_frame.copy()

                # salvo: 63 coord + 63 velocita + 1 etichetta = 127 valori
                riga_completa = coordinate_frame + velocita + [etichetta]
                dati_sessione.append(riga_completa)

                frame_registrati += 1

                if frame_registrati >= BURST:
                    break

            cv2.imshow("Dataset LIS", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                return

            elif key == ord("s"):
                burst_attivo = True
                print(f"BURST {BURST} iniziato")

    # salvataggio finale
    if len(dati_sessione) > 0:

        dataset_finale = dataset_totale + dati_sessione

        np.save(
            dataset_path,
            np.array(dataset_finale)
        )

        print(
            f"Salvato! "
            f"+{len(dati_sessione)} frame "
            f"(Totale: {len(dataset_finale)})"
        )

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    raccogli_frames()