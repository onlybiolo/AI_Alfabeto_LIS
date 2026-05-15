import numpy as np
import os
import matplotlib.pyplot as plt

def leggi_dataset():
    dataset = np.load('dati/dataset.npy')
    print(f"Shape dataset: {dataset.shape}")  # utile per capire quante righe/colonne

    # colonna 63 = etichetta, colonne 0:63 = coordinate
    label = dataset[:, 126]
    coord = dataset[:, :63]
    vel = dataset[:, 63:126]

    print(f"Etichette uniche presenti: {np.unique(label)}")
    print(f"Totale frame: {len(label)}")
    print(f"Prime 3 etichette: {label[:3]}")
    print(f"Prime coords (frame 0): {coord[0]}")
    print(f"Prime velocita (frame 0): {vel[0]}")

    rappresentazione_grafica_mano(coord[0])

def rappresentazione_grafica_mano(mano):
    x = mano[0::3]
    y = mano[1::3]

    plt.scatter(x, y)
    plt.gca().invert_yaxis()  # MediaPipe ha y invertita (0 = top)
    plt.title("Landmark mano (frame 0)")
    plt.xlabel("X normalizzata")
    plt.ylabel("Y normalizzata")
    plt.show()

if __name__ == "__main__":
    leggi_dataset()