import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os

# ======================== CONFIGURAZIONE ========================

LAYERS = [126, 256, 128, 64, 26]
LAYER_NAMES = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
MAX_NODI = 12  # massimo nodi visibili per layer

STORICO_PATH = "modelli/storico.json"
MODELLO_PATH = "modelli/modello.npz"
DATASET_PATH = "dati/dataset.npy"

# ======================== FUNZIONI UTILITA ========================

def carica_storico():
    if not os.path.exists(STORICO_PATH):
        return []
    with open(STORICO_PATH, 'r') as f:
        return json.load(f)

def carica_modello():
    if not os.path.exists(MODELLO_PATH):
        return None, None
    m = np.load(MODELLO_PATH)
    n_layers = 0
    while f'pesi_{n_layers}' in m:
        n_layers += 1
    pesi = [m[f'pesi_{i}'] for i in range(n_layers)]
    bias = [m[f'bias_{i}'] for i in range(n_layers)]
    return pesi, bias

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def calcola_attivazioni_medie(pesi, bias):
    """Fa un forward pass su un campione del dataset e restituisce l'attivazione media per layer."""
    if not os.path.exists(DATASET_PATH):
        return None

    dataset = np.load(DATASET_PATH)
    # controlla se il dataset ha il formato nuovo (127 colonne) o vecchio (64)
    if dataset.shape[1] >= 127:
        X = dataset[:, :126].astype(float)
    else:
        X = dataset[:, :63].astype(float)

    # campione di max 200 frame
    if len(X) > 200:
        idx = np.random.choice(len(X), 200, replace=False)
        X = X[idx]

    # se la dimensione input non corrisponde ai pesi, non posso fare forward
    if X.shape[1] != pesi[0].shape[0]:
        return None

    attivazioni_medie = [np.mean(np.abs(X), axis=0)]  # media input

    a = X
    for i in range(len(pesi) - 1):
        z = np.dot(a, pesi[i]) + bias[i]
        a = relu(z)
        attivazioni_medie.append(np.mean(a, axis=0))  # media per neurone

    z = np.dot(a, pesi[-1]) + bias[-1]
    a = softmax(z)
    attivazioni_medie.append(np.mean(a, axis=0))

    return attivazioni_medie

# ======================== VISUALIZZATORE ========================

class Visualizzatore:
    def __init__(self):
        self.storico = carica_storico()
        self.indice = len(self.storico) - 1 if self.storico else -1

        # trova il migliore
        if self.storico:
            self.migliore_idx = max(
                range(len(self.storico)),
                key=lambda i: self.storico[i]['accuracy_test']
            )
        else:
            self.migliore_idx = -1

        # carica modello e attivazioni
        self.pesi_modello, self.bias_modello = carica_modello()
        self.attivazioni = None
        if self.pesi_modello is not None:
            self.attivazioni = calcola_attivazioni_medie(self.pesi_modello, self.bias_modello)

        # finestra matplotlib
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title("Visualizzatore Rete Neurale LIS")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.disegna()
        plt.show()

    def on_key(self, event):
        if not self.storico:
            return
        if event.key == 'left':
            self.indice = max(0, self.indice - 1)
        elif event.key == 'right':
            self.indice = min(len(self.storico) - 1, self.indice + 1)
        else:
            return
        self.disegna()

    def disegna(self):
        self.fig.clear()

        if not self.storico:
            self.fig.text(
                0.5, 0.5,
                "Nessun training ancora\n\nLancia train.py per iniziare",
                ha='center', va='center', fontsize=22,
                color='#aaaaaa', fontstyle='italic'
            )
            self.fig.canvas.draw()
            return

        run = self.storico[self.indice]

        # titolo
        titolo = f"Training #{self.indice + 1}/{len(self.storico)}  —  {run['data']}"
        acc_test = run['accuracy_test']
        badge = ""
        if self.indice == self.migliore_idx:
            badge = "  ★ MIGLIORE"

        self.fig.suptitle(
            f"{titolo}\nAccuracy Test: {acc_test:.2f}%{badge}",
            fontsize=15, fontweight='bold',
            color='#ffffff' if not badge else '#FFD700',
            y=0.97
        )

        # assi
        ax_rete = self.fig.add_axes([0.05, 0.42, 0.9, 0.50])
        ax_loss = self.fig.add_axes([0.08, 0.07, 0.38, 0.28])
        ax_acc = self.fig.add_axes([0.56, 0.07, 0.38, 0.28])

        self.disegna_rete(ax_rete, run)
        self.disegna_loss(ax_loss, run)
        self.disegna_accuracy(ax_acc, run)

        # hint navigazione
        self.fig.text(
            0.5, 0.01,
            "◀ ▶  Frecce per navigare i training",
            ha='center', fontsize=10, color='#666666'
        )

        self.fig.canvas.draw()

    # -------------------- RETE NEURALE --------------------

    def disegna_rete(self, ax, run):
        ax.set_xlim(-0.8, len(LAYERS) - 0.2)
        ax.set_ylim(-1.5, MAX_NODI + 0.5)
        ax.axis('off')
        ax.set_title("Architettura Rete Neurale", fontsize=13, color='#cccccc', pad=10)

        pesi_abs = run.get('pesi_medi_abs', [0.05] * (len(LAYERS) - 1))
        pesi_segno = run.get('pesi_medi_segno', [0.01] * (len(LAYERS) - 1))

        # calcolo posizioni nodi
        posizioni = []
        for l, n in enumerate(LAYERS):
            n_vis = min(n, MAX_NODI)
            ys = np.linspace(0.5, MAX_NODI - 0.5, n_vis)
            posizioni.append([(l, y) for y in ys])

        # connessioni tra layer
        for l in range(len(LAYERS) - 1):
            w_abs = pesi_abs[l] if l < len(pesi_abs) else 0.05
            w_segno = pesi_segno[l] if l < len(pesi_segno) else 0.0
            lw = max(0.2, min(2.5, w_abs * 25))
            colore = '#4FC3F7' if w_segno >= 0 else '#EF5350'
            alpha = max(0.03, min(0.25, w_abs * 3))

            for (x1, y1) in posizioni[l]:
                for (x2, y2) in posizioni[l + 1]:
                    ax.plot([x1, x2], [y1, y2],
                            color=colore, alpha=alpha, linewidth=lw)

        # colori nodi basati su attivazione
        layer_colors = ['#4FC3F7', '#81C784', '#81C784', '#81C784', '#FFB74D']

        for l, layer_pos in enumerate(posizioni):
            n_reali = LAYERS[l]
            n_vis = len(layer_pos)

            for j, (x, y) in enumerate(layer_pos):
                # colore basato sull'attivazione media se disponibile
                if self.attivazioni is not None and l < len(self.attivazioni):
                    att = self.attivazioni[l]
                    # campiono gli indici corrispondenti
                    idx_reale = int(j * (len(att) - 1) / max(1, n_vis - 1)) if n_vis > 1 else 0
                    val = float(att[idx_reale])
                    # normalizzo: 0 = grigio scuro, alto = giallo/arancione
                    max_att = float(np.max(att)) if np.max(att) > 0 else 1.0
                    intensita = min(1.0, val / max_att)
                    r = 0.3 + 0.7 * intensita
                    g = 0.3 + 0.5 * intensita
                    b_c = 0.3 * (1 - intensita)
                    colore_nodo = (r, g, b_c)
                else:
                    colore_nodo = layer_colors[l]

                cerchio = plt.Circle(
                    (x, y), 0.18,
                    color=colore_nodo, ec='#555555',
                    linewidth=0.8, zorder=5
                )
                ax.add_patch(cerchio)

            # indicatore "..." se troncato
            if n_reali > MAX_NODI:
                ax.text(l, -0.2, "⋮", ha='center', fontsize=16, color='#888888')

            # etichetta layer
            ax.text(
                l, -0.9,
                f"{LAYER_NAMES[l]}\n({n_reali})",
                ha='center', fontsize=10, fontweight='bold', color='#cccccc'
            )

    # -------------------- GRAFICI --------------------

    def disegna_loss(self, ax, run):
        loss = run.get('loss_per_epoca', [])
        if not loss:
            ax.text(0.5, 0.5, "Nessun dato", ha='center', va='center',
                    transform=ax.transAxes, color='#888888')
            return

        epoche = list(range(len(loss)))
        ax.plot(epoche, loss, color='#EF5350', linewidth=2, alpha=0.9)
        ax.fill_between(epoche, loss, alpha=0.15, color='#EF5350')
        ax.set_title("Loss per Epoca", fontsize=11, color='#cccccc')
        ax.set_xlabel("Epoca", fontsize=9, color='#999999')
        ax.set_ylabel("Cross Entropy", fontsize=9, color='#999999')
        ax.grid(True, alpha=0.15, color='#555555')
        ax.set_ylim(bottom=0)
        ax.tick_params(colors='#999999', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#444444')

    def disegna_accuracy(self, ax, run):
        acc = run.get('accuracy_per_epoca', [])
        if not acc:
            ax.text(0.5, 0.5, "Nessun dato", ha='center', va='center',
                    transform=ax.transAxes, color='#888888')
            return

        epoche = list(range(len(acc)))
        ax.plot(epoche, acc, color='#66BB6A', linewidth=2, alpha=0.9)
        ax.fill_between(epoche, acc, alpha=0.15, color='#66BB6A')
        ax.set_title("Accuracy per Epoca", fontsize=11, color='#cccccc')
        ax.set_xlabel("Epoca", fontsize=9, color='#999999')
        ax.set_ylabel("Accuracy %", fontsize=9, color='#999999')
        ax.grid(True, alpha=0.15, color='#555555')
        ax.set_ylim(0, 105)
        ax.tick_params(colors='#999999', labelsize=8)

        # linea tratteggiata per accuracy test
        acc_test = run.get('accuracy_test', 0)
        ax.axhline(y=acc_test, color='#FFD700', linestyle='--',
                   linewidth=1, alpha=0.7, label=f'Test: {acc_test:.1f}%')
        ax.legend(fontsize=8, loc='lower right',
                  facecolor='#1e1e1e', edgecolor='#444444', labelcolor='#cccccc')

        for spine in ax.spines.values():
            spine.set_color('#444444')


# ======================== MAIN ========================

if __name__ == "__main__":
    Visualizzatore()
