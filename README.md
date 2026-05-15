# 🤟 Riconoscitore Alfabeto LIS (Lingua Italiana dei Segni)

Progetto di riconoscimento dell'alfabeto LIS in tempo reale tramite webcam, costruito con Python, OpenCV, MediaPipe e una rete neurale MLP fatta a mano con NumPy.

**Autori:** Cristian Somma & Biolo Derek

---

## 📁 Struttura del progetto

```
Progetto_Informatica/
│
├── raccogli_frames.py   # Raccolta dati dal vivo con webcam, per popolare il dataset
├── train.py             # Training della rete neurale, traina la rete con i dati raccolti da raccogli_frames.py
|                        - TRAINARE SOLO UNA VOLTA REGISTRATE ALMENO 4 LETTERE NUOVE!
├── livecamera.py        # Riconoscimento in tempo reale
├── visualizza.py        # Visualizzatore grafico rete e training, demo da sistemare (probabile faremo l'html? col server? come finale) da rivedere
├── dataset_reader.py    # Ispezione e debug del dataset, penso che non servira alla fine era debug iniziale
├── jsonexporter.py      # Esporta il dataset in formato JSON, potrebbe servire nel caso facessimo la parte web per l'esportazione dei dati in un formato leggibile
│
├── dati/
│   └── dataset.npy          # Dataset raccolta dati (generato) 
│
├── modelli/
│   ├── modello.npz          # Pesi della rete neurale (generato)
│   └── storico.json         # Storico dei training (generato) - serve per visualizza.py
│
├── assets/
│   ├── ASL_Reference.png    # Immagine di riferimento alfabeto - referenze finali scelte
│   └── indice_dita.png      # Schema numerazione dita MediaPipe - era per capire come mediapipe indicizza le dita
│
└── mediapipe/
    └── hand_landmarker.task # Modello MediaPipe per landmark mano - Modello principale di MediaPipe
```

---

## ⚙️ Setup iniziale

> Ogni membro del team deve fare il setup sul proprio PC.

### 1. Clona il repository
```bash
git clone <url-del-repo>
cd Progetto_Informatica
```

### 2. Crea e attiva il virtual environment
```bash
# Crea il venv (lo fai solo una volta)
python -m venv venv

# Attivalo (Linux/Mac)
source venv/bin/activate

# Attivalo (Windows)
venv\Scripts\activate
```

### 3. Installa le dipendenze
```bash
pip install opencv-contrib-python mediapipe numpy matplotlib
```

### 4. Verifica di avere il file mediapipe
Assicurati che nella cartella `mediapipe/` ci sia il file `hand_landmarker.task`.  
Se non c'è, scaricalo da: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

---

## 🚀 Flusso di lavoro

```
1. raccogli_frames.py  →  dataset.npy
2. train.py            →  modello.npz + storico.json
3. livecamera.py       →  test in tempo reale
4. visualizza.py       →  analisi grafica dei training
```

---

## 📄 Descrizione degli script

### `raccogli_frames.py` — Raccolta dati
Apre la webcam e registra i movimenti della mano per ogni lettera dell'alfabeto.

**Come funziona:**
- Usa MediaPipe per rilevare i 21 punti landmark della mano
- Normalizza le coordinate sul polso (punto 0), così la posizione della mano non conta
- Calcola anche la **velocità** tra un frame e l'altro (differenza di posizione) per catturare il movimento di lettere dinamiche come J e Z
- Salva ogni frame come riga nel file `dataset.npy` nel formato: `[63 coordinate] + [63 velocità] + [1 etichetta]` = **127 valori**

**Utilizzo:**
```bash
python raccogli_frames.py
```
- Ti chiede quale lettera raccogliere (es. `A`)
- Per ogni lettera esegui **8 pose** (4 mano sinistra, 4 mano destra)
- Premi `S` per iniziare la registrazione di ogni posa (250 frame)
- Premi `Q` per uscire prima della fine

**Quanti dati raccogliere:**
- Ideale: ~2000 frame per lettera (automatico con le 8 pose × 250)
- Per lettere dinamiche (J, Z): **muovi la mano continuamente** durante il burst

---

### `train.py` — Training della rete neurale
Allena la rete neurale MLP (Multi-Layer Perceptron) sul dataset raccolto.

**Architettura:**
```
Input (126) → Hidden1 (256) → Hidden2 (128) → Hidden3 (64) → Output (26)
```
- Input: 63 coordinate + 63 velocità = 126 valori
- Output: 26 neuroni, uno per ogni lettera A-Z
- Attivazione: ReLU sugli hidden layer, Softmax sull'output
- Loss: Cross Entropy
- Ottimizzazione: Backpropagation manuale con mini-batch

**Utilizzo:**
```bash
python train.py
```

**Cosa produce:**
- `modello.npz` — salvato **solo se è il migliore** in accuracy sul test set
- `storico.json` — aggiornato con la curva loss/accuracy di ogni training

**Parametri modificabili nel file:**
| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `epoche` | 100 | Numero di epoche di training |
| `learing_rate` | 0.01 | Velocità di apprendimento |
| `batch` | 32 | Frame per mini-batch |

---

### `livecamera.py` — Riconoscimento in tempo reale
Apre la webcam e riconosce le lettere LIS in diretta usando il modello salvato.

**Come funziona:**
- Carica i pesi da `modello.npz`
- Per ogni frame, estrae i 21 landmark e calcola le coordinate + velocità (stesso preprocessing del training)
- Fa un forward pass della rete e mostra la lettera con la percentuale di confidenza
- Mostra `??` se la confidenza è sotto il 60% (mano non riconosciuta o gesto ambiguo)
- Resetta la velocità quando la mano esce dal campo

**Utilizzo:**
```bash
python livecamera.py
```
Premi `Q` per uscire.

---

### `visualizza.py` — Visualizzatore grafico
Mostra una finestra grafica con la struttura della rete neurale e lo storico dei training.

**Cosa mostra:**
- **Grafico superiore**: rete neurale con nodi e connessioni
  - Colore connessioni: 🔵 blu = pesi positivi, 🔴 rosso = pesi negativi
  - Spessore connessioni: proporzionale al peso medio
  - Colore nodi: basato sull'attivazione media sul dataset reale
- **Grafico inferiore sinistro**: curva della Loss durante il training
- **Grafico inferiore destro**: curva dell'Accuracy durante il training
- Badge **★ MIGLIORE** sul training con accuracy test più alta

**Utilizzo:**
```bash
python visualizza.py
```
- Usa le **frecce ← →** per navigare tra i training passati

---

### `dataset_reader.py` — Ispezione dataset
Utility per leggere e ispezionare il contenuto di `dataset.npy`.

**Utilizzo:**
```bash
python dataset_reader.py
```
Stampa: numero di frame, etichette presenti, coordinate e velocità del primo frame. Mostra anche un grafico 2D dei landmark della mano (frame 0).

---

### `jsonexporter.py` — Esportatore JSON
Converte `dataset.npy` in un file JSON (`dataset_export.json`).

**Utilizzo:**
```bash
python jsonexporter.py
```
Utile se volete analizzare i dati con altri strumenti (es. JavaScript, Excel).
> ⚠️ Il file JSON generato può essere molto grande (centinaia di MB). Non committatelo su Git.

---

## 🤝 Collaborazione (Workflow in due)

Per lavorare in parallelo senza problemi:

### Regola d'oro: Pull prima di tutto
```bash
git pull   # SEMPRE prima di raccogliere dati o fare training
```

### Raccolta dati parallela
Se raccogliete lettere diverse in parallelo (es. Cristian fa A-M, Derek fa N-Z), **non potete pushare il dataset contemporaneamente** altrimenti si crea un conflitto binario.

Strategia consigliata:
1. Uno raccoglie i dati e fa `push`
2. L'altro fa `pull`, raccoglie i suoi dati, fa `push`

### Training
Il training sovrascrive `modello.npz` solo se migliora l'accuracy. Potete fare `push` del modello migliore e l'altro farà `pull` per averlo.

---

## 🔤 Guida alle lettere dinamiche (J e Z)

Queste lettere richiedono movimento, non sono pose statiche:

| Lettera | Movimento |
|---------|-----------|
| **J** | Mignolo teso, traccia una J nell'aria (curva verso il basso) |
| **Z** | Indice teso, traccia una Z nell'aria (zigzag orizzontale) |

**Durante la raccolta dati:** ripeti il gesto continuamente per tutta la durata del burst (250 frame). Non fermarti a metà.

---

## 📊 Formato del dataset

Ogni riga di `dataset.npy` contiene **127 valori**:

| Colonne | Contenuto |
|---------|-----------|
| 0 – 62 | Coordinate XYZ dei 21 landmark, normalizzate sul polso |
| 63 – 125 | Velocità (differenza coordinate rispetto al frame precedente) |
| 126 | Etichetta: intero da 0 (A) a 25 (Z) |
