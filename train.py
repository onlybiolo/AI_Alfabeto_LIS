import numpy as np
import datetime
import json
import os

# carico il dataset
dataset = np.load('dati/dataset.npy')

# separazione etichette e coordinate
X = dataset[:, :126].astype(float) # 63 coords + 63 velocita
y = dataset[:, 126].astype(int)     # etichette intere (0-25)

# per categorizzare, ond'evitare che l'ai pensa che numeri grandi siano simili o piccoli diversi
# uso l'one hot encoding, mi garantisce di trasformare i vari numeri in array con un bit "caldo"
# cioe se avessi [1,2,3] diventerebbe [1,0,0] [0,1,0] [0,0,1] cosi e capace di 'categorizzare' le cose
# in modo efficiente 
def one_hot(y, n_classi=26):
    out = np.zeros((len(y), n_classi))
    out[np.arange(len(y)), y] = 1
    return out # out sarebbe l'etichetta cambiata in un array di zeri e uno 

Y = one_hot(y)  # (n_frame, 26)

# creo un random seed col tempo
np.random.seed(int(datetime.datetime.now().timestamp()))
indici = np.random.permutation(len(X))
split = int(len(indici) * 0.8) # divido il dataset in 2 (80% train, 20% test)

# sessione di train e di test
X_training, X_testing = X[indici[:split]], X[indici[split:]]  
Y_training, Y_testing = Y[indici[:split]], Y[indici[split:]]

print(f"Dimensioni del dataset: {dataset.shape}")
print(f"Dimensione training set: {X_training.shape}")
print(f"Dimensione test set: {X_testing.shape}")

layers = [126, 256, 128, 64, 26]
# 63 -> coordinate per frame (fisso)
# 256, 128, 64 -> strati nascosti (possono cambiare, per i 26.000 circa vanno bene questi, ma se dovessi testare posso abbassare a 128, 64, 32)
# 26 -> output (numero di lettere) (fisso)

pesi = []
bias = []

for i in range(len(layers) - 1):
    w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]) # sqrt(2 / n_neuroni_layer_precedente)
    b = np.zeros((1, layers[i+1]))
    pesi.append(w)
    bias.append(b)

# ============================================ FUNZIONI DI ATTIVAZIONE ============================================


# ReLU (Rectified Linear Unit), e abbastanza usata, fa in modo che i numeri negativi diventino zero, questo 
# evita problemi di gradienti che non funzionano bene, infatti le reti neurali moderne lo usano spesso
def relu(x):
    return np.maximum(0, x)

# trasforma i neuroni accesi in 1, e quelli spenti in 0, nella backpropagation, ovvero quando la rete 'impara'
# serve a capire con quali neuroni aggiornare i pesi, se io ho commesso un errore ma non stavo usando la mano destra
# non ha senso incolpare i movimenti o stato della destra, infatti la faccio moltiplicare per 0 per anullarla
# mentre se la usavo, moltiplico per 1, cosi il suo valore influenza ancora gli altri neuroni 
def relu_derivative(x):
    return (x > 0).astype(float)

# la rete neurale deve prevedere il risultato, per prevederlo usa le probabilità, tutte le opzioni devono
# sommare a 100, percio per evitare calcolo con numeri grandi ma ottenendo lo stesso risultato, si fa in questo modo
# la softmax prende tutti i valori e li schiaccia tra 0 e 1, in modo che la loro somma faccia 1
# cosi alla fine avrò tipo ( A = 0.2 , B = 0.1 , C = 0.7) e la rete sapra che molto probabilmente ho fatto la C
def softmax(x):
    exponent = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exponent / np.sum(exponent, axis=1, keepdims=True)

# =======================================================================================================================

# ================================================ FUNZIONI DI LOSS =====================================================
# le funzioni di loss sono le funzioni che danno voti alla rete
# voto alto = la rete sta sparando a caso, non ragiona
# voto basso = la rete ha trovato un metodo per fare le cose giuste (generalizzando, pero col rischio della overfitting, ovvero impara le cose a memoria e non capisce il concetto)

# per il nostro caso la migliore e la cross entropy
# la cross entropy prende i valori previsti dalla rete (softmax) e i valori veri, e calcola quanto si discostano
def cross_entropy(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # evita log(0) = -inf
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# =======================================================================================================================

# =============================================== FORWARD PASS ========================================================
# forward = passaggi in avanti (nella rete)
# significa che prendo i valori di input e li uso per calcolare i valori di output
# ho bisogno di attivazioni intermedie, perche ogni layer influenza il successivo, quindi ho bisogno di salvarmele per poter fare il backward pass in seguito

def forward_pass(X):
    # prendo X e lo chiamo attivazione iniziale
    attivazioni = [X]
    a = X

    # ciclo for per tutti i layer tranne l'ultimo
    # sommo i pesi
    for i in range(len(pesi) - 1):
        z = np.dot(a, pesi[i]) + bias[i]
        a = relu(z)
        attivazioni.append(a)
    
    # softmax per le probabilità
    z = np.dot(a, pesi[-1]) + bias[-1]
    a = softmax(z)
    attivazioni.append(a)

    return attivazioni # ritorno le nuove attivazioni per ogni layer 

# =======================================================================================================================

# =============================================== BACKWARD PASS =======================================================
# backward = passaggi indietro (nella rete)
# significa che prendo i valori di output e li uso per ricalcolare i pesi e il bias
# questo procedimento serve per modificare il giudizio della rete
# per riuscire a fare il backward pass devo aver preso i valori di attivazione per ogni layer durante il forward pass
# quindi glieli passo come input e lui modifica
# il risultato e una variazione dei pesi per abituarsi a non sbagliare 
# algoritmo 
# 1. prendo output
# 2. calcolo errore (cross entropy)
# 3. prendo ultimo layer di attivazioni
# 4. calcolo gradiente (come impatta l'errore sui pesi)
# 5. propagazione all'indietro (come impatta l'errore sui pesi del layer precedente)
# 6. output finale (più vicino a quello giusto)

def backward_pass(attivazioni, Y_true, learing_rate):
    delta = attivazioni[-1] - Y_true

    for i in reversed(range(len(pesi))):
        gradiente_pesi = attivazioni[i].T.dot(delta) / len(Y_true)
        gradiente_bias = np.sum(delta, axis=0, keepdims=True) / len(Y_true)

        if i > 0:
            delta = delta.dot(pesi[i].T) * relu_derivative(attivazioni[i])

        pesi[i] -= learing_rate * gradiente_pesi
        bias[i] -= learing_rate * gradiente_bias
        
# =======================================================================================================================

# =================================================== TRAINING LOOP ===================================================

epoche = 100
learing_rate = 0.01
batch = 32 # 32 frame alla volta

# facciamo un training loop che divide l'intero dataset in piccoli pezzi che vengono chiamati batch
# il vantaggio del batch e che alleno la rete a piccoli step, raffinando i pesi sul singono passo
# piuttosto che utilizzare molta computazione e aggiornarli fortemente dopo un giro completo del dataset

storico_loss = []
storico_accuracy = []

for epoca in range(epoche):
    # mescolo i dati ad ogni epoca (non ogni batch). Evito che impara a memoria
    indici_batch = np.random.permutation(len(X_training))
    X_mescolato = X_training[indici_batch]
    Y_mescolato = Y_training[indici_batch]

    # divisione in batch
    for i in range(0, len(X_training), batch):
        X_batch = X_mescolato[i : i + batch]
        Y_batch = Y_mescolato[i : i + batch]

        # fase di attivazione (fp)
        layer_attivazione = forward_pass(X_batch)

        # fase di backward (bp)
        backward_pass(layer_attivazione, Y_batch, learing_rate)
    
    # calcolo loss e accuracy ad ogni epoca (per lo storico)
    attivazione_complessiva = forward_pass(X_training)
    loss = cross_entropy(attivazione_complessiva[-1], Y_training)
    predizioni = np.argmax(attivazione_complessiva[-1], axis=1)
    predizioni_corrette = np.argmax(Y_training, axis=1)
    accuratezza = np.mean(predizioni == predizioni_corrette) * 100

    storico_loss.append(round(float(loss), 4))
    storico_accuracy.append(round(float(accuratezza), 2))

    # stampo ogni 10 epoche
    if epoca % 10 == 0:
        print(f"Epoca {epoca}, Loss Media: {loss:.4f}, Accuratezza: {accuratezza:.2f}%")


# final valutation sul test set (non lo ha mai visto e provo a capire se generalizza bene)
attivazione_test = forward_pass(X_testing)
loss_test = cross_entropy(attivazione_test[-1], Y_testing)
predizioni_test = np.argmax(attivazione_test[-1], axis=1)
predizioni_corrette_test = np.argmax(Y_testing, axis=1)
accuratezza_test = np.mean(predizioni_test == predizioni_corrette_test) * 100

print(f"Accuratezza sul training set: {accuratezza:.2f}%")
print(f"Accuratezza sul test set: {accuratezza_test:.2f}%")

# =================================================== SALVATAGGIO STORICO ===================================================
# salvo lo storico di questo training in un file json
# cosi posso confrontare i vari training e visualizzarli con visualizza.py

pesi_medi_abs = [round(float(np.mean(np.abs(p))), 6) for p in pesi]
pesi_medi_segno = [round(float(np.mean(p)), 6) for p in pesi]

storico_path = 'modelli/storico.json'
if os.path.exists(storico_path):
    with open(storico_path, 'r') as f:
        storico = json.load(f)
else:
    storico = []

run_corrente = {
    "data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "loss_per_epoca": storico_loss,
    "accuracy_per_epoca": storico_accuracy,
    "accuracy_test": round(float(accuratezza_test), 2),
    "pesi_medi_abs": pesi_medi_abs,
    "pesi_medi_segno": pesi_medi_segno
}

storico.append(run_corrente)

with open(storico_path, 'w') as f:
    json.dump(storico, f, indent=2)

print(f"Storico aggiornato ({len(storico)} training salvati)")

# =================================================== SALVATAGGIO MODELLO ===================================================
# salvo il modello solo se e il migliore in accuracy test

migliore_precedente = max(
    (s['accuracy_test'] for s in storico[:-1]),
    default=-1
)

if accuratezza_test >= migliore_precedente:
    salvataggio = {}
    for i, (p, b) in enumerate(zip(pesi, bias)):
        salvataggio[f'pesi_{i}'] = p
        salvataggio[f'bias_{i}'] = b

    np.savez('modelli/modello.npz', **salvataggio)
    print(f"Nuovo modello migliore salvato! Accuracy test: {accuratezza_test:.2f}%")

    controllo = np.load('modelli/modello.npz')
    for i in range(len(pesi)):
        print(f"  layer {i}: pesi {controllo[f'pesi_{i}'].shape} | bias {controllo[f'bias_{i}'].shape}")
else:
    print(f"Non e il migliore (migliore: {migliore_precedente:.2f}%)")
