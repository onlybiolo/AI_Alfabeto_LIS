import numpy as np
import json

dataset = np.load('dati/dataset.npy')
data = {"frames": dataset.tolist()}
with open('dataset_export.json', 'w') as f:
    json.dump(data, f)
print(f"Esportati {len(dataset)} frame")