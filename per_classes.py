from datasets import load_dataset
from collections import Counter

# Cargar dataset
dataset = load_dataset("CristianR8/BINARY-IA4CACAO-RGB")

# Obtener nombres de clases
id2label = dataset["train"].features["label"].names

def contar_muestras_por_split(ds, split="train"):
    labels = ds[split]["label"]
    counts = Counter(labels)
    print(f"\nDistribución en split: {split}")
    for i, count in sorted(counts.items()):
        print(f"Clase {i} ({id2label[i]}): {count} muestras")

# Contar en train
contar_muestras_por_split(dataset, "train")

# Contar en test
contar_muestras_por_split(dataset, "test")
