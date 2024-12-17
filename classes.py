from huggingface_hub import login
import matplotlib.pyplot as plt
from collections import Counter

login("HF_API_KEY")

from datasets import load_dataset

dataset_name = "SemilleroCV/Cocoa-dataset"
dataset = load_dataset(dataset_name)

# Count the occurrences of each class in the training set
label_counts = Counter(dataset["train"]["label"])

# Convert label IDs to their string names (if applicable)
label_names = dataset["train"].features["label"].names
label_counts_named = {label_names[label]: count for label, count in label_counts.items()}

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.bar(label_counts_named.keys(), label_counts_named.values(), color='skyblue')
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Class Distribution in SemilleroCV/Cocoa-dataset")
plt.xticks(rotation=45)
plt.show()
