from huggingface_hub import login
import matplotlib.pyplot as plt
from collections import Counter
import os
from datasets import load_dataset
import datasets
from transformers import AutoFeatureExtractor, AutoModel
import torch
from annoy import AnnoyIndex
from renumics import spotlight
from cleanlab.outlier import OutOfDistribution
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sys

load_dotenv()

api_key = os.getenv("HF_API_KEY")

if api_key:
    login(api_key)
else:
    raise ValueError("HF_API_KEY environment variable not found!")

# Load the dataset
dataset_name = "SemilleroCV/Cocoa-dataset"
dataset = load_dataset(dataset_name)

# Convert to pandas DataFrame
df = dataset['train'].to_pandas()

ft_model_name = "google/vit-base-patch16-224-in21k" # Fine-tuned model
base_model_name = "google/vit-base-patch16-224" # Base model

def extract_embeddings(model, feature_extractor, image_name="image"):
    """
    Utility to compute embeddings.
    Args:
        model: huggingface model
        feature_extractor: huggingface feature extractor
        image_name: name of the image column in the dataset
    Returns:
        function to compute embeddings
    """
    device = model.device
    def pp(batch):
        images = batch[image_name]
        inputs = feature_extractor(
            images=[x.convert("RGB") for x in images], return_tensors="pt"
        ).to(device)
        embeddings = model(**inputs).last_hidden_state[:, 0].cpu()
        return {"embedding": embeddings}
    return pp

def huggingface_embedding(
    df,
    image_name="image",
    modelname="google/vit-base-patch16-224",
    batched=True,
    batch_size=24,
):
    """
    Compute embeddings using huggingface models.
    Args:
        df: dataframe with images
        image_name: name of the image column in the dataset
        modelname: huggingface model name
        batched: whether to compute embeddings in batches
        batch_size: batch size
    Returns:
        new dataframe with embeddings
    """
    # initialize huggingface model
    feature_extractor = AutoFeatureExtractor.from_pretrained(modelname)
    model = AutoModel.from_pretrained(modelname, output_hidden_states=True)
    # create huggingface dataset from df
    dataset = datasets.Dataset.from_pandas(df).cast_column(image_name, datasets.Image())
    # compute embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device), feature_extractor, image_name)
    updated_dataset = dataset.map(extract_fn, batched=batched, batch_size=batch_size)
    df_temp = updated_dataset.to_pandas()
    df_emb = pd.DataFrame()
    df_emb["embedding"] = df_temp["embedding"]
    return df_emb


embeddings_df = huggingface_embedding(
    df,
    modelname=ft_model_name,
    batched=True,
    batch_size=24,
)
embeddings_df_found = huggingface_embedding(
    df, modelname=base_model_name, batched=True, batch_size=24
)
df["embedding_ft"] = embeddings_df["embedding"]
df["embedding_foundation"] = embeddings_df_found["embedding"]

def outlier_score_by_embeddings_cleanlab(df, embedding_name="embedding"):
    """
    Calculate outlier score by embeddings using cleanlab
        Args:
            df: dataframe with embeddings
            embedding_name: name of the column with embeddings
        Returns:
            new df_out: dataframe with outlier score
    """
    embs = np.stack(df[embedding_name].to_numpy())
    ood = OutOfDistribution()
    ood_train_feature_scores = ood.fit_score(features=np.stack(embs))
    df_out = pd.DataFrame()
    df_out["outlier_score_embedding"] = ood_train_feature_scores
    return df_out

df["outlier_score_ft"] = outlier_score_by_embeddings_cleanlab(
    df, embedding_name="embedding_ft"
)["outlier_score_embedding"]
df["outlier_score_found"] = outlier_score_by_embeddings_cleanlab(
    df, embedding_name="embedding_foundation"
)["outlier_score_embedding"]

def nearest_neighbor_annoy(
    df, embedding_name="embedding", threshold=0.3, tree_size=100
):
    """
    Find nearest neighbor using annoy.
    Args:
        df: dataframe with embeddings
        embedding_name: name of the embedding column
        threshold: threshold for outlier detection
        tree_size: tree size for annoy
    Returns:
        new dataframe with nearest neighbor information
    """
    embs = df[embedding_name]
    t = AnnoyIndex(len(embs[0]), "angular")
    for idx, x in enumerate(embs):
        t.add_item(idx, x)
    t.build(tree_size)
    images = df["image"]
    df_nn = pd.DataFrame()
    nn_id = [t.get_nns_by_item(i, 2)[1] for i in range(len(embs))]
    df_nn["nn_id"] = nn_id
    df_nn["nn_image"] = [images[i] for i in nn_id]
    df_nn["nn_distance"] = [t.get_distance(i, nn_id[i]) for i in range(len(embs))]
    df_nn["nn_flag"] = df_nn.nn_distance < threshold
    return df_nn

df_nn = nearest_neighbor_annoy(
    df, embedding_name="embedding_ft", threshold=0.3, tree_size=100
)
df["nn_image"] = df_nn["nn_image"]

df["label_str"] = df["label"].apply(lambda x: dataset['train'].features["label"].int2str(x))
dtypes = {
    "nn_image": spotlight.Image,
    "image": spotlight.Image,
    "embedding_ft": spotlight.Embedding,
    "embedding_foundation": spotlight.Embedding,
}
spotlight.show(
    df,
    dtype=dtypes,
    layout="https://spotlight.renumics.com/resources//layout_pre_post_ft.json",
)