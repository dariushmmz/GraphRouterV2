# embed_queries.py
import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_router_csv(path="data/router_data.csv"):
    df = pd.read_csv(path)
    return df


def unique_queries_from_router(df, num_llms):
    """
    Given router_data where rows = num_queries * num_llms,
    return the unique query indices (starting row indices for each query)
    and the query texts corresponding to them.
    """
    total = len(df)
    if total % num_llms != 0:
        raise ValueError(f"rows ({total}) not divisible by num_llms ({num_llms})")
    unique_idxs = list(range(0, total, num_llms))
    queries = [df.loc[i, 'query'] for i in unique_idxs]
    return unique_idxs, queries


def compute_embeddings(queries, model_name="paraphrase-multilingual-MiniLM-L12-v2", batch_size=64, device=None):
    """
    Compute sentence embeddings using sentence-transformers.
    Returns numpy array shape (len(queries), dim)
    """
    print("Loading model:", model_name)
    model = SentenceTransformer(model_name, device=device)
    all_emb = []
    for i in tqdm(range(0, len(queries), batch_size), desc="Embedding batches"):
        batch = queries[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_emb.append(emb)
    all_emb = np.vstack(all_emb)
    return all_emb


def save_embeddings(arr, out_path="data/query_semantic_embeddings.pkl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(arr, f)
    print("Saved embeddings to", out_path)


def build_semantic_embedd(router_csv="data/router_data.csv", num_llms=7,
         model_name="paraphrase-multilingual-MiniLM-L12-v2", batch_size=64,
         out_path="data/query_semantic_embeddings.pkl",
         device=None):
    df = load_router_csv(router_csv)

    unique_idxs, queries = unique_queries_from_router(df, num_llms)
    print(f"Found {len(queries)} unique queries, using {model_name} to embed them.")

    embs = compute_embeddings(queries, model_name=model_name, batch_size=batch_size, device=device)
    print("Embeddings shape:", embs.shape)

    save_embeddings(embs, out_path)



