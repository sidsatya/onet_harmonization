"""Step 2: cluster semantically similar task statements into canonical IDs.

This step creates embeddings for cleaned tasks, builds a nearest-neighbor graph,
and assigns connected-component IDs (`canon_id`) used downstream.
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pipeline_utils import INTERMEDIATE_DIR, clean_text


def run_step2(similarity_threshold: float = 0.97, k_neighbors: int = 50) -> None:
    load_dotenv()
    print("[step2] Loading unique task statements for clustering...")
    data = pd.read_csv(INTERMEDIATE_DIR / "unique_task_statements.csv")
    data["task_clean"] = data["Task"].apply(clean_text)
    data = data.drop_duplicates("task_clean").reset_index(drop=True)
    print(f"[step2] Using {len(data):,} unique cleaned task statements")

    try:
        import faiss
        import networkx as nx
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("step2 requires faiss-cpu, networkx, and openai packages.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for step2 clustering.")
    client = OpenAI(api_key=api_key)

    emb_path = INTERMEDIATE_DIR / "task_embeddings.npy"
    if emb_path.exists():
        print("[step2] Found cached embeddings, loading from disk")
        embeddings = np.load(emb_path)
    else:
        print("[step2] Creating embeddings from OpenAI API")
        embeddings_list = []
        batch_size = 512
        for i in range(0, len(data), batch_size):
            batch = data["task_clean"].iloc[i : i + batch_size].tolist()
            resp = client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings_list.extend([x.embedding for x in resp.data])
        embeddings = np.array(embeddings_list, dtype=np.float32)
        np.save(emb_path, embeddings)
        print("[step2] Saved embeddings cache to intermediate_data/task_embeddings.npy")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(embeddings, min(k_neighbors, len(data)))

    graph = nx.Graph()
    graph.add_nodes_from(range(len(data)))
    for i, neighbors in enumerate(indices):
        for j, score in zip(neighbors, distances[i]):
            if i < j and score >= similarity_threshold:
                graph.add_edge(i, j)

    canonical_map = {}
    for cid, component in enumerate(nx.connected_components(graph), start=1):
        canon_id = f"C{cid:05d}"
        for idx in component:
            canonical_map[idx] = canon_id

    data["canon_id"] = data.index.map(canonical_map)
    data.to_csv(INTERMEDIATE_DIR / "task_statements_with_canon_id.csv", index=False)
    print(f"[step2] Saved canonical task clusters for {data['canon_id'].nunique():,} groups")
