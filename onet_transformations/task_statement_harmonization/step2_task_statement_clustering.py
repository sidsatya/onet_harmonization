"""
This script performs harmonization of O*NET task statements by clustering them based on semantic similarity.

The workflow is as follows:
1.  **Load Data**: Reads a CSV file of unique task statements.
2.  **Generate Embeddings**: Creates vector embeddings for each task statement using an OpenAI model. It caches these embeddings locally to avoid re-computation.
3.  **Find Nearest Neighbors**: Uses FAISS to efficiently find the 50 nearest neighbors for each task based on cosine similarity.
4.  **Cluster Tasks**: Constructs a graph where tasks are nodes and an edge exists if their similarity is above a set threshold (0.97). The connected components of this graph form the clusters.
5.  **Assign Canonical IDs**: Assigns a unique canonical ID to all tasks within the same cluster.
6.  **Save Results**: Saves the task statements with their new canonical IDs to a CSV file and creates a sample file for manual review.
"""
import os
import pandas as pd
import numpy as np
import faiss
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
import re

# --- Configuration ---
# Centralized configuration for file paths, model parameters, and algorithm settings.
# This makes it easier to modify the script's behavior without changing the core logic.
CONFIG = {
    "data_directory": "/Users/sidsatya/dev/ailabor/onet_transformations/task_statement_harmonization/intermediate_data/",
    "input_csv": "unique_task_statements.csv",
    "embeddings_file": "task_embeddings.npy",
    "output_csv": "task_statements_with_canon_id.csv",
    "sample_output_csv": "sample_task_statements_with_canon_id.csv",
    "embedding_model": "text-embedding-ada-002",
    "embedding_batch_size": 512,
    "faiss_k_neighbors": 50,
    "clustering_threshold": 0.97,
    "sample_size": 100,
    "random_state": 42,
}

# --- Helper Functions ---

def clean_text(text: str) -> str:
    # 2) remove any occurrences of "x92"
    text = re.sub(r'\x92', "'", text)
    # 1) replace any punctuation (i.e. non-word, non-space) with a space
    text = re.sub(r'[^\w\s]', '', text)
    # 3) lowercase everything
    text = text.lower()
    # 4) collapse multiple whitespace into one, and strip ends
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(directory, filename):
    """
    Loads task statements from a CSV file, cleans them, and prepares them for processing.

    Args:
        directory (str): The directory where the input file is located.
        filename (str): The name of the input CSV file.

    Returns:
        pd.DataFrame: A DataFrame with the loaded and cleaned task statements.
    """
    print("Step 1: Loading and cleaning data...")
    data_path = os.path.join(directory, filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data file not found at: {data_path}")

    data = pd.read_csv(data_path, encoding='latin1')
    
    # Clean the 'Task' column by converting to lowercase, replacing multiple whitespace
    # characters with a single space, and removing leading/trailing whitespace.
    data['task_clean'] = data['Task'].apply(clean_text)

    # Drop duplicates based on the cleaned task statements.
    data = data.drop_duplicates(subset='task_clean', keep='first').reset_index(drop=True)

    # Ensure the 'task_clean' column is not empty.
    if data['task_clean'].isnull().any() or data['task_clean'].str.strip().eq('').any():
        raise ValueError("Some task statements are empty after cleaning. Please check the input data.")

    print(f"Loaded {len(data)} unique task statements from '{filename}'.")
    return data

def get_openai_embeddings_batch(texts, model, client):
    """
    Retrieves embeddings for a batch of texts using the OpenAI API.

    Args:
        texts (list): A list of strings to embed.
        model (str): The name of the OpenAI embedding model to use.
        client (OpenAI): The OpenAI API client.

    Returns:
        list: A list of embedding vectors.
    """
    try:
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"An error occurred while fetching embeddings: {e}")
        # Depending on the use case, you might want to retry or handle this more gracefully.
        return []

def get_or_create_embeddings(data, config, client):
    """
    Loads embeddings from a file if they exist, otherwise creates them using the OpenAI API.

    Args:
        data (pd.DataFrame): DataFrame containing the 'task_clean' column.
        config (dict): The configuration dictionary.
        client (OpenAI): The OpenAI API client.

    Returns:
        np.ndarray: A numpy array of embeddings.
    """
    print("\nStep 2: Getting or creating embeddings for task statements...")
    embeddings_path = os.path.join(config["data_directory"], config["embeddings_file"])

    if os.path.exists(embeddings_path):
        print(f"Embeddings file found. Loading from '{config['embeddings_file']}'...")
        embeddings_matrix = np.load(embeddings_path)
        print(f"Loaded {embeddings_matrix.shape[0]} embeddings.")
        return embeddings_matrix

    print("Embeddings file not found. Creating new embeddings...")
    embeddings = []
    batch_size = config["embedding_batch_size"]
    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(0, len(data), batch_size):
        batch_texts = data['task_clean'].iloc[i:i + batch_size].tolist()
        print(f"Processing batch {i // batch_size + 1}/{num_batches}...")
        batch_embeddings = get_openai_embeddings_batch(batch_texts, config["embedding_model"], client)
        if batch_embeddings:
            embeddings.extend(batch_embeddings)

    if not embeddings:
        raise ValueError("Failed to create any embeddings. Please check API connection and key.")

    embeddings_matrix = np.vstack(embeddings)
    np.save(embeddings_path, embeddings_matrix)
    print(f"Embeddings created and saved to '{config['embeddings_file']}'.")
    return embeddings_matrix

def find_nearest_neighbors(embeddings, k):
    """
    Uses FAISS to find the k-nearest neighbors for each embedding based on cosine similarity.

    Args:
        embeddings (np.ndarray): The matrix of embeddings.
        k (int): The number of nearest neighbors to find for each item.

    Returns:
        tuple: A tuple containing:
            - D (np.ndarray): An array of distances (cosine similarities) to the neighbors.
            - I (np.ndarray): An array of indices of the neighbors.
    """
    print(f"\nStep 3: Finding the top {k} nearest neighbors using FAISS...")
    
    # Ensure the embedding matrix is in the correct format (float32, C-contiguous) for FAISS.
    if embeddings.dtype != np.float32 or not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    num_embeddings, dim = embeddings.shape
    
    # Normalize embeddings to unit vectors (L2 norm of 1). When using dot product (Inner Product, IP)
    # on unit vectors, the result is the cosine similarity. This is a standard and efficient
    # way to perform cosine similarity searches.
    faiss.normalize_L2(embeddings)

    # Create a FAISS index for fast nearest neighbor search.
    # IndexFlatIP uses the inner product (dot product) as the similarity measure.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Searching for {k} nearest neighbors for {num_embeddings} embeddings...")
    # Search the index for the k nearest neighbors of each embedding.
    distances, indices = index.search(embeddings, k)
    print("Nearest neighbor search complete.")
    return distances, indices

def cluster_tasks(num_tasks, indices, distances, threshold):
    """
    Creates a graph of tasks and clusters them based on similarity.

    Args:
        num_tasks (int): The total number of tasks (nodes in the graph).
        indices (np.ndarray): The indices of nearest neighbors from FAISS.
        distances (np.ndarray): The distances (similarities) from FAISS.
        threshold (float): The cosine similarity threshold for creating an edge.

    Returns:
        list: A list of sets, where each set represents a cluster of task indices.
    """
    print(f"\nStep 4: Clustering tasks with a similarity threshold of {threshold}...")
    G = nx.Graph()
    G.add_nodes_from(range(num_tasks))

    # Create an edge between two tasks if their cosine similarity is above the threshold.
    # We iterate through each task's nearest neighbors.
    for i, neighbors in enumerate(indices):
        for j, score in zip(neighbors, distances[i]):
            # To avoid duplicate edges and self-loops, we only add an edge if i < j.
            if i < j and score >= threshold:
                G.add_edge(i, j)

    # Find connected components in the graph. Each component is a cluster of similar tasks.
    clusters = list(nx.connected_components(G))
    print(f"Found {len(clusters)} clusters from {num_tasks} tasks.")
    return clusters

def assign_and_save_results(data, clusters, config):
    """
    Assigns canonical IDs to tasks based on clusters and saves the results.

    Args:
        data (pd.DataFrame): The original DataFrame with task data.
        clusters (list): The list of task clusters.
        config (dict): The configuration dictionary.
    """
    print("\nStep 5: Assigning canonical IDs and saving results...")
    
    # Create a mapping from the original data index to a canonical cluster ID.
    # The format "C{:05d}" ensures a consistent, zero-padded ID (e.g., C00001).
    canonical_id_map = {}
    for cid, component in enumerate(clusters, 1):
        for task_index in component:
            canonical_id_map[task_index] = f"C{cid:05d}"
            
    data['canon_id'] = data.index.map(canonical_id_map)

    # Save the full dataset with the new canonical IDs.
    output_path = os.path.join(config["data_directory"], config["output_csv"])
    data.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'.")

    # --- Create and save a sample for manual inspection ---
    # This helps in verifying the quality of the clustering.
    
    # Group tasks by their new canonical ID.
    grouped = data.groupby('canon_id')['Task'].apply(lambda x: '; '.join(x.unique())).reset_index()
    grouped['count'] = grouped['Task'].apply(lambda x: len(x.split('; ')))
    
    # Filter for clusters that contain more than one unique task statement.
    # These are the most interesting cases where harmonization occurred.
    multi_task_clusters = grouped[grouped['count'] > 1]
    
    # Take a reproducible random sample from these interesting clusters.
    sample_size = min(len(multi_task_clusters), config["sample_size"])
    if sample_size > 0:
        sampled = multi_task_clusters.sample(n=sample_size, random_state=config["random_state"])
        sample_output_path = os.path.join(config["data_directory"], config["sample_output_csv"])
        sampled.to_csv(sample_output_path, index=False)
        print(f"A sample of {sample_size} harmonized clusters saved to '{config['sample_output_csv']}'.")
    else:
        print("No multi-task clusters found to create a sample from.")

def main():
    """
    Main function to orchestrate the task statement clustering workflow.
    """
    # --- Initialization ---
    print("Starting task statement harmonization workflow...")
    load_dotenv()  # Load environment variables from a .env file
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # --- Workflow Steps ---
    # 1. Load and clean the input data.
    data = load_and_prepare_data(CONFIG["data_directory"], CONFIG["input_csv"])

    # 2. Generate or load embeddings for the task statements.
    embeddings = get_or_create_embeddings(data, CONFIG, client)

    # 3. Find nearest neighbors using FAISS for efficient similarity search.
    distances, indices = find_nearest_neighbors(embeddings, k=CONFIG["faiss_k_neighbors"])

    # 4. Build a graph and identify clusters of similar tasks.
    clusters = cluster_tasks(len(data), indices, distances, threshold=CONFIG["clustering_threshold"])

    # 5. Assign canonical IDs and save the final results and a sample for review.
    assign_and_save_results(data, clusters, CONFIG)
    
    print("\nWorkflow completed successfully.")

if __name__ == "__main__":
    main()

