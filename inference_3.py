import torch
import numpy as np
import os
import csv
import faiss
import json
from tqdm import tqdm

def load_embeddings(folder_path):
    """
    Load all embeddings and create a mapping to their IDs
    """
    print("Loading embeddings...")
    embeddings = []
    id_to_index = {}  # Maps song ID to index in embeddings list
    index_to_id = {}  # Maps FAISS index to song ID

    # Read metadata to get ID mapping
    metadata_path = os.path.join(folder_path, "metadata.csv")
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(tqdm(list(reader))):
            embedding_path = os.path.join(folder_path, "song", row['song'])
            if os.path.exists(embedding_path):
                # Load embedding
                embedding = np.load(embedding_path)
                embeddings.append(embedding.reshape(1, -1))

                # Store mappings
                song_id = row['id']
                id_to_index[song_id] = idx
                index_to_id[idx] = song_id

    # Concatenate all embeddings
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        raise ValueError("No embeddings found in the specified folder")

    return embeddings, id_to_index, index_to_id

def create_faiss_index(embeddings, index_type="L2"):
    """
    Create and populate a FAISS index
    """
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]

    if index_type == "L2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "IP":  # Inner Product
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(embeddings)
    return index

def main():
    # Configuration
    embeddings_folder = "output/output7"
    output_folder = "output/output8"
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load embeddings and create mappings
        embeddings, id_to_index, index_to_id = load_embeddings(embeddings_folder)
        print(f"Loaded {len(embeddings)} embeddings")

        # Create FAISS index
        index = create_faiss_index(embeddings)
        print("FAISS index created")

        # Save the index and mappings
        faiss_path = os.path.join(output_folder, "song_index.faiss")
        faiss.write_index(index, faiss_path)

        mapping_path = os.path.join(output_folder, "index_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({
                'index_to_id': index_to_id,
                'id_to_index': id_to_index
            }, f)

        print(f"Index saved to {faiss_path}")
        print(f"Mappings saved to {mapping_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
