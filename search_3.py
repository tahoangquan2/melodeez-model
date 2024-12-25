import faiss
import numpy as np
import os
import json
import csv
from tqdm import tqdm

def validate_and_reshape_embedding(embedding, expected_dim=512):
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)

    if len(embedding.shape) == 1:
        if embedding.shape[0] != expected_dim:
            raise ValueError(f"Invalid embedding dimension: {embedding.shape[0]}, expected {expected_dim}")
        embedding = embedding.reshape(1, -1)
    elif len(embedding.shape) == 2:
        if embedding.shape[1] != expected_dim:
            raise ValueError(f"Invalid embedding dimension: {embedding.shape[1]}, expected {expected_dim}")
        if embedding.shape[0] != 1:
            embedding = embedding.reshape(1, -1)
    else:
        raise ValueError(f"Invalid embedding shape: {embedding.shape}")

    return embedding

def search_3(input_folder, output_folder):
    query_folder = os.path.join(input_folder, "embedding")
    results_folder = os.path.join(output_folder, "results")
    os.makedirs(results_folder, exist_ok=True)

    skipped_files = []

    try:
        index_path = os.path.join("output", "output8", "song_index.faiss")
        mapping_path = os.path.join("output", "output8", "index_mapping.json")
        metadata_path = os.path.join("output", "output7", "metadata.csv")

        print(f"Loading index from: {index_path}")
        index = faiss.read_index(index_path)
        print(f"Loading mappings from: {mapping_path}")
        index_to_id = load_mappings(mapping_path)
        print(f"Loading metadata from: {metadata_path}")
        metadata = load_metadata(metadata_path)

        print(f"Loaded index with dimension: {index.d}")
    except Exception as e:
        raise RuntimeError(f"Failed to load required files: {e}")

    query_files = [f for f in os.listdir(query_folder) if f.endswith('_embedding.npy')]
    results = {}

    for query_file in tqdm(query_files, desc="Processing queries"):
        try:
            query_path = os.path.join(query_folder, query_file)
            query_embedding = np.load(query_path)
            print(f"\nProcessing {query_file}")
            print(f"Initial query shape: {query_embedding.shape}")

            query_embedding = validate_and_reshape_embedding(query_embedding)
            print(f"Reshaped query shape: {query_embedding.shape}")

            distances, indices = index.search(query_embedding.astype(np.float32), 10)
            print(f"Search results shape - distances: {distances.shape}, indices: {indices.shape}")

            query_results = []
            for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
                if idx == -1:
                    continue

                song_id = index_to_id[idx]
                query_results.append({
                    'rank': rank,
                    'song_id': song_id,
                    'song_name': metadata[song_id]['song'],
                    'info': metadata[song_id]['info'],
                    'distance': float(distance)
                })

            query_name = os.path.splitext(query_file)[0].replace('_embedding', '')
            results[query_name] = {
                'matches': query_results
            }

        except Exception as e:
            print(f"\nError processing {query_file}: {str(e)}")
            skipped_files.append({'file': query_file, 'error': str(e)})
            continue

    results_path = os.path.join(results_folder, "search_results.json")
    error_path = os.path.join(results_folder, "search_errors.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    if skipped_files:
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump({'skipped_files': skipped_files}, f, indent=2)

    return results

def load_mappings(mapping_path):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, 'r', encoding='utf-8') as f:
        try:
            mappings = json.load(f)
            if 'metadata' not in mappings:
                raise ValueError("Missing metadata in mapping file")
            return {int(k): v['id'] for k, v in mappings['metadata'].items()}
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in mapping file")

def load_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['id']] = {
                'song': row['song'],
                'info': row['info']
            }
    return metadata
