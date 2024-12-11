import faiss
import numpy as np
import os
import json
import csv
from collections import defaultdict
from tqdm import tqdm

def load_faiss_index(index_path):
    try:
        index = faiss.read_index(index_path)
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index: {e}")

def load_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['id']] = {
                'song': row['song'],
                'info': row['info']
            }
    return metadata

def load_mappings(mapping_path):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, 'r') as f:
        try:
            mappings = json.load(f)
            if 'metadata' not in mappings:
                raise ValueError("Missing metadata in mapping file")

            return {int(k): v['id'] for k, v in mappings['metadata'].items()}
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in mapping file")

def validate_and_reshape_embedding(embedding, expected_dim=512):
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)
    elif len(embedding.shape) == 2:
        if embedding.shape[0] != 1:
            embedding = embedding.reshape(1, -1)
    elif len(embedding.shape) == 3:
        embedding = embedding.reshape(1, -1)

    if embedding.shape[1] != expected_dim:
        raise ValueError(f"Invalid embedding dimension: {embedding.shape[1]}, expected {expected_dim}")

    return embedding

def search_3(input_folder, output_folder, top_k=20):
    search_k = top_k * 3
    query_folder = os.path.join(input_folder, "embedding")
    results_folder = os.path.join(output_folder, "results")
    os.makedirs(results_folder, exist_ok=True)

    error_log = defaultdict(list)
    skipped_files = []

    try:
        index_path = os.path.join("output", "output8", "song_index.faiss")
        mapping_path = os.path.join("output", "output8", "index_mapping.json")
        metadata_path = os.path.join("output", "output7", "metadata.csv")

        print(f"Loading index from: {index_path}")
        print(f"Loading mappings from: {mapping_path}")
        print(f"Loading metadata from: {metadata_path}")

        if not all(os.path.exists(p) for p in [index_path, mapping_path, metadata_path]):
            missing = [p for p in [index_path, mapping_path, metadata_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"Missing required files: {missing}")

        index = load_faiss_index(index_path)
        index_to_id = load_mappings(mapping_path)
        metadata = load_metadata(metadata_path)

        print(f"Loaded index with dimension: {index.d}")
        print(f"Number of mappings: {len(index_to_id)}")
        print(f"Number of metadata entries: {len(metadata)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load required files: {e}")

    query_files = [f for f in os.listdir(query_folder) if f.endswith('_embedding.npy')]
    print(f"Found {len(query_files)} query files")
    results = {}

    for query_file in tqdm(query_files, desc="Processing queries"):
        try:
            query_path = os.path.join(query_folder, query_file)
            query_embedding = np.load(query_path)
            print(f"\nLoaded query embedding shape: {query_embedding.shape}")

            query_embedding = validate_and_reshape_embedding(query_embedding)
            print(f"Reshaped query embedding shape: {query_embedding.shape}")

            distances, indices = index.search(query_embedding.astype(np.float32), search_k)
            print(f"Search returned {len(indices[0])} results")

            unique_songs = {}
            for idx, distance in zip(indices[0], distances[0]):
                if idx not in index_to_id:
                    error_log["invalid_indices"].append(idx)
                    continue

                song_id = index_to_id[idx]
                if song_id not in metadata:
                    error_log["missing_metadata"].append(song_id)
                    continue

                if song_id not in unique_songs or distance < unique_songs[song_id]['distance']:
                    unique_songs[song_id] = {
                        'distance': float(distance),
                        'confidence': float(1 / (1 + distance))
                    }

            sorted_results = sorted(unique_songs.items(), key=lambda x: x[1]['distance'])[:top_k]
            query_results = []

            for rank, (song_id, metrics) in enumerate(sorted_results, 1):
                query_results.append({
                    'rank': rank,
                    'song_id': song_id,
                    'song_name': metadata[song_id]['song'],
                    'info': metadata[song_id]['info'],
                    'distance': metrics['distance'],
                    'confidence': metrics['confidence']
                })

            query_name = os.path.splitext(query_file)[0].replace('_embedding', '')
            results[query_name] = {
                'matches': query_results,
                'total_unique_matches': len(unique_songs),
                'actual_matches_returned': len(query_results)
            }

        except Exception as e:
            print(f"\nError processing {query_file}: {str(e)}")
            skipped_files.append({'file': query_file, 'error': str(e)})
            continue

    results_path = os.path.join(results_folder, "search_results.json")
    error_path = os.path.join(results_folder, "search_errors.json")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    error_report = {
        'skipped_files': skipped_files,
        'error_logs': dict(error_log)
    }
    with open(error_path, 'w') as f:
        json.dump(error_report, f, indent=2)

    print(f"\nSearch complete. Results saved to {results_path}")
    print(f"Processed {len(results)} queries successfully")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files. Check {error_path} for details")
