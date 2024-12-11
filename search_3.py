import faiss
import numpy as np
import os
import json
from tqdm import tqdm
import csv

def load_faiss_index(index_path):
    """Load the FAISS index from file"""
    try:
        index = faiss.read_index(index_path)
        print(f"Successfully loaded FAISS index from {index_path}")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def load_mappings(mapping_path):
    """Load the index to ID mappings"""
    try:
        with open(mapping_path, 'r') as f:
            mappings = json.load(f)
        return mappings
    except Exception as e:
        print(f"Error loading mappings: {e}")
        return None

def load_metadata(metadata_path):
    """Load the original metadata to get song information"""
    metadata = {}
    try:
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['id']] = {
                    'song': row['song'],
                    'info': row['info']
                }
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def search_3(input_folder, output_folder, top_k=5):
    """
    Perform similarity search using the query embeddings
    Args:
        input_folder: folder containing query embeddings
        output_folder: folder to save search results
        top_k: number of similar songs to return (default: 5)
    """
    # Set up paths
    query_folder = os.path.join(input_folder, "embedding")
    index_path = os.path.join("output", "output8", "song_index.faiss")
    mapping_path = os.path.join("output", "output8", "index_mapping.json")
    metadata_path = os.path.join("output", "output7", "metadata.csv")

    # Load FAISS index
    index = load_faiss_index(index_path)
    if index is None:
        return

    # Load mappings
    mappings = load_mappings(mapping_path)
    if mappings is None:
        return

    # Fix: Convert string keys to integers for index_to_id
    index_to_id = {}
    for k, v in mappings['index_to_id'].items():
        try:
            index_to_id[int(k)] = v
        except ValueError:
            print(f"Warning: Invalid index mapping key: {k}")
            continue

    # Load metadata
    metadata = load_metadata(metadata_path)
    if metadata is None:
        return

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    results_path = os.path.join(output_folder, "search_results.json")

    # Process each query embedding
    results = {}
    query_files = [f for f in os.listdir(query_folder) if f.endswith('_embedding.npy')]

    print("Performing similarity search...")
    for query_file in tqdm(query_files):
        try:
            # Load query embedding
            query_path = os.path.join(query_folder, query_file)
            query_embedding = np.load(query_path)

            # Ensure embedding is in the correct shape
            if len(query_embedding.shape) == 3:  # If shape is (1, 1, dim)
                query_embedding = query_embedding.reshape(1, -1)

            # Perform search
            distances, indices = index.search(query_embedding, top_k)

            # Format results
            query_results = []
            for i in range(top_k):
                idx = indices[0][i]
                distance = distances[0][i]

                # Fix: Use integer index directly
                if idx not in index_to_id:
                    print(f"Warning: Index {idx} not found in mappings")
                    continue

                song_id = index_to_id[idx]
                if song_id not in metadata:
                    print(f"Warning: Song ID {song_id} not found in metadata")
                    continue

                song_info = metadata[song_id]

                query_results.append({
                    'rank': i + 1,
                    'song_id': song_id,
                    'distance': float(distance),
                    'song_name': song_info['song'],
                    'info': song_info['info']
                })

            # Store results for this query
            query_name = os.path.splitext(query_file)[0].replace('_embedding', '')
            results[query_name] = query_results

        except Exception as e:
            print(f"Error processing query {query_file}: {e}")
            continue

    # Save results
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Search results saved to {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    input_folder = "search"
    output_folder = "search/results"
    search_3(input_folder, output_folder)

if __name__ == "__main__":
    main()
