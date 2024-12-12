import numpy as np
import os
import csv
import faiss
import json
from tqdm import tqdm
from train_model_2 import Config

class FAISSIndexBuilder:
    def __init__(self):
        self.config = Config()
        self.embedding_dim = self.config.num_classes
        self.embeddings = []
        self.metadata = {}
        self.failed_embeddings = []

    def validate_embedding(self, embedding, identifier):
        try:
            if embedding.shape[1] != self.embedding_dim:
                print(f"Invalid embedding dimension for {identifier}: {embedding.shape[1]}, expected {self.embedding_dim}")
                return False
            if not np.isfinite(embedding).all():
                print(f"Invalid values in embedding for {identifier}")
                return False
            return True
        except Exception as e:
            print(f"Error validating embedding for {identifier}: {e}")
            return False

    def add_embedding(self, embedding, metadata_entry):
        try:
            if self.validate_embedding(embedding, metadata_entry['song']):
                self.embeddings.append(embedding)
                self.metadata[len(self.embeddings) - 1] = metadata_entry
                return True
            else:
                self.failed_embeddings.append(metadata_entry['song'])
                return False
        except Exception as e:
            print(f"Error adding embedding for {metadata_entry['song']}: {e}")
            self.failed_embeddings.append(metadata_entry['song'])
            return False

    def build_index(self):
        if not self.embeddings:
            raise ValueError("No valid embeddings to build index")

        try:
            embeddings = np.vstack([emb.reshape(1, -1) for emb in self.embeddings])
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(embeddings)
            return index
        except Exception as e:
            print(f"Error building index: {e}")
            raise

def create_faiss_index(output_folder):
    input_folder = os.path.join(output_folder, "output7")
    output_folder = os.path.join(output_folder, "output8")
    os.makedirs(output_folder, exist_ok=True)

    try:
        index_builder = FAISSIndexBuilder()

        metadata_path = os.path.join(input_folder, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        print("Loading and validating embeddings...")
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            batch_size = index_builder.config.train_batch_size

            for i in tqdm(range(0, len(rows), batch_size)):
                batch_rows = rows[i:i + batch_size]
                for row in batch_rows:
                    try:
                        embedding_path = os.path.join(input_folder, "song", row['song'])
                        if not os.path.exists(embedding_path):
                            print(f"Embedding file not found: {embedding_path}")
                            continue

                        embedding = np.load(embedding_path)
                        index_builder.add_embedding(embedding, row)

                    except Exception as e:
                        print(f"Error processing {row['song']}: {e}")
                        index_builder.failed_embeddings.append(row['song'])

        print("Building FAISS index...")
        index = index_builder.build_index()

        print("Saving index and metadata...")
        faiss_path = os.path.join(output_folder, "song_index.faiss")
        mapping_path = os.path.join(output_folder, "index_mapping.json")

        faiss.write_index(index, faiss_path)
        with open(mapping_path, 'w') as f:
            json.dump({
                'metadata': index_builder.metadata,
                'embedding_dim': index_builder.embedding_dim,
                'total_embeddings': len(index_builder.embeddings)
            }, f, indent=2)

        print(f"Index creation completed:")
        print(f"- Total embeddings processed: {len(index_builder.embeddings)}")
        if index_builder.failed_embeddings:
            print(f"- Failed embeddings: {len(index_builder.failed_embeddings)}")
        print(f"- Index saved to: {faiss_path}")
        print(f"- Mapping saved to: {mapping_path}")

    except Exception as e:
        print(f"Error creating index: {e}")
        raise

def batch_similarity_search(index_path, mapping_path, query_embeddings, k=10):
    try:
        index = faiss.read_index(index_path)
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = np.array(query_embeddings)

        if len(query_embeddings.shape) == 2:
            if query_embeddings.shape[1] != mapping_data['embedding_dim']:
                raise ValueError(f"Invalid query embedding dimension: {query_embeddings.shape[1]}, expected {mapping_data['embedding_dim']}")
        else:
            raise ValueError("Query embeddings must be 2-dimensional")

        distances, indices = index.search(query_embeddings, k)

        results = []
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            batch_results = []
            seen_ids = set()
            for dist, idx in zip(dists, idxs):
                if idx != -1:
                    metadata = mapping_data['metadata'][str(idx)]
                    if metadata['id'] not in seen_ids:
                        batch_results.append({
                            'metadata': metadata,
                            'distance': float(dist)
                        })
                        seen_ids.add(metadata['id'])
            results.append(batch_results)

        return results

    except Exception as e:
        print(f"Error during similarity search: {e}")
        raise
