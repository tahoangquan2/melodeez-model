import torch
import numpy as np
import os
import csv
from torch.nn import DataParallel
from train_model_3 import ResNetFace
from train_model_2 import Config
from tqdm import tqdm
import torch.nn.functional as F

class EmbeddingGenerator:
    def __init__(self, model_path, device=None):
        self.config = Config()
        self.embedding_dim = self.config.embedding_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Input shape: {self.config.input_shape}")
        print(f"Embedding dimension: {self.embedding_dim}")
        self.model = self._initialize_model(model_path)

    def _initialize_model(self, model_path):
        model = ResNetFace(feature_dim=self.embedding_dim)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        if all(k.startswith('module.') for k in state_dict.keys()):
            model = DataParallel(model)
        else:
            if torch.cuda.device_count() > 1:
                # Create new state dict with 'module.' prefix
                new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                state_dict = new_state_dict
                model = DataParallel(model)

        # Load the state dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Failed to load state dict with strict=True: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()
        return model

    def load_and_preprocess(self, npy_path):
        mel_spec = np.load(npy_path)
        mel_spec = torch.from_numpy(mel_spec).float()
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)

        target_shape = (
            self.config.input_shape[1],
            self.config.input_shape[2]
        )

        mel_spec = F.interpolate(
            mel_spec,
            size=target_shape,
            mode='bilinear',
            align_corners=False
        )

        return mel_spec

    def generate_embedding(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            embedding = self.model(input_tensor)

            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"Invalid embedding dimension: {embedding.shape[1]}, expected {self.embedding_dim}")

            return embedding.cpu().numpy()

def process_inference_data(input_folder, output_folder, model_path):
    input_folder = os.path.join(input_folder, "output6")
    output_folder = os.path.join(output_folder, "output7")
    os.makedirs(os.path.join(output_folder, "song"), exist_ok=True)

    generator = EmbeddingGenerator(model_path)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    output_metadata = []
    failed_files = []

    print("Reading metadata and processing files...")
    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        batch_size = generator.config.train_batch_size
        print(f"Using batch size: {batch_size}")

        for i in tqdm(range(0, len(rows), batch_size), desc="Processing batches"):
            batch_rows = rows[i:i + batch_size]

            for row in batch_rows:
                try:
                    input_path = os.path.join(input_folder, "song", row['song'])
                    if not os.path.exists(input_path):
                        print(f"Warning: Input file not found: {input_path}")
                        failed_files.append(row)
                        continue

                    output_filename = os.path.splitext(row['song'])[0] + "_embedding.npy"
                    output_path = os.path.join(output_folder, "song", output_filename)

                    input_tensor = generator.load_and_preprocess(input_path)

                    try:
                        embedding = generator.generate_embedding(input_tensor)
                        np.save(output_path, embedding)

                        output_metadata.append({
                            'id': row['id'],
                            'song': output_filename,
                            'info': row['info']
                        })

                    except ValueError as ve:
                        print(f"Error with embedding generation for {row['song']}: {ve}")
                        failed_files.append(row)

                except Exception as e:
                    print(f"Error processing file {row['song']}: {str(e)}")
                    failed_files.append(row)

    output_metadata_path = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['id', 'song', 'info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_metadata)

    print(f"Processing completed! Successfully processed {len(output_metadata)} files")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")
    print(f"Embeddings saved to: {os.path.join(output_folder, 'song')}")
    print(f"Metadata saved to: {output_metadata_path}")
