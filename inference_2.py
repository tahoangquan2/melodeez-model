import torch
import numpy as np
import os
import csv
from train_model_3 import ResNetFace, IRBlock
from tqdm import tqdm
import torch.nn.functional as F

def load_and_preprocess(npy_path, target_width=630):
    """
    Load and preprocess the numpy file to match model's expected input shape
    """
    # Load the numpy file
    mel_spec = np.load(npy_path)

    # Convert to tensor and add batch and channel dimensions
    # Shape becomes (1, 1, 80, width)
    mel_spec = torch.from_numpy(mel_spec).float()
    mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)

    # Resize to target width using interpolation
    # Use 'bilinear' interpolation as it works well with spectrograms
    mel_spec = F.interpolate(
        mel_spec,
        size=(80, target_width),
        mode='bilinear',
        align_corners=False
    )

    return mel_spec

def process_inference_data(input_folder, output_folder, model_path):
    """
    Extract embeddings from song files using the trained model
    """
    # Construct correct input/output paths
    input_folder = os.path.join(input_folder, "output6")
    output_folder = os.path.join(output_folder, "output7")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=True)
    model = model.to(device)

    # Load trained weights
    try:
        state_dict = torch.load(model_path, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set model to evaluation mode
    model.eval()

    # Create output directory structure
    os.makedirs(os.path.join(output_folder, "song"), exist_ok=True)

    # Read metadata
    metadata_path = os.path.join(input_folder, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    output_metadata = []

    print("Reading metadata and processing files...")
    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    with torch.no_grad():
        for row in tqdm(rows, desc="Processing files"):
            try:
                # Construct input path
                input_path = os.path.join(input_folder, "song", row['song'])
                if not os.path.exists(input_path):
                    print(f"Warning: Input file not found: {input_path}")
                    continue

                output_filename = os.path.splitext(row['song'])[0] + "_embedding.npy"
                output_path = os.path.join(output_folder, "song", output_filename)

                # Load and preprocess the spectrogram
                input_tensor = load_and_preprocess(input_path)
                input_tensor = input_tensor.to(device)

                # Get embedding
                embedding = model(input_tensor)

                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)

                # Convert to numpy and save
                embedding_np = embedding.cpu().numpy()
                np.save(output_path, embedding_np)

                # Add to metadata with new filename
                output_metadata.append({
                    'id': row['id'],
                    'song': output_filename,
                    'info': row['info']
                })

            except Exception as e:
                print(f"Error processing file {row['song']}: {str(e)}")
                if 'input_tensor' in locals():
                    print(f"Input tensor shape was: {input_tensor.shape}")
                continue

    # Save new metadata
    output_metadata_path = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['id', 'song', 'info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_metadata)

    print(f"Processing completed! Processed {len(output_metadata)} files")
    print(f"Embeddings saved to: {os.path.join(output_folder, 'song')}")
    print(f"New metadata saved to: {output_metadata_path}")
