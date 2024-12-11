import torch
import numpy as np
import os
from inference_1 import process_audio_to_mel
from inference_2 import ResNetFace, IRBlock, load_and_preprocess
import librosa
import torch.nn.functional as F
from tqdm import tqdm

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def process_audio_file(audio_path, output_path, sr=22050):
    """Convert audio file to mel spectrogram"""
    try:
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=sr)
        # Convert to mel spectrogram
        spec = process_audio_to_mel(audio, sr=sr)
        # Save as numpy file
        np.save(output_path, spec)
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

def search_2(input_folder, output_folder, model_path):
    input_folder = os.path.join(input_folder, "processed")
    temp_folder = os.path.join(output_folder, "temp_spectrograms")
    output_folder = os.path.join(output_folder, "embedding")

    ensure_directory(temp_folder)
    ensure_directory(output_folder)

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

    model.eval()

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]

    if not audio_files:
        print(f"No MP3 files found in {input_folder}")
        return

    print("Processing audio files...")
    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            try:
                # Set up paths
                audio_path = os.path.join(input_folder, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                spec_path = os.path.join(temp_folder, f"{base_name}.npy")
                embedding_path = os.path.join(output_folder, f"{base_name}_embedding.npy")

                # Convert audio to mel spectrogram
                if not process_audio_file(audio_path, spec_path):
                    print(f"Failed to create spectrogram for {audio_file}")
                    continue

                # Load and preprocess the spectrogram
                input_tensor = load_and_preprocess(spec_path)
                input_tensor = input_tensor.to(device)

                # enerate embedding
                embedding = model(input_tensor)

                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)

                # Save embedding
                embedding_np = embedding.cpu().numpy()
                np.save(embedding_path, embedding_np)

            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

            if os.path.exists(spec_path):
                os.remove(spec_path)

    if os.path.exists(temp_folder):
        os.rmdir(temp_folder)

    print(f"Processing completed! Embeddings saved to: {output_folder}")

if __name__ == "__main__":
    # Example usage
    input_folder = "search"
    output_folder = "search"
    model_path = "checkpoints/resnet18_best.pth"
    search_2(input_folder, output_folder, model_path)
