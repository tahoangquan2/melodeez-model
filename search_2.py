import numpy as np
import os
import tempfile
from process_audio_3 import process_audio as process_mel
from inference_2 import EmbeddingGenerator
from train_model_2 import Config
import librosa
from tqdm import tqdm

def process_batch(generator, batch_tensors):
    embeddings = []
    for tensor in batch_tensors:
        embedding = generator.generate_embedding(tensor)
        embeddings.append(embedding)
    return np.vstack(embeddings)

def process_audio_to_mel(audio, sr=22050, n_mels=80, n_fft=1024, hop_length=256,
                        win_length=1024, fmin=0.0, fmax=8000.0):
    return process_mel(audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
                      hop_length=hop_length, win_length=win_length,
                      fmin=fmin, fmax=fmax)

def search_2(input_folder, output_folder, model_path):
    config = Config()
    print(f"Using model configuration:")
    print(f"- Input shape: {config.input_shape}")
    print(f"- Embedding dimension: {config.embedding_dim}")
    print(f"- Batch size: {config.train_batch_size}")

    input_folder = os.path.join(input_folder, "processed")
    output_folder = os.path.join(output_folder, "embedding")
    os.makedirs(output_folder, exist_ok=True)

    print("Initializing embedding generator...")
    generator = EmbeddingGenerator(model_path)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]
    valid_files = [f for f in audio_files if os.path.exists(os.path.join(input_folder, f))]

    if not valid_files:
        print(f"No valid MP3 files found in {input_folder}")
        return

    print(f"Found {len(valid_files)} files to process")
    batch_tensors = []
    batch_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for audio_file in tqdm(valid_files, desc="Processing audio files"):
            try:
                audio_path = os.path.join(input_folder, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                spec_path = os.path.join(temp_dir, f"{base_name}.npy")
                embedding_path = os.path.join(output_folder, f"{base_name}_embedding.npy")

                audio, sr = librosa.load(audio_path, sr=22050)

                spec = process_audio_to_mel(audio, sr=sr)
                np.save(spec_path, spec)

                input_tensor = generator.load_and_preprocess(spec_path)
                batch_tensors.append(input_tensor)
                batch_files.append(embedding_path)

                if len(batch_tensors) >= config.train_batch_size:
                    print(f"\nProcessing batch of {len(batch_tensors)} files...")
                    embeddings = process_batch(generator, batch_tensors)
                    for emb, path in zip(embeddings, batch_files):
                        np.save(path, emb)
                    batch_tensors = []
                    batch_files = []

            except Exception as e:
                print(f"\nError processing {audio_file}: {str(e)}")
                continue

        if batch_tensors:
            print(f"\nProcessing final batch of {len(batch_tensors)} files...")
            embeddings = process_batch(generator, batch_tensors)
            for emb, path in zip(embeddings, batch_files):
                np.save(path, emb)

    print(f"\nProcessing completed! Embeddings saved to: {output_folder}")
