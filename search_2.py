import numpy as np
import os
import tempfile
from inference_1 import process_audio_to_mel
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

def search_2(input_folder, output_folder, model_path):
    config = Config()

    input_folder = os.path.join(input_folder, "processed")
    output_folder = os.path.join(output_folder, "embedding")
    os.makedirs(output_folder, exist_ok=True)

    generator = EmbeddingGenerator(model_path)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]
    valid_files = [f for f in audio_files if os.path.exists(os.path.join(input_folder, f))]

    if not valid_files:
        print(f"No valid MP3 files found in {input_folder}")
        return

    batch_tensors = []
    batch_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for audio_file in tqdm(valid_files):
            try:
                audio_path = os.path.join(input_folder, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                spec_path = os.path.join(temp_dir, f"{base_name}.npy")
                embedding_path = os.path.join(output_folder, f"{base_name}_embedding.npy")

                # Use standard sample rate of 22050
                audio, sr = librosa.load(audio_path, sr=22050)
                spec = process_audio_to_mel(audio, sr=sr)
                np.save(spec_path, spec)

                input_tensor = generator.load_and_preprocess(spec_path)
                batch_tensors.append(input_tensor)
                batch_files.append(embedding_path)

                if len(batch_tensors) >= config.train_batch_size:
                    embeddings = process_batch(generator, batch_tensors)
                    for emb, path in zip(embeddings, batch_files):
                        np.save(path, emb)
                    batch_tensors = []
                    batch_files = []

            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue

        if batch_tensors:
            embeddings = process_batch(generator, batch_tensors)
            for emb, path in zip(embeddings, batch_files):
                np.save(path, emb)

    print(f"Processing completed! Embeddings saved to: {output_folder}")
