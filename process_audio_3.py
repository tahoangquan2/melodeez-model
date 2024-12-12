import os
import numpy as np
import torch
import torchaudio
import librosa
import joblib
import csv
import random
from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn

def process_audio(audio, sr=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=1024,
                 fmin=0.0, fmax=8000.0):
    waveform = torch.FloatTensor(audio)
    mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_basis).float()

    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        return_complex=True
    )
    spec = torch.abs(spec)
    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel.numpy()

def process_file(audio_path, out_path, sr=22050):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        spec = process_audio(audio, sr=sr)
        np.save(out_path, spec)
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

def process_data(data_folder, output_folder):
    random.seed(1234)
    test_ratio = 0.2

    input_dir = os.path.join(data_folder, "output2")
    output_dir = os.path.join(output_folder, "output3")
    os.makedirs(output_dir, exist_ok=True)

    for sub in ["hum", "song"]:
        print(f"Processing {sub} files...")
        input_path = os.path.join(input_dir, sub)
        output_path = os.path.join(output_dir, sub)

        if not os.path.isdir(input_path):
            continue

        os.makedirs(output_path, exist_ok=True)
        files = [f for f in os.listdir(input_path) if f.endswith('.mp3')]

        jobs = [joblib.delayed(process_file)(
            os.path.join(input_path, f),
            os.path.join(output_path, f"{f[:-4]}.npy")
        ) for f in files]

        joblib.Parallel(n_jobs=4, verbose=1)(jobs)

    if os.path.exists(os.path.join(input_dir, "metadata.csv")):
        input_metadata = os.path.join(input_dir, "metadata.csv")
        output_metadata = os.path.join(output_dir, "metadata.csv")

        all_ids = set()
        rows = []
        with open(input_metadata, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_ids.add(row['id'])
                rows.append(row)

        test_ids = set(random.sample(list(all_ids), k=int(len(all_ids) * test_ratio)))

        with open(output_metadata, 'w', newline='') as f:
            fieldnames = ["id", "hum", "song", "info", "testing"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                row["testing"] = "test" if row["id"] in test_ids else "train"
                writer.writerow(row)

    print("Processing complete.")
