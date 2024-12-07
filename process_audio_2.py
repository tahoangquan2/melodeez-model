import torch
import torchaudio
import torchaudio.transforms as T
import os
import random
from tqdm import tqdm
import csv

def pitch_shift(audio, sr=16000, min_shift=-4, max_shift=4):
    shift_steps = random.uniform(min_shift, max_shift)
    return torchaudio.functional.pitch_shift(audio, sr, shift_steps)

def add_noise(audio, min_snr=5, max_snr=20):
    snr = random.uniform(min_snr, max_snr)
    noise = torch.randn_like(audio)
    scale = audio.norm(p=2) / (noise.norm(p=2) * (10 ** (snr/20.0)))
    return audio + noise * scale

def gain(audio, min_gain=-10, max_gain=10):
    gain_db = random.uniform(min_gain, max_gain)
    return audio * (10 ** (gain_db / 20.0))

def resample(audio, sr, min_sr=8000, max_sr=48000):
    new_sr = random.randint(min_sr, max_sr)
    audio = torchaudio.functional.resample(audio, sr, new_sr)
    return torchaudio.functional.resample(audio, new_sr, sr)

def aug_combination(audio, sr=16000, path=None):
    effects = [pitch_shift, add_noise, gain, resample]
    random.shuffle(effects)
    num_effects = random.randint(2, 4)
    processed = audio.clone()

    for effect in effects[:num_effects]:
        processed = effect(processed, sr) if effect in [pitch_shift, resample] else effect(processed)

    torchaudio.save(path, processed, sr, format='mp3')

def process_data(data_folder, output_folder, tries=5):
    input_folder = os.path.join(data_folder, "output1")
    output_folder = os.path.join(output_folder, "output2")
    os.makedirs(output_folder, exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    output_metadata = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing Files"):
            subs = ["hum", "song"]
            processed_files = {sub: [] for sub in subs}

            for sub in subs:
                input_file = row[sub]
                input_path = os.path.join(input_folder, sub, input_file)
                base_name = os.path.splitext(input_file)[0]

                try:
                    audio, sr = torchaudio.load(input_path)
                    aug_path = os.path.join(output_folder, sub)
                    os.makedirs(aug_path, exist_ok=True)

                    original_output = os.path.join(aug_path, f"{base_name}.mp3")
                    torchaudio.save(original_output, audio, sr, format='mp3')
                    processed_files[sub].append(f"{base_name}.mp3")

                    for i in range(tries):
                        aug_filename = f"{base_name}_aug{i}.mp3"
                        aug_output = os.path.join(aug_path, aug_filename)
                        aug_combination(audio, sr, aug_output)
                        processed_files[sub].append(aug_filename)

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

            if processed_files["hum"] and processed_files["song"]:
                for hum_file in processed_files["hum"]:
                    for song_file in processed_files["song"]:
                        output_metadata.append([row['id'], hum_file, song_file, row['info']])

    output_metadata_path = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "hum", "song", "info"])
        writer.writerows(output_metadata)

    print("Processing complete.")
