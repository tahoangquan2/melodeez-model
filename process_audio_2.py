import librosa
import soundfile as sf
import numpy as np
import os
import random
from tqdm import tqdm
import csv
from scipy import signal

def pitch_shift(audio, sr, shift_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift_steps)

def add_noise(audio, scale):
    noise = np.random.randn(len(audio))
    return audio + noise * scale

def apply_gain(audio, gain_factor):
    return audio * gain_factor

def apply_single_effect(audio, sr, effect_type, path):
    if effect_type == 'pitch_shift':
        shift_steps = random.uniform(-3, 3)
        processed = pitch_shift(audio, sr, shift_steps)
    elif effect_type == 'noise':
        snr = random.uniform(15, 30)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        scale = np.sqrt(noise_power)
        processed = add_noise(audio, scale)
    elif effect_type == 'gain':
        gain_db = random.uniform(-4, 4)
        gain_factor = 10 ** (gain_db / 20.0)
        processed = apply_gain(audio, gain_factor)

    processed = np.clip(processed, -1, 1)
    sf.write(path, processed, sr, format='mp3')

def check_existing_augmentations(base_name, aug_path, tries):
    existing_augs = []
    for i in range(tries):
        aug_filename = f"{base_name}_aug{i}.mp3"
        if os.path.exists(os.path.join(aug_path, aug_filename)):
            existing_augs.append(aug_filename)
    return existing_augs

def process_data(data_folder, output_folder, tries=5):
    """Main processing function"""
    input_folder = os.path.join(data_folder, "output1")
    output_folder = os.path.join(output_folder, "output2")
    os.makedirs(output_folder, exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    output_metadata = []

    hum_out = os.path.join(output_folder, "hum")
    song_out = os.path.join(output_folder, "song")
    os.makedirs(hum_out, exist_ok=True)
    os.makedirs(song_out, exist_ok=True)

    effects = ['pitch_shift', 'noise', 'gain']

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing Files"):
            subs = ["hum", "song"]
            processed_files = {sub: [] for sub in subs}

            for sub in subs:
                input_file = row[sub]
                input_path = os.path.join(input_folder, sub, input_file)
                base_name = os.path.splitext(input_file)[0]
                aug_path = os.path.join(output_folder, sub)

                try:
                    existing_augs = check_existing_augmentations(base_name, aug_path, tries)
                    original_file = f"{base_name}.mp3"
                    original_path = os.path.join(aug_path, original_file)

                    if len(existing_augs) == tries and os.path.exists(original_path):
                        processed_files[sub].extend([original_file] + existing_augs)
                        print(f"Skipping {base_name} - all files exist")
                        continue

                    audio, sr = librosa.load(input_path, sr=None)

                    if not os.path.exists(original_path):
                        sf.write(original_path, audio, sr, format='mp3')
                        print(f"Saved original: {original_file}")
                    processed_files[sub].append(original_file)

                    augmentation_effects = effects * 2
                    random.shuffle(augmentation_effects)

                    for i in range(tries):
                        aug_filename = f"{base_name}_aug{i}.mp3"
                        aug_output = os.path.join(aug_path, aug_filename)

                        if not os.path.exists(aug_output):
                            effect = augmentation_effects[i]
                            apply_single_effect(audio, sr, effect, aug_output)
                            print(f"Generated {effect} augmentation: {aug_filename}")
                        else:
                            print(f"Skipping existing: {aug_filename}")

                        processed_files[sub].append(aug_filename)

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                    continue

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
