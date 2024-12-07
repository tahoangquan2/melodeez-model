import torch
import torchaudio
import torchaudio.transforms as T
import os
import random
from tqdm import tqdm
import csv

# If reproducibility is desired, uncomment and set a seed
# random.seed(42)
# torch.manual_seed(42)

def pitch_shift(audio, sr, shift_steps):
    return torchaudio.functional.pitch_shift(audio, sr, shift_steps)

def add_noise(audio, scale):
    noise = torch.randn_like(audio)
    return audio + noise * scale

def apply_gain(audio, gain_factor):
    return audio * gain_factor

def double_resample(audio, sr, new_sr):
    audio = torchaudio.functional.resample(audio, sr, new_sr)
    audio = torchaudio.functional.resample(audio, new_sr, sr)
    return audio

def aug_combination(audio, sr, path, min_shift=-4, max_shift=4,
                    min_snr=5, max_snr=20, min_gain_db=-10, max_gain_db=10,
                    min_sr=8000, max_sr=48000):
    effects = []

    shift_steps = random.uniform(min_shift, max_shift)
    effects.append(('pitch_shift', shift_steps))

    snr = random.uniform(min_snr, max_snr)
    # scale for add_noise
    scale = audio.norm(p=2) / (torch.randn_like(audio).norm(p=2) * (10 ** (snr/20.0)))
    effects.append(('add_noise', scale))

    gain_db = random.uniform(min_gain_db, max_gain_db)
    gain_factor = 10 ** (gain_db / 20.0)
    effects.append(('gain', gain_factor))

    new_sr = random.randint(min_sr, max_sr)
    effects.append(('resample', new_sr))

    random.shuffle(effects)
    num_effects = random.randint(2, 4)
    chosen_effects = effects[:num_effects]

    processed = audio
    for eff, param in chosen_effects:
        if eff == 'pitch_shift':
            processed = pitch_shift(processed, sr, param)
        elif eff == 'add_noise':
            processed = add_noise(processed, param)
        elif eff == 'gain':
            processed = apply_gain(processed, param)
        elif eff == 'resample':
            processed = double_resample(processed, sr, param)

    torchaudio.save(path, processed, sr, format='mp3')

def check_existing_augmentations(base_name, aug_path, tries):
    existing_augs = []
    for i in range(tries):
        aug_filename = f"{base_name}_aug{i}.mp3"
        if os.path.exists(os.path.join(aug_path, aug_filename)):
            existing_augs.append(aug_filename)
    return existing_augs

def process_data(data_folder, output_folder, tries=5):
    input_folder = os.path.join(data_folder, "output1")
    output_folder = os.path.join(output_folder, "output2")
    os.makedirs(output_folder, exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    output_metadata = []

    hum_out = os.path.join(output_folder, "hum")
    song_out = os.path.join(output_folder, "song")
    os.makedirs(hum_out, exist_ok=True)
    os.makedirs(song_out, exist_ok=True)

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
                    # Check existing augmentations first
                    existing_augs = check_existing_augmentations(base_name, aug_path, tries)
                    original_file = f"{base_name}.mp3"
                    original_path = os.path.join(aug_path, original_file)

                    # If all files exist (original + all augmentations), skip processing
                    if len(existing_augs) == tries and os.path.exists(original_path):
                        processed_files[sub].extend([original_file] + existing_augs)
                        print(f"Skipping {base_name} - all files exist")
                        continue

                    # Load audio only if we need to process something
                    audio, sr = torchaudio.load(input_path)

                    # Save original if it doesn't exist
                    if not os.path.exists(original_path):
                        torchaudio.save(original_path, audio, sr, format='mp3')
                        print(f"Saved original: {original_file}")
                    processed_files[sub].append(original_file)

                    # Generate only missing augmentations
                    for i in range(tries):
                        aug_filename = f"{base_name}_aug{i}.mp3"
                        aug_output = os.path.join(aug_path, aug_filename)

                        if not os.path.exists(aug_output):
                            aug_combination(audio, sr, aug_output)
                            print(f"Generated: {aug_filename}")
                        else:
                            print(f"Skipping existing: {aug_filename}")

                        processed_files[sub].append(aug_filename)

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                    continue

            # Only write to metadata if both hum and song have processed files
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
