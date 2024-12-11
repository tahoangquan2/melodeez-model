import torch
import torchaudio
import torchaudio.transforms as T
import os
import random
from tqdm import tqdm
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def pitch_shift(audio, sr, shift_steps):
    return torchaudio.functional.pitch_shift(audio.cpu(), sr, shift_steps).to(device)

def add_noise(audio, scale):
    noise = torch.randn_like(audio, device=audio.device)
    return audio + noise * scale

def apply_gain(audio, gain_factor):
    return audio * gain_factor

def get_augmentation_params():
    effects = []

    shift_steps = random.uniform(-2, 2)
    effects.append(('pitch_shift', shift_steps))

    scale = random.uniform(0.01, 0.05)
    effects.append(('add_noise', scale))

    gain_db = random.uniform(-3, 3)
    gain_factor = 10 ** (gain_db / 20.0)
    effects.append(('gain', gain_factor))

    chosen_effect = random.choice(effects)
    return chosen_effect

def aug_combination(audio, sr, path, aug_params):
    processed = audio
    effect_type, param = aug_params

    if effect_type == 'pitch_shift':
        processed = pitch_shift(processed, sr, param)
    elif effect_type == 'add_noise':
        processed = add_noise(processed, param)
    elif effect_type == 'gain':
        processed = apply_gain(processed, param)

    torchaudio.save(path, processed.cpu(), sr, format='mp3')

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
            try:
                hum_input = os.path.join(input_folder, "hum", row['hum'])
                song_input = os.path.join(input_folder, "song", row['song'])

                hum_base_name = os.path.splitext(row['hum'])[0]
                song_base_name = os.path.splitext(row['song'])[0]

                hum_audio, hum_sr = torchaudio.load(hum_input)
                song_audio, song_sr = torchaudio.load(song_input)
                hum_audio = hum_audio.to(device)
                song_audio = song_audio.to(device)

                hum_original = f"{hum_base_name}.mp3"
                song_original = f"{song_base_name}.mp3"
                hum_original_path = os.path.join(hum_out, hum_original)
                song_original_path = os.path.join(song_out, song_original)

                if not os.path.exists(hum_original_path):
                    torchaudio.save(hum_original_path, hum_audio.cpu(), hum_sr, format='mp3')
                if not os.path.exists(song_original_path):
                    torchaudio.save(song_original_path, song_audio.cpu(), song_sr, format='mp3')

                processed_files = []
                processed_files.append((hum_original, song_original))

                for i in range(tries):
                    hum_aug = f"{hum_base_name}_aug{i}.mp3"
                    song_aug = f"{song_base_name}_aug{i}.mp3"
                    hum_aug_path = os.path.join(hum_out, hum_aug)
                    song_aug_path = os.path.join(song_out, song_aug)

                    if not os.path.exists(hum_aug_path) or not os.path.exists(song_aug_path):
                        aug_params = get_augmentation_params()
                        aug_combination(hum_audio, hum_sr, hum_aug_path, aug_params)
                        aug_combination(song_audio, song_sr, song_aug_path, aug_params)
                        print(f"Generated augmentation {i} for {row['id']}")
                    else:
                        print(f"Skipping existing augmentation {i} for {row['id']}")

                    processed_files.append((hum_aug, song_aug))

                # Add to metadata
                for hum_file, song_file in processed_files:
                    output_metadata.append([row['id'], hum_file, song_file, row['info']])

            except Exception as e:
                print(f"Error processing {row['id']}: {e}")
                continue

    output_metadata_path = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "hum", "song", "info"])
        writer.writerows(output_metadata)

    print("Processing complete.")
