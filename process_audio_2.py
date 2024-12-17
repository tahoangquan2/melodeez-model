import librosa
import soundfile as sf
import numpy as np
import os
import random
from tqdm import tqdm
import csv

def pitch_shift(audio, sr, shift_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift_steps)

def add_noise(audio, scale):
    noise = np.random.randn(len(audio))
    return audio + noise * scale

def apply_gain(audio, gain_factor):
    return audio * gain_factor

def change_tempo(audio, sr, bpm_change):
    tempo_ratio = 1.0 + (bpm_change / 100.0)
    return librosa.effects.time_stretch(audio, rate=tempo_ratio)

def apply_effect(audio, sr, effect_type):
    if effect_type == 'pitch_shift':
        shift_steps = random.uniform(-3, 3)
        return pitch_shift(audio, sr, shift_steps)
    elif effect_type == 'noise':
        snr = random.uniform(15, 30)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        scale = np.sqrt(noise_power)
        return add_noise(audio, scale)
    elif effect_type == 'gain':
        gain_db = random.uniform(-4, 4)
        gain_factor = 10 ** (gain_db / 20.0)
        return apply_gain(audio, gain_factor)
    elif effect_type == 'tempo':
        bpm_change = random.uniform(-10, 10)
        return change_tempo(audio, sr, bpm_change)
    return audio

def apply_effects(audio, sr, effects_list):
    processed = audio.copy()
    for effect in effects_list:
        processed = apply_effect(processed, sr, effect)
    return processed

def generate_effect_combination():
    effects = ['pitch_shift', 'noise', 'gain', 'tempo']
    use_two_effects = random.random() < 0.5

    if use_two_effects:
        selected_effects = random.sample(effects, 2)
        effects_str = "+".join(selected_effects)
    else:
        selected_effects = [random.choice(effects)]
        effects_str = selected_effects[0]

    return selected_effects, effects_str

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
            processed_hums = []
            processed_songs = []

            # Process song file
            song_input = os.path.join(input_folder, "song", row['song'])
            song_base_name = os.path.splitext(row['song'])[0]
            song_output = os.path.join(song_out, f"{song_base_name}.mp3")

            try:
                if not os.path.exists(song_output):
                    audio, sr = librosa.load(song_input, sr=None)
                    sf.write(song_output, audio, sr, format='mp3')
                    print(f"Saved song: {song_base_name}.mp3")
                processed_songs.append(f"{song_base_name}.mp3")
            except Exception as e:
                print(f"Error processing song {song_input}: {e}")
                continue

            # Process both versions of hum files
            hum_input = os.path.join(input_folder, "hum", row['hum'])
            hum_base_name = os.path.splitext(row['hum'])[0]

            # Process middle 45s version
            try:
                existing_augs = check_existing_augmentations(hum_base_name, hum_out, tries)
                original_hum = f"{hum_base_name}.mp3"
                original_hum_path = os.path.join(hum_out, original_hum)

                if not os.path.exists(original_hum_path):
                    audio, sr = librosa.load(hum_input, sr=None)
                    sf.write(original_hum_path, audio, sr, format='mp3')
                    print(f"Saved original hum: {original_hum}")
                processed_hums.append(original_hum)

                for i in range(tries):
                    aug_filename = f"{hum_base_name}_aug{i}.mp3"
                    aug_output = os.path.join(hum_out, aug_filename)

                    if not os.path.exists(aug_output):
                        effects_list, effects_str = generate_effect_combination()
                        processed = apply_effects(audio, sr, effects_list)
                        processed = np.clip(processed, -1, 1)
                        sf.write(aug_output, processed, sr, format='mp3')
                        print(f"Generated augmentation with effects {effects_str}: {aug_filename}")

                    processed_hums.append(aug_filename)

                # Process first 60s version
                hum_input_60s = os.path.join(input_folder, "hum", f"{hum_base_name}__2.mp3")
                if os.path.exists(hum_input_60s):
                    original_hum_60s = f"{hum_base_name}__2.mp3"
                    original_hum_path_60s = os.path.join(hum_out, original_hum_60s)

                    if not os.path.exists(original_hum_path_60s):
                        audio_60s, sr = librosa.load(hum_input_60s, sr=None)
                        sf.write(original_hum_path_60s, audio_60s, sr, format='mp3')
                        print(f"Saved original hum (60s): {original_hum_60s}")
                    processed_hums.append(original_hum_60s)

                    for i in range(tries):
                        aug_filename_60s = f"{hum_base_name}__2_aug{i}.mp3"
                        aug_output_60s = os.path.join(hum_out, aug_filename_60s)

                        if not os.path.exists(aug_output_60s):
                            effects_list, effects_str = generate_effect_combination()
                            processed = apply_effects(audio_60s, sr, effects_list)
                            processed = np.clip(processed, -1, 1)
                            sf.write(aug_output_60s, processed, sr, format='mp3')
                            print(f"Generated augmentation with effects {effects_str}: {aug_filename_60s}")

                        processed_hums.append(aug_filename_60s)

            except Exception as e:
                print(f"Error processing hum {hum_input}: {e}")
                continue

            # Create metadata entries
            if processed_hums and processed_songs:
                for hum_file in processed_hums:
                    for song_file in processed_songs:
                        output_metadata.append([row['id'], hum_file, song_file])

    output_metadata_path = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "hum", "song"])
        writer.writerows(output_metadata)

    print("Processing complete.")
