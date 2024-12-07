import torch
import torchaudio
import torchaudio.transforms as T
import os
import random
from tqdm import tqdm
import threading
import csv

def aug_pitch(audio, sr=16000, min_pitch=-4, max_pitch=4):
    n_steps = random.uniform(min_pitch, max_pitch)
    pitch_shifter = T.PitchShift(sr, n_steps)
    return pitch_shifter(audio)

def aug_speed(audio, min_speed=0.8, max_speed=1.2):
    speed_factor = random.uniform(min_speed, max_speed)
    time_stretcher = T.TimeStretch()
    return time_stretcher(audio, speed_factor)

def aug_timeshift(audio, max_shift=0.3):
    shift = int(random.uniform(0, max_shift) * audio.shape[1])
    return torch.roll(audio, shifts=shift, dims=1)

def aug_volume(audio, min_gain=-10, max_gain=10):
    gain = random.uniform(min_gain, max_gain)
    return audio * (10 ** (gain / 20.0))

def add_noise(audio, min_snr=5, max_snr=20):
    snr = random.uniform(min_snr, max_snr)
    noise = torch.randn_like(audio)
    signal_power = audio.norm(p=2)
    noise_power = noise.norm(p=2)
    scale = signal_power / (noise_power * (10 ** (snr/20.0)))
    return audio + noise * scale

def aug_combination(audio, sr=16000, path=None):
    effect_list = [aug_pitch, aug_speed, aug_timeshift, aug_volume, add_noise]
    num_effects = random.randint(2, len(effect_list))
    selected_effects = random.sample(effect_list, num_effects)

    augmented = audio
    for effect in selected_effects:
        if effect == aug_pitch:
            augmented = effect(augmented, sr=sr)
        else:
            augmented = effect(augmented)

    torchaudio.save(path, augmented, sr, format='mp3')

def process_data(data_folder, output_folder, tries=5):
    random.seed(1234)
    output_folder = os.path.join(output_folder, "output2")
    os.makedirs(output_folder, exist_ok=True)

    metadata_path = os.path.join(data_folder, "metadata.csv")
    output_metadata = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing Files"):
            subs = ["hum", "song"]
            processed_files = {"hum": [], "song": []}

            for sub in subs:
                input_file = row[sub]
                input_path = os.path.join(data_folder, sub, input_file)
                base_name = os.path.splitext(input_file)[0]

                try:
                    audio, sr = torchaudio.load(input_path)
                    aug_path = os.path.join(output_folder, sub)
                    os.makedirs(aug_path, exist_ok=True)

                    # Save original file
                    original_output = os.path.join(aug_path, base_name + ".mp3")
                    torchaudio.save(original_output, audio, sr, format='mp3')
                    processed_files[sub].append(base_name + ".mp3")

                    # Create augmented versions
                    threads = []
                    for i in range(tries):
                        aug_filename = f"{base_name}_aug{i}.mp3"
                        aug_output = os.path.join(aug_path, aug_filename)
                        t = threading.Thread(target=aug_combination,
                                          args=(audio, sr, aug_output))
                        threads.append(t)
                        t.start()
                        processed_files[sub].append(aug_filename)

                    for t in threads:
                        t.join()

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                    continue

            # Add to metadata for successfully processed files
            if processed_files["hum"] and processed_files["song"]:
                for hum_file in processed_files["hum"]:
                    for song_file in processed_files["song"]:
                        output_metadata.append([
                            row['id'], hum_file, song_file, row['info']
                        ])

    output_metadata_path = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "hum", "song", "info"])
        writer.writerows(output_metadata)

    print("Processing complete.")
