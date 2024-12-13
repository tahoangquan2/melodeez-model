import os
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from process_audio_1 import load_audio, save_audio, process_file

def process_inference_step1(data_folder, output_folder, target_db=-20.0, num_workers=8):
    output_folder = os.path.join(output_folder, "output4")
    os.makedirs(output_folder, exist_ok=True)
    metadata_path = os.path.join(data_folder, "metadata.csv")
    output_metadata = []
    processing_args = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            song_file = os.path.splitext(row['song'])[0] + ".mp3"
            song_input = os.path.join(data_folder, "song", row['song'])
            song_output = os.path.join(output_folder, "song", song_file)

            os.makedirs(os.path.dirname(song_output), exist_ok=True)
            processing_args.append(((song_input, song_output, 0.5, None, target_db), row))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for (args, row) in tqdm(processing_args, desc="Processing Files"):
            future = executor.submit(process_file, args)
            if future.result():
                output_metadata.append([row['id'],
                                     os.path.basename(args[1]),
                                     row['info']])

    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "song", "info"])
        writer.writerows(output_metadata)

def split_audio(audio_path, output_folder, base_name, split_duration=60000, overlap_duration=30000):
    try:
        audio_data, sr = load_audio(audio_path)

        total_samples = len(audio_data)
        samples_per_split = int(split_duration * sr / 1000)
        overlap_samples = int(overlap_duration * sr / 1000)
        stride = samples_per_split - overlap_samples
        splits = []

        for i, start in enumerate(range(0, total_samples - overlap_samples, stride)):
            end = start + samples_per_split
            if end > total_samples:
                end = total_samples

            segment = audio_data[start:end]

            if len(segment) < overlap_samples:
                continue

            output_path = os.path.join(output_folder, f"{base_name}_split{i}.mp3")
            if save_audio(segment, sr, output_path):
                splits.append(f"{base_name}_split{i}.mp3")

        return splits
    except Exception as e:
        print(f"Error splitting {audio_path}: {e}")
        return []

def process_inference_step2(input_folder, output_folder):
    input_folder = os.path.join(input_folder, "output4")
    output_folder = os.path.join(output_folder, "output5")
    os.makedirs(os.path.join(output_folder, "song"), exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    output_metadata = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(list(reader), desc="Splitting Files"):
            input_path = os.path.join(input_folder, "song", row['song'])
            base_name = os.path.splitext(row['song'])[0]

            split_files = split_audio(
                input_path,
                os.path.join(output_folder, "song"),
                base_name
            )

            for split_file in split_files:
                output_metadata.append([row['id'], split_file, row['info']])

    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "song", "info"])
        writer.writerows(output_metadata)

def process_inference_step3(input_folder, output_folder):
    from process_audio_3 import process_file as process_mel_file

    input_folder = os.path.join(input_folder, "output5")
    output_folder = os.path.join(output_folder, "output6")
    os.makedirs(os.path.join(output_folder, "song"), exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    output_metadata = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        for row in tqdm(rows, desc="Processing Audio Files"):
            input_path = os.path.join(input_folder, "song", row['song'])
            output_path = os.path.join(output_folder, "song", f"{os.path.splitext(row['song'])[0]}.npy")

            if process_mel_file(input_path, output_path):
                output_metadata.append([row['id'],
                                     f"{os.path.splitext(row['song'])[0]}.npy",
                                     row['info']])

    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "song", "info"])
        writer.writerows(output_metadata)

def process_inference_data(data_folder, output_folder):
    print("Running step 1: Audio preprocessing...")
    process_inference_step1(data_folder, output_folder)

    print("Running step 2: Audio splitting...")
    process_inference_step2(output_folder, output_folder)

    print("Running step 3: Converting to mel spectrograms...")
    process_inference_step3(output_folder, output_folder)

    print("Inference preprocessing complete.")
