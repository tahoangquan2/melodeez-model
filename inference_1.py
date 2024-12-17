import os
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from process_audio_1 import process_file

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

            # Skip if output file already exists
            if os.path.exists(song_output):
                output_metadata.append([row['id'],
                                     os.path.basename(song_output),
                                     row['info']])
                continue

            os.makedirs(os.path.dirname(song_output), exist_ok=True)
            processing_args.append(((song_input, song_output, 0.5, None, target_db), row))

    if processing_args:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for (args, row) in tqdm(processing_args, desc="Processing Files"):
                future = executor.submit(process_file, args)
                if future.result():
                    output_metadata.append([row['id'],
                                         os.path.basename(args[1]),
                                         row['info']])
    else:
        print("No new files to process in step 1")

    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "song", "info"])
        writer.writerows(output_metadata)

def process_inference_step3(input_folder, output_folder):
    from process_audio_3 import process_file as process_mel_file

    input_folder = os.path.join(input_folder, "output4")
    output_folder = os.path.join(output_folder, "output6")
    os.makedirs(os.path.join(output_folder, "song"), exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    output_metadata = []
    files_to_process = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        for row in rows:
            input_path = os.path.join(input_folder, "song", row['song'])
            output_path = os.path.join(output_folder, "song", f"{os.path.splitext(row['song'])[0]}.npy")

            # Skip if output file already exists
            if os.path.exists(output_path):
                output_metadata.append([row['id'],
                                     f"{os.path.splitext(row['song'])[0]}.npy",
                                     row['info']])
                continue

            files_to_process.append((row, input_path, output_path))

    if files_to_process:
        for row, input_path, output_path in tqdm(files_to_process, desc="Processing Audio Files"):
            if process_mel_file(input_path, output_path):
                output_metadata.append([row['id'],
                                     f"{os.path.splitext(row['song'])[0]}.npy",
                                     row['info']])
    else:
        print("No new files to process in step 3")

    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "song", "info"])
        writer.writerows(output_metadata)

def process_inference_data(data_folder, output_folder):
    print("Running step 1: Audio preprocessing...")
    process_inference_step1(data_folder, output_folder)

    print("Running step 2: Converting to mel spectrograms...")
    process_inference_step3(output_folder, output_folder)

    print("Inference preprocessing complete.")
