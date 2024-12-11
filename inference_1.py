import os
from pydub import AudioSegment, effects
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import torchaudio
import librosa
import joblib
from librosa.filters import mel as librosa_mel_fn

def is_valid_sound(sound, min_dur=0.5, max_dur=None):
    dur = len(sound) / 1000
    return min_dur < dur and (max_dur is None or dur < max_dur)

def trim_sil(sound):
    return effects.strip_silence(sound, silence_len=500, silence_thresh=-40)

def adjust_volume(sound, target_dBFS=-20.0):
    difference = target_dBFS - sound.dBFS
    return sound.apply_gain(difference)

def process_file(input_path, output_path, audio_format="mp3", min_dur=0.5, max_dur=None, target_dBFS=-20.0):
    try:
        if not os.path.isfile(input_path):
            print(f"{input_path} not found")
            return False

        sound = AudioSegment.from_file(input_path, format=audio_format)
        volume_adjusted_sound = adjust_volume(sound, target_dBFS)
        trimmed_sound = trim_sil(volume_adjusted_sound)
        if not is_valid_sound(trimmed_sound, min_dur, max_dur):
            return False
        normalized_sound = effects.normalize(trimmed_sound)
        normalized_sound.export(output_path, format="mp3")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_row_inference(row, data_folder, output_folder, target_dBFS):
    song_file = os.path.splitext(row['song'])[0] + ".mp3"
    song_info = row['info']

    song_input = os.path.join(data_folder, "song", row['song'])
    song_output = os.path.join(output_folder, "song", song_file)

    song_processed = process_file(song_input, song_output, audio_format="mp3", target_dBFS=target_dBFS)

    if song_processed:
        return [row['id'], song_file, song_info]
    return None

def process_inference_step1(data_folder, output_folder, target_dBFS=-20.0):
    output_folder = os.path.join(output_folder, "output4")
    os.makedirs(os.path.join(output_folder, "song"), exist_ok=True)

    metadata_path = os.path.join(data_folder, "metadata.csv")

    output_metadata = []
    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_row_inference, row, data_folder, output_folder, target_dBFS)
                  for row in rows]
        for future in tqdm(futures, desc="Processing Files"):
            result = future.result()
            if result is not None:
                output_metadata.append(result)

    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "song", "info"])
        writer.writerows(output_metadata)
    print("Step 1 processing complete.")

def split_audio(audio_path, output_folder, base_name, split_duration=60000, overlap_duration=30000):
    try:
        audio = AudioSegment.from_file(audio_path, format="mp3")
        total_duration = len(audio)
        splits = []

        start_points = range(0, total_duration - overlap_duration, split_duration - overlap_duration)

        for i, start in enumerate(start_points):
            end = start + split_duration
            if end > total_duration:
                end = total_duration

            segment = audio[start:end]
            if len(segment) < overlap_duration:
                continue

            output_path = os.path.join(output_folder, f"{base_name}_split{i}.mp3")
            segment.export(output_path, format="mp3")
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

    print("Step 2 processing complete.")

def process_audio_to_mel(audio, sr=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=1024,
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

def process_audio_file(audio_path, out_path, sr=22050):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        spec = process_audio_to_mel(audio, sr=sr)
        np.save(out_path, spec)
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

def process_inference_step3(input_folder, output_folder):
    input_folder = os.path.join(input_folder, "output5")
    output_folder = os.path.join(output_folder, "output6")
    os.makedirs(os.path.join(output_folder, "song"), exist_ok=True)

    metadata_path = os.path.join(input_folder, "metadata.csv")
    output_metadata = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        # Process audio files to mel spectrograms
        input_files = []
        output_files = []
        for row in rows:
            input_path = os.path.join(input_folder, "song", row['song'])
            output_path = os.path.join(output_folder, "song", f"{os.path.splitext(row['song'])[0]}.npy")
            input_files.append(input_path)
            output_files.append(output_path)
            output_metadata.append([row['id'], f"{os.path.splitext(row['song'])[0]}.npy", row['info']])

        # Process files in parallel
        jobs = [
            joblib.delayed(process_audio_file)(input_path, output_path)
            for input_path, output_path in zip(input_files, output_files)
        ]
        joblib.Parallel(n_jobs=4, verbose=1)(jobs)

    # Write metadata
    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "song", "info"])
        writer.writerows(output_metadata)

    print("Step 3 processing complete.")

def process_inference_data(data_folder, output_folder):
    print("Running step 1: Audio preprocessing...")
    process_inference_step1(data_folder, output_folder)

    print("Running step 2: Audio splitting...")
    process_inference_step2(output_folder, output_folder)

    print("Running step 3: Converting to mel spectrograms...")
    process_inference_step3(output_folder, output_folder)

    print("Inference preprocessing complete.")
