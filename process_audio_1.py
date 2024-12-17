import os
import librosa
import soundfile as sf
import numpy as np
import csv
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import subprocess

def get_center(audio_data, sr):
    total_duration = len(audio_data) / sr
    if total_duration > 45:
        middle_point = len(audio_data) // 2
        samples_per_45_sec = 45 * sr
        start_sample = middle_point - (samples_per_45_sec // 2)
        end_sample = start_sample + samples_per_45_sec
        return audio_data[start_sample:end_sample]
    return audio_data

def get_first_60s(audio_data, sr):
    samples_per_60_sec = 60 * sr
    if len(audio_data) > samples_per_60_sec:
        return audio_data[:samples_per_60_sec]
    return audio_data

def load_audio(file_path):
    try:
        audio_data, sr = sf.read(file_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
    except Exception:
        try:
            command = [
                'ffmpeg',
                '-i', file_path,
                '-f', 'f32le',
                '-acodec', 'pcm_f32le',
                '-ac', '1',
                '-ar', '48000',
                'pipe:'
            ]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            out, err = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {err.decode()}")

            audio_data = np.frombuffer(out, dtype=np.float32)
            sr = 48000

        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")

    return audio_data, sr

def is_valid_sound(audio_data, sr, min_dur=0.5, max_dur=None):
    dur = len(audio_data) / sr
    return min_dur < dur and (max_dur is None or dur < max_dur)

def trim_silence(audio_data, sr, top_db=60):
    return librosa.effects.trim(audio_data, top_db=top_db)[0]

def adjust_volume(audio_data, target_db=-20.0):
    rms = np.sqrt(np.mean(audio_data**2))
    current_db = 20 * np.log10(max(rms, 1e-10))
    adjustment = target_db - current_db
    return audio_data * (10 ** (adjustment / 20))

def save_audio(audio_data, sr, output_path):
    try:
        audio_int16 = (audio_data * 32767).astype(np.int16)

        command = [
            'ffmpeg',
            '-f', 's16le',  # input format
            '-ar', str(sr),  # input sample rate
            '-ac', '1',      # input channels
            '-i', 'pipe:',   # input from pipe
            '-c:a', 'libmp3lame',  # output codec
            '-y',            # overwrite output file
            output_path
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        out, err = process.communicate(input=audio_int16.tobytes())

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {err.decode()}")

        return True
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return False

def process_file(args):
    input_path, output_path, min_dur, max_dur, target_db = args
    try:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"{input_path} not found")

        audio_data, sr = load_audio(input_path)
        success = False

        if "/hum/" in input_path or "\\hum\\" in input_path:
            # Process middle 45s version
            middle_audio = get_center(audio_data, sr)
            middle_audio = adjust_volume(middle_audio, target_db)
            middle_audio = trim_silence(middle_audio, sr)

            if is_valid_sound(middle_audio, sr, min_dur, max_dur):
                middle_audio = librosa.util.normalize(middle_audio)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                success = save_audio(middle_audio, sr, output_path)

            # Process first 60s version
            first_60s_audio = get_first_60s(audio_data, sr)
            first_60s_audio = adjust_volume(first_60s_audio, target_db)
            first_60s_audio = trim_silence(first_60s_audio, sr)

            if is_valid_sound(first_60s_audio, sr, min_dur, max_dur):
                first_60s_audio = librosa.util.normalize(first_60s_audio)
                output_path_60s = output_path.replace('.mp3', '__2.mp3')
                os.makedirs(os.path.dirname(output_path_60s), exist_ok=True)
                success = save_audio(first_60s_audio, sr, output_path_60s) and success
        else:
            # Normal processing for song files
            audio_data = adjust_volume(audio_data, target_db)
            audio_data = trim_silence(audio_data, sr)

            if is_valid_sound(audio_data, sr, min_dur, max_dur):
                audio_data = librosa.util.normalize(audio_data)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                success = save_audio(audio_data, sr, output_path)

        return success

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_data(data_folder, output_folder, target_db=-20.0, num_workers=8):
    output_folder = os.path.join(output_folder, "output1")
    os.makedirs(output_folder, exist_ok=True)
    metadata_path = os.path.join(data_folder, "metadata.csv")
    output_metadata = []
    processing_args = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hum_file = os.path.splitext(row['hum'])[0] + ".mp3"
            song_file = os.path.splitext(row['song'])[0] + ".mp3"

            hum_input = os.path.join(data_folder, "hum", row['hum'])
            song_input = os.path.join(data_folder, "song", row['song'])
            hum_output = os.path.join(output_folder, "hum", hum_file)
            song_output = os.path.join(output_folder, "song", song_file)

            os.makedirs(os.path.dirname(hum_output), exist_ok=True)
            os.makedirs(os.path.dirname(song_output), exist_ok=True)

            processing_args.append((
                (hum_input, hum_output, 0.5, None, target_db),
                (song_input, song_output, 0.5, None, target_db),
                row
            ))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for (hum_args, song_args, row) in tqdm(processing_args, desc="Processing Files"):
            hum_future = executor.submit(process_file, hum_args)
            song_future = executor.submit(process_file, song_args)

            if hum_future.result() and song_future.result():
                output_metadata.append([row['id'],
                                     os.path.basename(hum_args[1]),
                                     os.path.basename(song_args[1])])
            else:
                print(f"Skipping {row['id']} due to processing failure.")

    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "hum", "song"])
        writer.writerows(output_metadata)

    print("Processing complete.")
