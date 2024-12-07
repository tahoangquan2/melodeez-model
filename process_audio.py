import os
from pydub import AudioSegment, effects
import csv
from tqdm import tqdm

# Validate sound duration
def is_valid_sound(sound, min_dur=0.5, max_dur=None):
    dur = len(sound) / 1000
    return min_dur < dur and (max_dur is None or dur < max_dur)

# Trim silence
def trim_sil(sound):
    return effects.strip_silence(sound, silence_len=500, silence_thresh=-40)

# Process file
def process_file(input_path, output_path, audio_format="mp3", min_dur=0.5, max_dur=None):
    try:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"{input_path} not found")

        sound = AudioSegment.from_file(input_path, format=audio_format)
        trimmed_sound = trim_sil(sound)
        if not is_valid_sound(trimmed_sound, min_dur, max_dur):
            return False
        normalized_sound = effects.normalize(trimmed_sound)
        normalized_sound.export(output_path, format="mp3")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

# Main processing function
def process_data(data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    metadata_path = os.path.join(data_folder, "metadata.csv")
    output_metadata = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing Files"):
            hum_file = row['hum']
            song_file = row['song']
            song_info = row['info']

            hum_input = os.path.join(data_folder, "hum", hum_file)
            song_input = os.path.join(data_folder, "song", song_file)
            hum_output = os.path.join(output_folder, "hum", hum_file)
            song_output = os.path.join(output_folder, "song", song_file)

            os.makedirs(os.path.dirname(hum_output), exist_ok=True)
            os.makedirs(os.path.dirname(song_output), exist_ok=True)

            hum_processed = process_file(hum_input, hum_output, audio_format="m4a")
            song_processed = process_file(song_input, song_output, audio_format="mp3")

            if hum_processed and song_processed:
                output_metadata.append([row['id'], hum_file, song_file, song_info])
            else:
                print(f"Skipping {row['id']} due to processing failure.")

    # Save the updated metadata
    output_metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(output_metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "hum", "song", "info"])
        writer.writerows(output_metadata)
    print("Processing complete.")
