import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import librosa
from inference_1 import process_file as process_audio_file
from process_audio_3 import process_audio as process_mel

def process_audio_to_mel(audio_path, sr=22050):
    try:
        audio, _ = librosa.load(audio_path, sr=sr)

        spec = process_mel(audio, sr=sr)

        return spec
    except Exception as e:
        print(f"Error converting audio to mel: {e}")
        return None

def process_single_file(file_data):
    input_path, output_folder = file_data
    try:
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_mp3 = os.path.join(output_folder, f"{name}.mp3")

        if process_audio_file((input_path, output_mp3, 0.5, None, -20.0)):
            return True, base_name
        return False, f"Failed to process {base_name}"
    except Exception as e:
        return False, f"Error processing {base_name}: {str(e)}"

def search_1(input_folder, output_folder):
    processed_folder = os.path.join(output_folder, "processed")
    os.makedirs(processed_folder, exist_ok=True)

    print("Starting audio processing...")

    files_to_process = []
    supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
    for audio_file in os.listdir(input_folder):
        if os.path.splitext(audio_file)[1].lower() in supported_formats:
            input_path = os.path.join(input_folder, audio_file)
            if os.path.isfile(input_path):
                files_to_process.append((input_path, processed_folder))

    if not files_to_process:
        print("No supported audio files found in input folder")
        return

    successful_files = []
    for file_data in files_to_process:
        success, result = process_single_file(file_data)
        if success:
            print(f"Successfully processed: {result}")
            successful_files.append(result)
        else:
            print(f"Error: {result}")

    if successful_files:
        print(f"Completed processing {len(successful_files)} files")
    else:
        print("No files were successfully processed")

    print("Audio processing complete.")
