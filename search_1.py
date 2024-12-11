import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from inference_1 import process_file
from pydub import AudioSegment

def process_single_file(file_data):
    input_path, output_folder = file_data
    try:
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_mp3 = os.path.join(output_folder, f"{name}.mp3")

        if ext.lower() == '.mp3':
            if input_path != output_mp3:
                AudioSegment.from_mp3(input_path).export(output_mp3, format="mp3")
        else:
            sound = AudioSegment.from_file(input_path, format=ext[1:])
            sound.export(output_mp3, format="mp3")

        if process_file(output_mp3, output_mp3, audio_format="mp3"):
            return True, base_name
        return False, f"Failed to process {base_name}"
    except Exception as e:
        return False, f"Error processing {base_name}: {str(e)}"

def search_1(input_folder, output_folder):
    processed_folder = os.path.join(output_folder, "processed")
    os.makedirs(processed_folder, exist_ok=True)

    print("Starting parallel audio processing...")

    files_to_process = []
    for audio_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, audio_file)
        if os.path.isfile(input_path):
            files_to_process.append((input_path, processed_folder))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, file_data)
                  for file_data in files_to_process]

        for future in as_completed(futures):
            success, result = future.result()
            if success:
                print(f"Processed: {result}")
            else:
                print(result)

    print("Audio processing complete.")
