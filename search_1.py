import os
from inference_1 import process_file
from pydub import AudioSegment

def search_1(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    print("Starting audio processing...")

    # Process each audio file in the input folder
    for audio_file in os.listdir(input_folder):
        try:
            input_path = os.path.join(input_folder, audio_file)

            # Skip non-files
            if not os.path.isfile(input_path):
                print(f"Skipping non-file: {audio_file}")
                continue

            # Convert to MP3 if needed
            base_name, ext = os.path.splitext(audio_file)
            output_mp3 = os.path.join(output_folder, f"{base_name}.mp3")

            if ext.lower() != ".mp3":
                try:
                    # Convert to MP3 using pydub
                    sound = AudioSegment.from_file(input_path, format=ext[1:])
                    sound.export(output_mp3, format="mp3")
                    print(f"Converted {audio_file} to MP3")
                except Exception as e:
                    print(f"Failed to convert {audio_file}: {e}")
                    continue
            else:
                # Directly copy MP3 files to output folder
                output_mp3 = input_path

            # Process the file as done in inference_1.py
            if process_file(output_mp3, output_mp3, audio_format="mp3"):
                print(f"Processed audio: {audio_file}")
            else:
                print(f"Failed to process: {audio_file}")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    print("Audio processing complete.")
