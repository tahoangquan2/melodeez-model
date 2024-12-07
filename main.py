from process_audio_1 import process_data as process_data_1
from process_audio_2 import process_data as process_data_2

if __name__ == "__main__":
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"

    print("Running process_audio_1...")
    process_data_1(DATA_FOLDER, OUTPUT_FOLDER)

    print("Running process_audio_2...")
    process_data_2(DATA_FOLDER, OUTPUT_FOLDER)
