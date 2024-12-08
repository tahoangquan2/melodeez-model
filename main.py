from process_audio_1 import process_data as process_data_1
from process_audio_2 import process_data as process_data_2
from process_audio_3 import process_data as process_data_3

from train_model_2 import main as train_model_1

if __name__ == "__main__":
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"

    print("Running process_audio_1...")
    # process_data_1(DATA_FOLDER, OUTPUT_FOLDER)

    print("Running process_audio_2...")
    # process_data_2(OUTPUT_FOLDER, OUTPUT_FOLDER)

    print("Running process_audio_3...")
    # process_data_3(OUTPUT_FOLDER, OUTPUT_FOLDER)

    print("Starting model training...")
    train_model_1()
