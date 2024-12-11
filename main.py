from process_audio_1 import process_data as process_data_1
from process_audio_2 import process_data as process_data_2
from process_audio_3 import process_data as process_data_3
from inference_1 import process_inference_data
from inference_2 import process_inference_data as process_model_inference
from train_model_2 import main as train_model_1
import os

def ensure_directories_exist():
    directories = ["data", "output"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_training_pipeline():
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"

    print("Running process_audio_1...")
    process_data_1(DATA_FOLDER, OUTPUT_FOLDER)

    print("Running process_audio_2...")
    process_data_2(OUTPUT_FOLDER, OUTPUT_FOLDER)

    print("Running process_audio_3...")
    process_data_3(OUTPUT_FOLDER, OUTPUT_FOLDER)

    print("Starting model training...")
    train_model_1()

def run_inference_pipeline():
    INPUT_FOLDER = "song"
    OUTPUT_FOLDER = "output"
    MODEL_PATH = "checkpoints/resnet18_best.pth"

    print("Starting inference preprocessing pipeline...")
    # process_inference_data(INPUT_FOLDER, OUTPUT_FOLDER)

    print("Running model inference...")
    process_model_inference(OUTPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH)

    print("Inference pipeline complete.")

if __name__ == "__main__":
    import sys
    ensure_directories_exist()

    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        run_inference_pipeline()
    else:
        run_training_pipeline()
