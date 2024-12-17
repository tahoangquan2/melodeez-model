from process_audio_1 import process_data as process_data_1
from process_audio_2 import process_data as process_data_2
from process_audio_3 import process_data as process_data_3
from inference_1 import process_inference_data
from inference_2 import process_inference_data as process_model_inference
from inference_3 import create_faiss_index
from train_model_2 import main as train_model
from search_1 import search_1
from search_2 import search_2
from search_3 import search_3
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def run_process_pipeline():
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"

    print("Running process_audio_1...")
    process_data_1(DATA_FOLDER, OUTPUT_FOLDER)

    print("Running process_audio_2...")
    process_data_2(OUTPUT_FOLDER, OUTPUT_FOLDER)

    print("Running process_audio_3...")
    process_data_3(OUTPUT_FOLDER, OUTPUT_FOLDER)

def run_inference_pipeline():
    INPUT_FOLDER = "song"
    OUTPUT_FOLDER = "output"
    MODEL_PATH = "checkpoints/resnetface_latest.pth"

    print("Starting inference preprocessing pipeline...")
    process_inference_data(INPUT_FOLDER, OUTPUT_FOLDER)

    print("Running model inference...")
    process_model_inference(OUTPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH)

    print("Creating FAISS index...")
    create_faiss_index(OUTPUT_FOLDER)

def run_search_pipeline():
    INPUT_FOLDER = "search"
    OUTPUT_FOLDER = "search"
    MODEL_PATH = "checkpoints/resnetface_latest.pth"

    print("Processing input audio files...")
    search_1(INPUT_FOLDER, OUTPUT_FOLDER)

    print("Creating embeddings...")
    search_2(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH)

    print("Performing similarity search...")
    search_3(INPUT_FOLDER, OUTPUT_FOLDER)

def run_train_model():
    print("Starting model training...")
    train_model()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "inference":
            run_inference_pipeline()
        elif sys.argv[1] == "search":
            run_search_pipeline()
        elif sys.argv[1] == "process":
            run_process_pipeline()
        elif sys.argv[1] == "train":
            run_train_model()
        else:
            print("Invalid argument. Use 'inference' or 'search' or 'process' or 'train'")
    else:
        print("No argument provided. Use 'inference' or 'search' or 'process' or 'train'")
