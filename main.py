from process_audio import process_data

if __name__ == "__main__":
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output"

    print(f"Processing data from '{DATA_FOLDER}' to '{OUTPUT_FOLDER}'...")
    process_data(DATA_FOLDER, OUTPUT_FOLDER)
