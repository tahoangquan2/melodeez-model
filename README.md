# Audio Processing and Search System

---

## **Installation and Setup**

### **Step 1: Install Requirements**

1. Ensure Python 3.10+ is installed.

2. Install the required libraries:

   ```bash
   pip install torch torchaudio librosa numpy pydub tqdm joblib faiss
   ```

3. Check if CUDA is available (for GPU acceleration):

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

   If `True`, GPU support is enabled.

---

### **Step 2: Ensure Folder Structure**

Before running the pipeline, verify that the following folders exist:

- **For Preprocessing:**

  ```
  data/
    hum/
    song/
    metadata.csv
  ```

- **For Searching:**

  ```
  search/
  ```

Make sure to have one audio file in folder "search".

---

## **Usage**

### **Step 1: Preprocessing**

Run:

```bash
python main.py preprocess
```

This processes and augments the audio data, saving outputs to `output/output1/`, `output/output2/`, and `output/output3/`.

---

### **Step 2: Inference**

Run:

```bash
python main.py inference
```

This processes audio data, generates embeddings, and builds the FAISS index saving outputs to `output/output4/`, `output/output5/`, `output/output6/`, `output/output7/` and `output/output8/`.

Once you have the output8 folder, delete the other 7 output folders to avoid wasting memory.

---

### **Step 3: Search**

Run:

```bash
python main.py search
```

This processes queries, generates embeddings, and searches using the FAISS index. Results are saved in `search/results/search_results.json`.

---
