import os
import numpy as np

BASE_DIR = os.path.join(".", "Separated_Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Preprocessed_Data")

def z_score_normalize(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / (std_val + 1e-8)

def preprocess_dataset():
    print(f"Starting Preprocessing in: {BASE_DIR}\n")
    total_files_processed = 0

    for item in os.listdir(BASE_DIR):
        item_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(item_path) and item != "Preprocessed_Data":
            for emotion in os.listdir(item_path):
                emotion_path = os.path.join(item_path, emotion)
                if os.path.isdir(emotion_path):
                    out_dir = os.path.join(OUTPUT_DIR, item, emotion)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    for file_name in os.listdir(emotion_path):
                        if file_name.endswith('.npy'):
                            in_file = os.path.join(emotion_path, file_name)
                            out_file = os.path.join(out_dir, file_name)
                            try:
                                raw_data = np.load(in_file)
                                np.save(out_file, z_score_normalize(raw_data))
                                total_files_processed += 1
                            except Exception as e:
                                print(f"[ERROR] Failed on {in_file}: {e}")

    print(f"Preprocessing Complete! {total_files_processed} files saved.")

if __name__ == "__main__":
    preprocess_dataset()