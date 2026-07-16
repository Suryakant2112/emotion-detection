import os
import numpy as np

# Uses a relative path
BASE_DIR = os.path.join(".", "Separated_Data")
SPLITS = ["Train", "Test", "Validation"]
EMOTIONS = ["0_Neutral", "1_Sad", "2_Fear", "3_Happy"]

def run_checks():
    print("Starting Data Readability and Mapping Check...\n")
    all_matched_and_readable = True

    for split in SPLITS:
        for emotion in EMOTIONS:
            eeg_dir = os.path.join(BASE_DIR, f"{split}_EEG", emotion)
            eye_dir = os.path.join(BASE_DIR, f"{split}_Eye_Tracking", emotion)

            if not os.path.exists(eeg_dir) or not os.path.exists(eye_dir):
                continue

            eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith('.npy')]
            eye_files = [f for f in os.listdir(eye_dir) if f.endswith('.npy')]

            if set(eeg_files) == set(eye_files):
                print(f"[OK] Mapping match for {split} -> {emotion} ({len(eeg_files)} paired files)")
            else:
                print(f"[ERROR] Mismatch found in {split} -> {emotion}!")
                all_matched_and_readable = False

            if eeg_files and (eeg_files[0] in eye_files):
                sample_file = eeg_files[0]
                try:
                    np.load(os.path.join(eeg_dir, sample_file))
                    np.load(os.path.join(eye_dir, sample_file))
                except Exception as e:
                    print(f"  -> [ERROR] Failed to read sample files: {e}")
                    all_matched_and_readable = False
                    
    if all_matched_and_readable:
        print("\nSUCCESS: All folders are perfectly mapped and readable!")

if __name__ == "__main__":
    run_checks()