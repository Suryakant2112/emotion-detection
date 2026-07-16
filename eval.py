import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import MultimodalDataset
from model import MultimodalModel

EVALUATION_DIR = os.path.join(".", "Separated_Data", "Preprocessed_Data")

BEST_MODEL_PATH = "seed_iv_best_fold_3.pth"
BATCH_SIZE = 32

def get_all_file_paths(base_dir):
    file_paths = []
    emotions = ["0_Neutral", "1_Sad", "2_Fear", "3_Happy"]
    
    eeg_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if "EEG" in d]
    
    for eeg_folder in eeg_folders:
        prefix = eeg_folder.split("_EEG")[0]
        eye_folder = prefix + "_Eye_Tracking"
        
        if not os.path.exists(eye_folder):
            continue
            
        for label, emotion in enumerate(emotions):
            eeg_emo_dir = os.path.join(eeg_folder, emotion)
            eye_emo_dir = os.path.join(eye_folder, emotion)
            
            if os.path.exists(eeg_emo_dir) and os.path.exists(eye_emo_dir):
                for file_name in os.listdir(eeg_emo_dir):
                    if file_name.endswith('.npy'):
                        eeg_path = os.path.join(eeg_emo_dir, file_name)
                        eye_path = os.path.join(eye_emo_dir, file_name)
                        if os.path.exists(eye_path):
                            file_paths.append((eeg_path, eye_path, label))
    return file_paths

def evaluate_model():
    print(f"Locating Evaluation Data in: {EVALUATION_DIR}...\n")
    print("-" * 60)
    
    eval_files = get_all_file_paths(EVALUATION_DIR)
    print(f"Found {len(eval_files)} paired files for evaluation.\n")

    eval_dataset = MultimodalDataset(eval_files, augment=False)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}\n")

    print(f"Loading Trained Model Weights ({BEST_MODEL_PATH})...\n")
    
    model = MultimodalModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    test_loss = 0.0

    print("Generating Predictions...\n")
    
    with torch.no_grad(): 
        for eeg, eye, labels in eval_loader:
            eeg, eye, labels = eeg.to(device), eye.to(device), labels.to(device)
            
            outputs = model(eeg, eye)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(eval_loader)
    correct_preds = np.sum(np.array(all_preds) == np.array(all_labels))
    accuracy = 100 * correct_preds / len(all_labels)

    target_names = ["0_Neutral", "1_Sad", "2_Fear", "3_Happy"]

    print(f"\nEvaluation Loss: {avg_test_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.2f}% (Target Cumulative Baseline: 86.11%)\n")

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("=" * 60)
    
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    
    total_samples = np.sum(cm)
    per_class_accuracies = []
    for i in range(len(target_names)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - (tp + fp + fn)
        class_acc = (tp + tn) / total_samples
        per_class_accuracies.append(class_acc)
        
    metrics_data = []
    for i, emotion in enumerate(target_names):
        metrics_data.append({
            "Emotion": emotion,
            "Accuracy": per_class_accuracies[i],
            "Precision": report_dict[emotion]['precision'],
            "Recall": report_dict[emotion]['recall'],
            "F1-Score": report_dict[emotion]['f1-score'],
            "Support": int(report_dict[emotion]['support'])
        })
        
    overall_acc = report_dict['accuracy']
    macro_avg = report_dict['macro avg']
    metrics_data.append({
        "Emotion": "Cumulative/Overall (Target: 86.11%)",
        "Accuracy": overall_acc,
        "Precision": macro_avg['precision'],
        "Recall": macro_avg['recall'],
        "F1-Score": macro_avg['f1-score'],
        "Support": int(macro_avg['support'])
    })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    print("\n" + "=" * 80)
    print("DETAILED METRICS TABLE (Per Class & Cumulative)")
    print("=" * 80)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df_metrics.to_string(index=False))
    print("=" * 80)
    
    df_metrics.to_csv("detailed_evaluation_metrics.csv", index=False)
    print("Detailed metrics table saved to 'detailed_evaluation_metrics.csv'\n")

    print("\nGenerating Confusion Matrix Heatmap...")
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={"size": 16, "weight": "bold"}, 
                xticklabels=target_names, yticklabels=target_names)
    
    plt.title('Emotion Classification Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True/Actual Emotion', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_heatmap.png', dpi=300) 
    plt.close()
    
    print("Graph saved successfully! (confusion_matrix_heatmap.png)")

if __name__ == "__main__":
    evaluate_model()