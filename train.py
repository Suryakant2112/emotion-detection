import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from model import MultimodalModel
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PREPROCESSED_DIR = os.path.join(".", "Separated_Data", "Preprocessed_Data")
BATCH_SIZE = 32
EPOCHS = 40 
STARTING_LR = 1e-4

def mixup_data(eeg, eye, y, alpha=0.4, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = eeg.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_eeg = lam * eeg + (1 - lam) * eeg[index, :]
    mixed_eye = lam * eye + (1 - lam) * eye[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_eeg, mixed_eye, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

def start_training():
    print("Locating all preprocessed files for 100% Data K-Fold...")
    all_files = get_all_file_paths(PREPROCESSED_DIR)
    print(f"Found {len(all_files)} total paired files ready for K-Fold Split.\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}\n")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    all_folds_histories = [] 
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        print(f"\n{'='*60}")
        print(f"               STARTING FOLD {fold + 1} / 5")
        print(f"{'='*60}")
        
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        train_dataset = MultimodalDataset(train_files, augment=True)
        val_dataset = MultimodalDataset(val_files, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = MultimodalModel().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=STARTING_LR, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
        
        best_val_acc = 0.0
        current_fold_history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for eeg, eye, labels in train_loader:
                eeg, eye, labels = eeg.to(device), eye.to(device), labels.to(device)
                optimizer.zero_grad()
                
                mixed_eeg, mixed_eye, targets_a, targets_b, lam = mixup_data(eeg, eye, labels, alpha=0.4, device=device)
                
                outputs = model(mixed_eeg, mixed_eye)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (lam * (predicted == targets_a).float() + (1 - lam) * (predicted == targets_b).float()).sum().item()

            train_acc = 100 * correct / total
            train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for eeg, eye, labels in val_loader:
                    eeg, eye, labels = eeg.to(device), eye.to(device), labels.to(device)
                    outputs = model(eeg, eye)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            current_fold_history['train_acc'].append(train_acc)
            current_fold_history['val_acc'].append(val_acc)
            current_fold_history['train_loss'].append(train_loss)
            current_fold_history['val_loss'].append(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Fold {fold+1} | Epoch {epoch+1}/{EPOCHS} [LR: {current_lr:.6f}] - loss: {train_loss:.4f} - acc: {train_acc:.2f}% - val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%")

            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"seed_iv_best_fold_{fold+1}.pth")
        
        print(f"\n--> Fold {fold + 1} Completed! Best Validation Accuracy: {best_val_acc:.2f}%")
        fold_results.append(best_val_acc)
        all_folds_histories.append(current_fold_history)
    
    print("\n" + "="*50)
    print("      5-FOLD CROSS-VALIDATION COMPLETE")
    print("="*50)
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1} Best Accuracy: {acc:.2f}%")
    
    final_average = np.mean(fold_results)
    print(f"\nAVERAGE ACCURACY ACROSS ALL 5 FOLDS: {final_average:.2f}%")
    print("="*50)
    
    print("\nGenerating Presentation Graphs...")
    
    plt.figure(figsize=(7, 6))
    sns.boxplot(y=fold_results, color='lightblue', width=0.4)
    sns.stripplot(y=fold_results, color='darkblue', size=10, jitter=0.1)
    plt.axhline(y=final_average, color='r', linestyle='--', linewidth=2, label=f'Mean Accuracy: {final_average:.2f}%')
    plt.title('5-Fold Cross-Validation Robustness', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.xticks([0], ['All 5 Folds'], fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('kfold_variance_boxplot.png', dpi=300)
    plt.close()
    
    for i, history in enumerate(all_folds_histories):
        epochs_ran = range(1, len(history['train_acc']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs_ran, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs_ran, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'Model Accuracy (Fold {i + 1})', fontsize=14)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        ax2.plot(epochs_ran, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs_ran, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title(f'Model Loss (Fold {i + 1})', fontsize=14)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f'learning_curves_fold_{i+1}.png', dpi=300)
        plt.close()
    
    print("Graphs saved successfully!")

if __name__ == "__main__":
    start_training()