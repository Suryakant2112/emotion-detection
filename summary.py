import torch
from torchinfo import summary
from model import MultimodalModel

def generate_summary():
    print("Loading Final Asymmetric Spatial-Temporal Model...\n")
    
    # Ensure the dummy data and model are on the same device to prevent crashes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating summary on device: {device}\n")
    
    # 1. Initialize your model and push to device
    model = MultimodalModel().to(device)
    
    # 2. Define the exact shapes of your dummy inputs
    # EEG format: (Batch Size, Channels, Scalp Electrodes, Time Steps)
    dummy_eeg = torch.randn(32, 5, 62, 64).to(device)
    
    # Eye format: (Batch Size, Channels, Eye Features, Time Steps)
    dummy_eye = torch.randn(32, 1, 31, 64).to(device)
    
    # 3. Generate and print the Keras-style summary
    print("="*90)
    print("      FINAL MULTIMODAL MODEL SUMMARY (CROSS-ATTENTION & ASYMMETRIC CONVS)")
    print("="*90)
    
    # We pass the dummy data as a tuple exactly how the forward() function expects it
    summary(
        model, 
        input_data=(dummy_eeg, dummy_eye),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        col_width=20,
        depth=4 # Increased depth to 4 to reveal the inner workings of the new architecture
    )

if __name__ == "__main__":
    generate_summary()