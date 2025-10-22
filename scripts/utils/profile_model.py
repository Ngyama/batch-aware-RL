# ================================================================
# Phase 1: Setup
# ================================================================
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
import os
import glob
import numpy as np
import json

print("[Phase 1] Setup")
# Set the device to use for computation (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  - Using device: {device}")

# Load the pre-trained ResNet-18 model
print("  - Loading pre-trained ResNet-18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()      # Set the model to evaluation mode
model.to(device)  # Move the model to the selected device
print(f"  - Model loaded and moved to {device}.")
print("-" * 50)


# ================================================================
# Phase 2: Data Processing
# ================================================================
print("[Phase 2] Data Processing")
# Define the standard image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("  - Preprocessing pipeline defined.")

# Point to the local dataset directory
dataset_path = os.path.join("data", "imagenette2")
val_dataset_path = os.path.join(dataset_path, 'val')
if not os.path.exists(val_dataset_path):
    raise FileNotFoundError(
        f"Error: Validation set path not found at {val_dataset_path}. "
        f"Please ensure the dataset is in the 'data' subfolder."
    )

# Use ImageFolder to load all images from the validation set
val_dataset = ImageFolder(root=val_dataset_path, transform=preprocess)
print(f"  - Validation set loaded: {len(val_dataset)} images found across {len(val_dataset.classes)} classes.")
print("-" * 50)


# ================================================================
# Phase 3 & 4: Core Profiling and Finalization
# ================================================================
print("[Phase 3-4] Starting Performance Profiling (Latency only)...")
batch_sizes_to_test = [1, 2, 4, 8, 16, 32, 64]
results = {}
print(f"  - Will test for Batch Sizes: {batch_sizes_to_test}")

with torch.no_grad():
    for batch_size in batch_sizes_to_test:
        # Create a DataLoader for the current batch size
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        latencies = []

        # Warm-up phase for the GPU
        if device.type == 'cuda':
            # Get the first batch for warm-up
            warmup_inputs, _ = next(iter(val_loader))
            warmup_inputs = warmup_inputs.to(device)
            for _ in range(10):
                _ = model(warmup_inputs)

        # Iterate through all batches in the DataLoader to get a representative average
        for inputs, _ in val_loader: # We ignore the labels with '_'
            inputs = inputs.to(device)

            # --- Timing Logic ---
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = model(inputs) # Perform inference

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)

        # Calculate the average latency for this batch size
        avg_batch_latency = np.mean(latencies)

        # Store the result
        results[batch_size] = {
            "avg_latency_s": avg_batch_latency
        }
        
        print(f"  - Batch Size: {batch_size:<3} | "
              f"Average latency per batch: {avg_batch_latency * 1000:.4f} ms")

print("\n--- Profiling Complete! ---")
print("Final results (batch_size: {avg_latency_in_seconds}):")
print(json.dumps(results, indent=4))