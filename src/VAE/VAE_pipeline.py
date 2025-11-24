import torch
import numpy as np
from VAE_model import VAE


# Assume 'input_vector' is your numpy array of size 1024
# Example: input_vector = np.random.rand(1024)
input_vector = np.random.rand(1024) # Just for this example

# --- 0. Device Selection Logic ---
# Check if CUDA (NVIDIA GPU) is available, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# IMPORTANT: Ensure your model is on the correct device
# 'model' is your VAE instance
model = VAE(num_genes=1024, latent_dim=128)  
model.to(device)
model.eval() # Set model to evaluation mode (disables dropout, etc.)

# --- Data Preparation ---

# 1. Convert NumPy array to PyTorch Tensor
input_tensor = torch.from_numpy(input_vector)

# 2. Force float32 type (NumPy is often float64 by default, PyTorch expects float32)
input_tensor = input_tensor.float()

# 3. Add batch dimension if missing (changes shape from 1024 to 1x1024)
if len(input_tensor.shape) == 1:
    input_tensor = input_tensor.unsqueeze(0)  # Adds a dimension at index 0

# 4. Send the tensor to the selected device (GPU or CPU)
input_tensor = input_tensor.to(device)

# --- Inference ---

# 5. Encode the vector
# We use 'no_grad()' because we don't need to calculate gradients for inference (saves memory)
with torch.no_grad():
    latent = model(input_tensor, return_latent=True)

print("Latent shape:", latent.shape)