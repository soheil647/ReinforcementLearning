import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# If CUDA is available, also print the CUDA version and the number of GPUs available
if torch.cuda.is_available():
    print("CUDA is available.")
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available, PyTorch is using CPU.")
