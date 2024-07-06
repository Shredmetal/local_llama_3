import torch
print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Number of GPUs Available:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))