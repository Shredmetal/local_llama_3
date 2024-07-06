# LLaMA Experiment

This project implements an interactive AI system using the LLaMA 3 8B model.

## Setup Instructions

### 1. Python Virtual Environment

1. Create a new virtual environment. You should use an IDE, but if you are a masochist, please go ahead and do it in terminal with:

`python3 -m venv .venv`

2. Activate the virtual environment:

`source .venv/bin/activate`

If you are a normal human being, you'll probably have this in PyCharm and your venv should be automatically activated.

### 2. CUDA and PyTorch Setup

1. Install PyTorch with CUDA support:

`pip install torch`

2. If PyTorch is using CUDA, add the following to your virtual environment's `activate` script:

```
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Deactivate your virtual environment and reactivate again. 

3. Verify PyTorch CUDA setup by running the GPU troubleshooting script:

`python src/scripts/gpu_troubleshoot.py`

This script contains:

```
import torch
print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Number of GPUs Available:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
```

4. Note that the system-wide CUDA installation (checked with `nvcc --version`) may differ from the CUDA version used by PyTorch. This is normal and shouldn't affect PyTorch's functionality.


### 3. Environment Variables

You can check whether your environment variables are properly set with:

```
echo $CUDA_HOME
echo $PATH
echo $LD_LIBRARY_PATH
```

### 4. Dependencies Installation

Install the required packages:

`pip install torch transformers`

### 5. Model Download

The LLaMA 3 8B model should be downloaded and placed in the `Meta-Llama-3-8B-Instruct/original/` directory. Ensure you have the necessary permissions to access and use this model (request permission from Meta).

I recommend using the huggingface CLI. Spin up your terminal, and type this:

`pip install -U "huggingface_hub[cli]"`

That should install the huggingface cli. Go to your huggingface Profile > Settings > Access Tokens, and generate an access token. Copy the token, and stash it somewhere.

Remember to permit read access under the Repos section of token permissions.

Then run:

`huggingface-cli login`

You will be prompted for the token. Paste it in, and you should have access to your huggingface account, together with gated repos which you have been granted access, like Llama.

Then run:

`huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir Meta-Llama-3-8B-Instruct`

### 6. Running the Project

To run the interactive AI system:

Hit the run button in PyCharm like a normal person on src/core/interact.py. Or if you are a mutant:

`python src/core/interact.py`


## Troubleshooting

If you encounter GPU-related issues, run the GPU troubleshooting script:

`python src/scripts/gpu_troubleshoot.py`

This script will provide information about your GPU setup and help diagnose any problems.

## Notes

- Ensure your GPU drivers are up-to-date.
- Remember to deactivate the virtual environment when you're done:

`deactivate`
