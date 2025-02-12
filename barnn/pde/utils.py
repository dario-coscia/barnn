import torch
import os
import math


def positional_embedding(timesteps: torch.Tensor, dim, max_period=1000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def check_directory(directory):
    """
    Check if log directory exists within experiments.
    """
    assert isinstance(directory, str)
    if not os.path.exists(directory):
        os.mkdir(directory)

def create_directories(experiment_name, seed, pde, model):
    directory_path = os.path.join(experiment_name, pde, model, f"log_{str(seed)}")
    os.makedirs(directory_path, exist_ok=True)
    return f'{experiment_name}/{pde}/{model}/log_{str(seed)}/'

def read_results(directory, avg=True):
    print('RESULTS')
    print('===========')
    # List all files in the directory
    files = os.listdir(directory)
    # Filter out the .pt files
    pt_files = [f for f in files if f.endswith('.pt')]
    # Extract extensions (file names without the .pt part)
    for pt_file in pt_files:
        # Split the file name by the dot and remove the last part (.pt)
        extension = os.path.splitext(pt_file)[0]
        # Print the result
        if avg:
            print(f'{extension} : {torch.load(directory + "/" + pt_file).as_subclass(torch.Tensor).mean()}')
        else:
            print(f'{extension} : {torch.load(directory + "/" + pt_file).as_subclass(torch.Tensor)}')
    print()