# --------------------------
# Check GPU stats: memory, throughput, etc.
# --------------------------

import torch
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("CUDA available:", torch.cuda.is_available())
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


def benchmark_gpu(matrix_size=4096, runs=10):
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU drivers or PyTorch installation.")
        return

    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    # create two large matrices on GPU
    a = torch.randn((matrix_size, matrix_size), device=device)
    b = torch.randn((matrix_size, matrix_size), device=device)

    # Warm-up
    print("Warming up...")
    for _ in range(5):
        torch.mm(a, b)

    # Timing
    print(f"Running benchmark with {runs} runs...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        torch.mm(a, b)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / runs
    print(f"Average time per matrix multiplication: {avg_time:.6f} seconds")
    print(f"Estimated throughput: {2 * (matrix_size ** 3) / (avg_time * 1e9):.2f} GFLOPS")

if __name__ == "__main__":
    benchmark_gpu()