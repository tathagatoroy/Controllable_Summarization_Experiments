import torch
import time
import matplotlib.pyplot as plt

def method1(A, B, C):
    return torch.mm(A, B + C)

def method2(A, B, C):
    return torch.mm(A, B) + torch.mm(A, C)

def benchmark(m, n, k, num_runs=20):
    A = torch.randn(m, k, device='cuda')
    B = torch.randn(k, n, device='cuda')
    C = torch.randn(k, n, device='cuda')

    # Warm-up run
    _ = method1(A, B, C)
    _ = method2(A, B, C)
    
    torch.cuda.synchronize()

    # Method 1: (B+C) then multiply by A
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = method1(A, B, C)
    torch.cuda.synchronize()
    time1 = (time.perf_counter() - start) / num_runs

    # Method 2: (AxB) and (AxC) separately, then add
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = method2(A, B, C)
    torch.cuda.synchronize()
    time2 = (time.perf_counter() - start) / num_runs

    return time1, time2

def run_benchmarks():
    sizes = [2**i for i in range(1, 16)]  # 2, 4, 8, ..., 1024
    times1 = []
    times2 = []

    for size in sizes:
        print(f"Benchmarking size {size}x{size}...")
        t1, t2 = benchmark(size, size, size)
        times1.append(t1)
        times2.append(t2)

    return sizes, times1, times2

def plot_results(sizes, times1, times2):
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, times1, label='Method 1: (B+C) then multiply by A')
    plt.plot(sizes, times2, label='Method 2: (AxB) and (AxC) separately, then add')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Matrix Size (n=m=k)')
    plt.ylabel('Average Computation Time (s)')
    plt.title('Performance Comparison of Matrix Multiplication Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig('matrix_mult_benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    torch.cuda.init()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    sizes, times1, times2 = run_benchmarks()
    
    plot_results(sizes, times1, times2)
    
    print("Benchmark complete. Results saved as 'matrix_mult_benchmark_results.png'")

    # Print the faster method for each size
    for size, t1, t2 in zip(sizes, times1, times2):
        faster = "Method 1" if t1 < t2 else "Method 2"
        print(f"Size {size}x{size}: {faster} is faster")