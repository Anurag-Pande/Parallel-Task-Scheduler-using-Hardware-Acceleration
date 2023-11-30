import numpy as np
import numba
from numba import cuda
import time

# Define matrix multiplication functions

# Sequential Matrix Multiplication
def sequential_matmul(A, B):
    return np.dot(A, B)

# Parallel Matrix Multiplication using Numba CUDA
@cuda.jit
def parallel_matmul(A, B, result):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        result[i, j] = 0
        for k in range(A.shape[1]):
            result[i, j] += A[i, k] * B[k, j]

# Function to perform parallel matrix multiplication and measure execution time
def parallel_matmul_with_time(A, B):
    # Copy matrices to device
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    result_device = cuda.device_array((A.shape[0], B.shape[1]))

    # Configure grid and block dimensions
    threadsperblock = (16, 16)
    blockspergrid_x = (A.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (B.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Measure execution time
    start_time = time.time()

    # Launch the CUDA kernel
    parallel_matmul[blockspergrid, threadsperblock](A_device, B_device, result_device)
    cuda.synchronize()

    end_time = time.time()

    # Copy the result back to the host
    result = result_device.copy_to_host()

    return result, end_time - start_time

# Generate random matrices for testing
matrix_size = 1000
matrix_A = np.random.rand(matrix_size, matrix_size)
matrix_B = np.random.rand(matrix_size, matrix_size)

# Perform sequential matrix multiplication
sequential_result = sequential_matmul(matrix_A, matrix_B)

# Perform parallel matrix multiplication and measure execution time
parallel_result, parallel_execution_time = parallel_matmul_with_time(matrix_A, matrix_B)

# Verify correctness by comparing sequential and parallel results
np.testing.assert_allclose(sequential_result, parallel_result, rtol=1e-5)

# Print execution times
print(f"Sequential Execution Time: {parallel_execution_time:.6f} seconds")
print(f"Parallel Execution Time: {parallel_execution_time:.6f} seconds")
