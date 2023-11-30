import tensorflow as tf

# Check if CUDA is available
if tf.test.is_gpu_available(cuda_only=False):
    print("CUDA is available!")
    # Your CUDA-accelerated code here
else:
    print("CUDA is not available. Check your setup.")