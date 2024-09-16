import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("Available physical devices:", physical_devices)

# Check for GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
print("Available GPU devices:", gpu_devices)
