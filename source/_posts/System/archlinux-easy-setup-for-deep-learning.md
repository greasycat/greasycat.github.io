---
title: archlinux-easy-setup-for-deep-learning
categories:
  - System
date: 2022-11-19 02:13:52
tags: 
- archlinux
- installation
- deep-learning
---

# My setup
- CPU: AMD Ryzen 9 5900X
- GPU: NVIDIA GeForce RTX 3090 FE
- Archlinux 6.0.8-arch1-1
- WM i3

# Update pacman source
```bash
sudo pacman -Syu
```

# Nvidia driver

**Skip** if you've installed properitary nvidia driver
- for those need to switch nouveu to nvidia, please check archlinux wiki for more information

```bash
sudo pacman -S nvidia nvidia-utils
# or nvidia-dkms and linux-header for custom kernel
```

# CUDA & CUDNN
```bash
sudo pacman -S cuda cudnn
```

If you havn't set the path for dynamic/shared library, it's time to set it. Otherwise errors like `# Unimplemented: DNN library is not found` might present when running CNN


Add the following line to either `~/.bashrc` or `~/.bash_profile` or `.profile` (last two are preferred)
```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib"
```

Verify the installation
```bash
cd /opt/cuda/extras/demo_suite
./deviceQuery
```
<!-- more -->
```
 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3090"
  CUDA Driver Version / Runtime Version          11.8 / 11.8
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 24265 MBytes (25443893248 bytes)
  (82) Multiprocessors, (128) CUDA Cores/MP:     10496 CUDA Cores
  GPU Max Clock rate:                            1695 MHz (1.70 GHz)
  Memory Clock rate:                             9751 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 6291456 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 9 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.8, CUDA Runtime Version = 11.8, NumDevs = 1, Device0 = NVIDIA GeForce RTX 3090
Result = PASS
```

# Tensorflow & Keras

`python-tensorflow-opt-cuda` vs `python-tensorflow-cuda`
- opt might be slightly faster as the package is optimized for certain intel cpu

```bash
sudo pacman -S python-tensorflow-cuda keras
```

# Try it out!
```python
from keras import layers  
from keras import models  
model = models.Sequential()  
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  
model.add(layers.MaxPooling2D((2, 2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  
model.add(layers.MaxPooling2D((2, 2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  
model.add(layers.Flatten())  
model.add(layers.Dense(64, activation='relu'))  
model.add(layers.Dense(10, activation='softmax'))  
print(model.summary())  
  
from keras.datasets import mnist  
from keras.utils import to_categorical  
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  
  
train_images = train_images.reshape((60000, 28, 28, 1))  
train_images = train_images.astype('float32') / 255  
test_images = test_images.reshape((10000, 28, 28, 1))  
test_images = test_images.astype('float32') / 255  
train_labels = to_categorical(train_labels)  
test_labels = to_categorical(test_labels)  
model.compile(optimizer='rmsprop',  
	loss='categorical_crossentropy',  
	metrics=['accuracy'])  
model.fit(train_images, train_labels, epochs=5, batch_size=64)  
  
test_loss, test_acc = model.evaluate(test_images, test_labels)  
print('test_acc:', test_acc)
```

```
2022-11-19 02:48:54.199427: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-11-19 02:48:54.280223: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2022-11-19 02:48:55.103112: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.112502: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.112652: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.112918: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-11-19 02:48:55.114464: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.114573: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.114662: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.310342: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.310471: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.310579: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:980] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-11-19 02:48:55.310662: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21342 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:09:00.0, compute capability: 8.6
2022-11-19 02:48:55.310982: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 64)          36928     
                                                                 
 flatten (Flatten)           (None, 576)               0         
                                                                 
 dense (Dense)               (None, 64)                36928     
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/5
2022-11-19 02:48:56.553880: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8600
2022-11-19 02:48:56.923119: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
938/938 [==============================] - 3s 2ms/step - loss: 0.1702 - accuracy: 0.9469
Epoch 2/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0460 - accuracy: 0.9853
Epoch 3/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0321 - accuracy: 0.9902
Epoch 4/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0254 - accuracy: 0.9919
Epoch 5/5
938/938 [==============================] - 2s 2ms/step - loss: 0.0204 - accuracy: 0.9940
313/313 [==============================] - 0s 970us/step - loss: 0.0403 - accuracy: 0.9880
test_acc: 0.9879999756813049
```