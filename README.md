# Experiment5-Neural-Network-CUDA
CUDA implementation of neural network layers - Cloud/GPU Computing Laboratory.

---

## 1. Aim

To implement and evaluate basic neural network layers using custom CUDA kernels:

- Matrix Multiplication with Bias
- ReLU Activation Function
- Sigmoid Activation Function

The experiment executes the same neural network forward pass on both CPU and GPU, validates numerical correctness, measures execution time, and calculates GPU speedup.

---

## 2. Technologies Used

- C++
- CUDA C/C++
- NVIDIA CUDA Toolkit 13.4
- Visual Studio Build Tools 2022
- Windows 11
- NVIDIA GeForce RTX 5060 Ti

---

## 3. Hardware Configuration

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| CUDA Toolkit | 13.4 |
| CUDA Compiler | NVCC |
| Operating System | Windows 11 |
| Compiler | Microsoft Visual C++ |

---

## 4. Network Configuration

The neural network uses the following configuration:

| Parameter | Value |
|---|---:|
| Batch Size | 256 |
| Input Size | 128 |
| Hidden Layer Size | 64 |
| Output Size | 10 |
| Number of Repetitions | 20 |

### Tensor Dimensions

| Tensor | Dimensions |
|---|---|
| Input X | 256 × 128 |
| Weight W1 | 128 × 64 |
| Bias b1 | 64 |
| Hidden Activation A1 | 256 × 64 |
| Weight W2 | 64 × 10 |
| Bias b2 | 10 |
| Final Output Y | 256 × 10 |

---

## 5. Neural Network Architecture

This experiment implements a two-layer feed-forward neural network using forward propagation.

### First Layer

Matrix multiplication with bias:

Z1 = X × W1 + b1

ReLU activation:

A1 = ReLU(Z1)

### Second Layer

Matrix multiplication with bias:

Z2 = A1 × W2 + b2

Sigmoid activation:

Y = Sigmoid(Z2)

### Activation Functions

**ReLU:**

ReLU(x) = max(0, x)

**Sigmoid:**

Sigmoid(x) = 1 / (1 + exp(-x))

The experiment performs inference only. It does not include training, loss calculation, backpropagation, or optimization.

---

## 6. CUDA Kernels Implemented

Three custom CUDA kernels were implemented.

### 1. Matrix Multiplication with Bias

Function: `matrixMulBiasKernel()`

- Performs matrix multiplication between two matrices.
- Adds the corresponding bias value to each output column.
- Each CUDA thread calculates one output matrix element.
- Uses block and grid dimensions for parallel execution.

### 2. ReLU Activation

Function: `reluKernel()`

- Applies the ReLU activation function.
- Converts negative values to zero.
- Preserves positive values.
- Each CUDA thread processes one element.

### 3. Sigmoid Activation

Function: `sigmoidKernel()`

- Applies the sigmoid activation function.
- Maps values into the range (0, 1).
- Each CUDA thread processes one element.

---

## 7. CPU Implementation

The CPU implementation uses standard C++ functions and nested loops.

The following operations are performed:

- Matrix multiplication with bias.
- ReLU activation.
- Sigmoid activation.
- Complete two-layer neural network forward pass.

The CPU implementation acts as the reference for validating GPU results.

---

## 8. GPU Implementation

The GPU implementation uses custom CUDA kernels.

The execution sequence is:

1. Copy input data and weights from host memory to device memory.
2. Execute matrix multiplication with bias.
3. Apply ReLU activation.
4. Execute the second matrix multiplication with bias.
5. Apply sigmoid activation.
6. Copy the final output from device memory to host memory.

CUDA kernels are launched sequentially on the default stream to maintain the forward-pass dependency.

---

## 9. Compilation and Execution

### Prerequisites

- NVIDIA CUDA-capable GPU.
- NVIDIA GPU driver.
- CUDA Toolkit.
- Visual Studio C++ Build Tools.
- MSVC compiler.

### Compile the Program

Open the Visual Studio Developer Command Prompt and navigate to the project directory.

Execute:

```bash
nvcc -O2 neural_network.cu -o neural_network.exe
