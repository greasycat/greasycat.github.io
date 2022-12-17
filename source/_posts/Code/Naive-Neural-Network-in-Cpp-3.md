---
title: Very Naive Neural Network in C++ Part 3 Layers
tags:
  - c++
  - deep-learning
  - machine-learning
  - neural-network
categories:
  - Code
date: 2021-01-05 19:15:34
---
Okay, now we have matrices and the activation functions, it's time to build up the layers.
For simplicity and purpose of the project, I will not gives the layers more abstraction by making it an interface and builds the subsequent layer above that. For now, I will put activation and non-activation layers together.

<!-- more -->

```cpp
#include "../Math/ActivationFunction.h"  
#include <random>  
#include <memory>  
  
class Layer {  
public:  
    // Constructor for input layer, which has no weights and biases  
    explicit Layer(int inputSize, int outputSize);  
    explicit Layer(int inputSize, int outputSize, Matrix weights, Matrix biases);  
    explicit Layer(ActivationFunctionType activationFunctionType);  

	//Forward propagation
    Matrix forwardPropagation(const Matrix& inputData);  
  
    // Backward propagation  
    Matrix backwardPropagation(const Matrix& error, double learningRate);  
  
    // Getters  
    Matrix getWeights();  
    Matrix getBiases();  
    Matrix getInput();  
  
    int getInputSize() const;  
    int getOutputSize() const;  
  
    bool isActivationLayer() const;  
    void setActivationType(ActivationFunctionType activationType);  
  
  
private:  
    Matrix weights;  
    Matrix biases;  
    ActivationFunctionType activationFunctionType;  
    Matrix input;  
    int inputSize;  
    int outputSize;  
  
    ActivationFunctionType getActivationType(ActivationFunctionType activationType);  
};
```
# Initialization of weights and bias
So here we will generate the weights and bias randomly by a normal distribution
The initial values are actually crucial to our naive design

```cpp
#include "Layer.h"  
Layer::Layer(ActivationFunctionType activationFunctionType) : weights(Matrix::EmptyMatrix()),  
                                                              biases(Matrix::EmptyMatrix()),  
                                                              input(Matrix::EmptyMatrix()), inputSize(0), outputSize(0),  
                                                              activationFunctionType(activationFunctionType) {}  
  
Layer::Layer(int inputSize, int outputSize) : weights(inputSize, outputSize),  
                                              biases(1, outputSize),  
                                              input(Matrix::EmptyMatrix()),  
                                              inputSize(inputSize),  
                                              outputSize(outputSize),  
                                              activationFunctionType(ActivationFunctionType::NONE) {  
    weights = Matrix(inputSize, outputSize);  
  
    // Set up random number generator  
    std::random_device rd;  
    std::mt19937 gen(rd());  
    std::normal_distribution<double> weights_distribution(inputSize, outputSize);  
    std::normal_distribution<double> bias_distribution(1, outputSize);  
  
    // Initialize weights and biases with random values from normal distribution  
    for (int i = 0; i < inputSize; i++) {  
        for (int j = 0; j < outputSize; j++) {  
            this->weights.changeAt(i, j, weights_distribution(gen)/ sqrt(inputSize+outputSize));  
        }  
    }  
    for (int i = 0; i < outputSize; i++) {  
        this->biases.changeAt(0, i, bias_distribution(gen)/ sqrt(inputSize+outputSize));  
    }  
    this->weights.print();  
    this->biases.print();  
}  
```
# Forward propagation
- For non-activation layer, we will simply sum up the product of each x and weights and plus the bias. So the information will be changed and transformed into the dimension of the inputSize times the output size
$$Y=XW+B$$
- For activation layer, we just apply the activation function and return the tensor to next layer
```cpp  
Matrix Layer::forwardPropagation(const Matrix &inputData) {  
  
    this->input = inputData;  
    if (this->activationFunctionType == ActivationFunctionType::NONE) {  
        return this->biases + this->input.dot(this->weights);  
    } else {  
        Matrix results = ActivationFunction::activation(inputData, activationFunctionType);  
        return results;  
    }  
  
}  

```

# Backward propagation
The clear part of the neural network is to find a set of weights for each layer that minimize the error, and the most common way is to find the gradient and move the weights and biases bits by bits based on the gradients.

We have to use chain rule to get the information from the previous layer (assume has `n` nodes)
$$
\frac{\partial E}{\partial W} = \frac{\partial E}{\partial w_{ij}} = \sum_{k = 1}^{n} \frac{\partial E}{\partial k_j} \frac{\partial y_k}{\partial w_{ij}} =\sum_{k = 1}^{n} \frac{\partial E}{\partial y_j} x_{i}=X^t \frac{\partial E}{\partial Y}
$$

$$
\frac{\partial E}{\partial B_i} = \sum_{j = 1}^{n} \frac{\partial E}{\partial y_j} \frac{\partial y_j}{\partial b_i} =\sum_{j = 1}^{n} \frac{\partial E}{\partial y_j} = \frac{\partial E}{\partial Y}
$$

$$
\frac{\partial E}{\partial X_i} = \sum_{j = 1}^{n} \frac{\partial E}{\partial y_j} \frac{\partial y_j}{\partial x_i} =\sum_{j = 1}^{n} \frac{\partial E}{\partial y_j} w_{ij}
$$

$$
\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y}W^t
$$

```cpp
// error is dE/DY
Matrix Layer::backwardPropagation(const Matrix &error, double learningRate) {  
  
    // If the layer is not the activation layer  
    if (this->activationFunctionType == ActivationFunctionType::NONE) {  
		// dE/dX
        Matrix previousError = error.dot(this->weights.transpose());  
  
        // Calculate gradient of weights and biases  
        // dE/dW
        Matrix gradientOfWeights = this->input.transpose().dot(error);  
  
        // dE/dB, just for better understanding  
        const Matrix &gradientOfBiases = error;  
  
        // Update weights and biases  
        this->weights -= gradientOfWeights * learningRate;  
        this->biases -= gradientOfBiases * learningRate;  
        return previousError;  
    } else {  
        return ActivationFunction::derivative(this->input, this->activationFunctionType) * error;  
    }  
  
}  

```
# Other methods
```cpp
Matrix Layer::getWeights() {  
    return this->weights;  
}  
  
Matrix Layer::getBiases() {  
    return this->biases;  
}  
  
Matrix Layer::getInput() {  
    return this->input;  
}  
  
int Layer::getInputSize() const {  
    return inputSize;  
}  
  
int Layer::getOutputSize() const {  
    return outputSize;  
}  
  
bool Layer::isActivationLayer() const {  
    return this->activationFunctionType != ActivationFunctionType::NONE;  
}  
  
void Layer::setActivationType(ActivationFunctionType activationType) {  
    this->activationFunctionType = activationType;  
}  
  
ActivationFunctionType Layer::getActivationType(ActivationFunctionType activationType) {  
    return this->activationFunctionType;  
}  
  
Layer::Layer(int inputSize, int outputSize, Matrix weights, Matrix biases):  
        weights(weights), biases(biases), input(Matrix::EmptyMatrix()), inputSize(inputSize), outputSize(outputSize),  
        activationFunctionType(ActivationFunctionType::NONE) {}
```