---
title: Very Naive Neural Network in C++ Part 2 Activation functions
tags:
  - c++
  - deep-learning
  - machine-learning
  - neural-network
categories:
  - Code
date: 2021-01-05 19:00:01
---

Now we've created the `Matrix`, we should build the layer that contains the neurons. But before that we need the activation functions. I'm gonna create a class that contains some static activation functions and their derivative forms (relu, sigmoid, tanh and softmax). These activation functions will take the input matrices and output the  matrices after applying the corresponding functions.

<!-- more -->

```cpp
#include "Matrix.h"  
#include <cmath>  

//Let's put everything to an enum for better organization
enum ActivationFunctionType {  
    RELU,  
    SIGMOID,  
    TANH,  
    SOFTMAX,  
    NONE  
};  
  
class ActivationFunction {  
public:  
	//The activation and derivative methods will serve as proxies to call the required function types
    static Matrix activation(const Matrix& input, ActivationFunctionType activationFunctionType);  
    static Matrix derivative(const Matrix& input, ActivationFunctionType activationFunctionType);  
  
private:  
    static Matrix reluActivation(Matrix input);  
    static Matrix reluDerivative(Matrix input);  
    static Matrix sigmoidActivation(Matrix input);  
    static Matrix sigmoidDerivative(Matrix input);  
    static Matrix tanhActivation(Matrix input);  
    static Matrix tanhDerivative(Matrix input);  
    static Matrix softmaxActivation(Matrix input);  
    static Matrix softmaxDerivative(Matrix input);  
};
```


```cpp
#include "ActivationFunction.h"  
  
Matrix ActivationFunction::activation(const Matrix& input, ActivationFunctionType activationFunctionType) {  
//    std::cout << "ActivationFunction::activation" << std::endl;  
    switch (activationFunctionType) {  
        case RELU:  
            return ActivationFunction::reluActivation(input);  
        case SIGMOID:  
            return ActivationFunction::sigmoidActivation(input);  
        case TANH:  
            return ActivationFunction::tanhActivation(input);  
        case SOFTMAX:  
            return ActivationFunction::softmaxActivation(input);  
        default:  
            return input;  
    }  
}  
  
Matrix ActivationFunction::derivative(const Matrix& input, ActivationFunctionType activationFunctionType) {  
    switch (activationFunctionType) {  
        case RELU:  
            return ActivationFunction::reluDerivative(input);  
        case SIGMOID:  
            return ActivationFunction::sigmoidDerivative(input);  
        case TANH:  
            return ActivationFunction::tanhDerivative(input);  
        case SOFTMAX:  
            return ActivationFunction::softmaxDerivative(input);  
        default:  
            return input;  
    }  
}  
```
# Relu
```cpp 
Matrix ActivationFunction::reluActivation(Matrix input) {  
    Matrix output = input;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            if (input(i, j) < 0) {  
                output(i, j) = 0;  
            }  
        }  
    }  
    return output;  
}  
  
Matrix ActivationFunction::reluDerivative(Matrix input) {  
    Matrix output = input;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            if (input(i, j) < 0) {  
                output(i, j) = 0;  
            } else {  
                output(i, j) = 1;  
            }  
        }  
    }  
    return output;  
}  
```

# Softmax
```cpp  
Matrix ActivationFunction::softmaxActivation(Matrix input) {  
    Matrix output = input;  
    double sum = 0;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            sum += exp(input(i, j));  
        }  
    }  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            output(i, j) = exp(input(i, j)) / sum;  
        }  
    }  
    return output;  
}  
  
Matrix ActivationFunction::softmaxDerivative(Matrix input) {  
    Matrix output = input;  
    double sum = 0;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            sum += exp(input(i, j));  
        }  
    }  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            output(i, j) = exp(input(i, j)) / sum * (1 - exp(input(i, j)) / sum);  
        }  
    }  
    return output;  
}  
  ```
# Sigmoid 
```
Matrix ActivationFunction::sigmoidActivation(Matrix input) {  
    Matrix output = input;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            output(i, j) = 1 / (1 + exp(-input(i, j)));  
        }  
    }  
    return output;  
}  
  
Matrix ActivationFunction::sigmoidDerivative(Matrix input) {  
    Matrix output = input;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            output(i, j) = exp(-input(i, j)) / pow(1 + exp(-input(i, j)), 2);  
        }  
    }  
    return output;  
}  
```

# Tanh
```  
Matrix ActivationFunction::tanhActivation(Matrix input) {  
    Matrix output = input;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            output(i, j) = (exp(input(i, j)) - exp(-input(i, j))) / (exp(input(i, j)) + exp(-input(i, j)));  
        }  
    }  
    return output;  
}  
  
Matrix ActivationFunction::tanhDerivative(Matrix input) {  
    Matrix output = input;  
    for (int i = 0; i < input.rows(); i++) {  
        for (int j = 0; j < input.cols(); j++) {  
            output(i, j) = 1 - pow((exp(input(i, j)) - exp(-input(i, j))) / (exp(input(i, j)) + exp(-input(i, j))), 2);  
        }  
    }  
    return output;  
}
```