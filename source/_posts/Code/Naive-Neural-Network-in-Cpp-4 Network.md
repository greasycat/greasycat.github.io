---
title: Very Naive Neural Network in C++ Part 4 Network
tags:
  - c++
  - deep-learning
  - machine-learning
  - neural-network
categories:
  - Code
date: 2021-01-05 19:32:23
---
Ok this is the final bit. We need the network to wrap things up and do the training

But again, we need some loss functions to determine the errors we gonna put into back propagation. I will use a simple MSE here. it's simply the sum of square of the difference between predicted value and training value
$$MSE = \frac{\sum^n_{i=1} (Y_i - \hat Y_i)^2}{n}$$

And I guess you can directly see the derivative from here
<!-- more -->

```cpp
#include "Matrix.h"  
class Loss {  
public:  
  
    static double meanSquaredError(const Matrix &predicted, const Matrix &actual);  
    static Matrix meanSquaredErrorDerivative(const Matrix &predicted, const Matrix &actual);  
};
```

```cpp
#include "Loss.h"  
  
double Loss::meanSquaredError(const Matrix& predicted, const Matrix& actual) {  
    return ((predicted - actual) * (predicted - actual)).mean();  
}  
  
Matrix Loss::meanSquaredErrorDerivative(const Matrix& predicted, const Matrix& actual) {  
//    std::cout<< "Predicted: " << predicted << std::endl;  
//    std::cout<< "Actual: " << actual << std::endl;  
//    std::cout<< "Actual size" << actual.size() << std::endl;  
    return (predicted - actual)*2/actual.size();  
}
```

# Network

Now finally, the ending bits. 
- The network is gonna take layers and stack them in order
- The network is gonna fit(train), aka change the weights and bias, based on input data
- and it's need to predict

```cpp
#include <vector>  
#include "../Layers/Layer.h"  
#include "../Math/Loss.h"  
  
class Network {  
public:  
    Network(): layers(std::vector<Layer>()) {};  
    ~Network();  
  
    void addLayer(const Layer& layer);  
    void fit(const Matrix &xTrain, const Matrix &yTrain, int numberOfEpoch, double learningRate);  
    Matrix predict(const Matrix& xTest);  
    int getOutputSize();  
  
private:  
    std::vector<Layer> layers;  
  
};
```

# Add layers
```cpp
#include "Network.h"  
  
Network::~Network() = default;
  
void Network::addLayer(const Layer &layer) {  
    layers.push_back(layer);  
}  
```

The fit method is gonna
- take input consisting of actual value (Y) and its label (X)
- the training time, epoch number
- the learning rate, how much of the gradient we apply to the weights and bias



```cpp
void Network::fit(const Matrix &xTrain, const Matrix &yTrain, int numberOfEpoch, double learningRate) {  
    // Check if the number of training data matches the number of labels  
    int trainingSize = xTrain.rows();  
  
    if (trainingSize != yTrain.rows()) {  
        throw std::invalid_argument("The number of training data does not match the number of labels");  
    }  
  
    for (int epoch = 0; epoch < numberOfEpoch; epoch++) {  
        double mse = 0;  
        for (int i = 0; i < trainingSize; ++i) {  
            Matrix output = xTrain.getRow(i);  
            for (auto &layer: layers) {  
                output = layer.forwardPropagation(output);  
            }  
			// We actually don't need this for computation, but we can print it to see if the training is doing properly
            mse += Loss::meanSquaredError(output, yTrain.getRow(i));  
	
			// The dE/dY
            Matrix err = Loss::meanSquaredErrorDerivative(output, yTrain.getRow(i));  

			//remeber it's back not forward, so we do it in reverse
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {  
                err = it->backwardPropagation(err, learningRate);  
            }  
        }  
		
        mse /= trainingSize;  
        std::cout << "Epoch " << epoch << " MSE: " << mse << std::endl;  
    }  
}  
```
# Predict
Prediction is rather easy, just propagate through all the layer and you will have the answer.

```cpp
Matrix Network::predict(const Matrix& xTest) {  
    int testingSize = xTest.rows();  
    Matrix result(0, getOutputSize());  
    for (int i = 0; i < testingSize; ++i) {  
        Matrix output = xTest.getRow(i);  
  
        for (auto &layer: layers) {  
            output = layer.forwardPropagation(output);  
        }  
  
        result.addRow(output);  
    }  
    return result;  
}  
  
int Network::getOutputSize() {  
    // check if the layer is not an activation layer reversely  
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {  
        if (!it->isActivationLayer()) {  
            return it->getOutputSize();  
        }  
    }  
}
```

# Test the code
```cpp
#include <iostream>  
#include "Networks/Network.h"

int main() {
	//XOR gate
	auto xTrain = Matrix({{0,0}, {0,1}, {1,0}, {1,1}});  
	auto yTrain = Matrix({{0}, {1}, {1}, {0}});  

	Network network = Network();  
	network.addLayer(Layer(2, 3));  
	network.addLayer(Layer(ActivationFunctionType::TANH));  
	network.addLayer(Layer(3,1));  
	network.addLayer(Layer(ActivationFunctionType::TANH));  
	network.fit(xTrain, yTrain, 100, 0.1);  

	auto result = network.predict(xTrain);  
	std::cout << result << std::endl;
	return 0
}
```

The output
```
Epoch 1 MSE: 0.48139
...
...
...
Epoch 99 MSE: 0.0133901
0.0314563, 
0.886822, 
0.819525, 
-0.00120216, 
```

Here we go, we get the fairly accurate output with a small epoch number. 

And we can also use the model on 2D images like the famous mnist dataset. The idea is convert the 2d image into a 1D array and create the initial activation layer with input size equals to the image size. However, it requires some image processing code which we will be doing in another day.