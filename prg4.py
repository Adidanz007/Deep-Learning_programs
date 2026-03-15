#4. Implement a simple feedforward neural network to demonstrate the impact of different learning rates and activation functions.

import numpy as np

# Sample dataset (XOR problem)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],[1],[1],[0]])

# Activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1,0)


# Training function
def train_network(activation, activation_derivative, learning_rate):

    np.random.seed(1)

    # Initialize weights
    weights = np.random.rand(2,1)

    print("\nTraining with learning rate:", learning_rate)

    for epoch in range(1000):

        # Forward pass
        z = np.dot(X, weights)
        output = activation(z)

        # Error
        error = y - output

        # Backpropagation
        d_output = error * activation_derivative(output)

        # Update weights
        weights += learning_rate * np.dot(X.T, d_output)

    print("Final output:")
    print(output)


# -------- Run Experiments --------

print("Using Sigmoid Activation")
train_network(sigmoid, sigmoid_derivative, 0.1)
train_network(sigmoid, sigmoid_derivative, 0.5)

print("\nUsing ReLU Activation")
train_network(relu, relu_derivative, 0.1)
train_network(relu, relu_derivative, 0.5)