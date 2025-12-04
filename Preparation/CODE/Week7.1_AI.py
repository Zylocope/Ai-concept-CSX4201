import numpy as np
 
# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
 
# Initializing weights and bias
np.random.seed(42)  # for reproducibility
weights = np.random.rand(2)  # weights for two inputs
bias = np.random.rand(1)  # one bias
 
# Dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 1])  # Adjusted to correct OR problem representation
 
# Learning rate
learning_rate = 0.1  # Increased for faster convergence
 
# Training loop
for epoch in range(10000):  # Reduced number of epochs
    total_error = 0
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
 
    for i in indices:
        input_layer = inputs[i]
        target = targets[i]
 
        z = np.dot(input_layer, weights) + bias
        output = sigmoid(z)
        error = 0.5 * (target - output) ** 2
        total_error += error
 
        dE_dy = output - target
        dy_dz = sigmoid_derivative(z)
        dz_dw = input_layer
        dz_db = 1
 
        gradient_weights = dE_dy * dy_dz * dz_dw
        gradient_bias = dE_dy * dy_dz * dz_db
 
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
 
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Average Loss: {total_error / len(inputs)}")
 
print("Final weights:", weights)
print("Final bias:", bias)
 
for i in range(len(inputs)):
    z = np.dot(inputs[i], weights) + bias
    output = sigmoid(z)
    print(f"Input: {inputs[i]}, Predicted Output: {output[0]:.4f}, Actual Target: {targets[i]}")