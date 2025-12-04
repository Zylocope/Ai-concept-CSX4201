import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input data for XOR problem
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

# Target output
targets = np.array([[0], [1], [1], [0]])  # Adjusted for XOR problem representation

# Learning rate
learning_rate = 0.1

# Number of epochs
epochs = 10000

# Initialize weights and bias
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

weights_input_hidden = np.random.uniform(size=(input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(size=(hidden_layer_size, output_layer_size))
bias_hidden = np.random.uniform(size=(1, hidden_layer_size))
bias_output = np.random.uniform(size=(1, output_layer_size))

# Training loop
for epoch in range(epochs):
    total_error = 0
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)

    for i in indices:
        # Forward pass
        input_layer = inputs[i].reshape(1, -1)
        target = targets[i].reshape(1, -1)

        # Hidden layer
        hidden_layer_input = np.dot(input_layer, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        # Output layer
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output = sigmoid(output_layer_input)

        # Calculate error
        error = target - output
        total_error += np.sum(error**2)

        # Backward pass
        d_output = error * sigmoid_derivative(output)
        error_hidden_layer = d_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += input_layer.T.dot(d_hidden_layer) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Total Error: {total_error}')

# Testing the trained network
for i in range(len(inputs)):
    input_layer = inputs[i].reshape(1, -1)
    hidden_layer_input = np.dot(input_layer, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)
    print(f'Input: {inputs[i]}, Predicted Output: {output}, Actual Output: {targets[i]}')