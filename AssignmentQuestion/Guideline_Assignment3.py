import numpy as np
import matplotlib.pyplot as plt

# Define training set
# Create an array of input values, reshaped to a column vector with 5 rows and 1 column
X = np.array([-3, -1, 0.0, 1, 3]).reshape(-1,1)
# Create an array of corresponding output values, reshaped to a column vector
y = np.array([-1.2, -0.7, 0.14, 0.67, 1.67]).reshape(-1,1)

# Define the maximum likelihood estimation function for calculating theta
def max_lik_estimate(X, y):
    # Calculate the inverse of X transposed times X
    inverse_term = np.linalg.inv(X.T @ X)
    # Multiply the inverse by X transposed, and then by y to compute theta maximum likelihood
    theta_ml = inverse_term @ (X.T @ y)
    return theta_ml

# Define function to make predictions based on estimated theta
def predict_with_estimate(Xtest, theta):
    # Return predictions as dot product of test inputs and theta
    prediction = Xtest @ theta
    return prediction 

# Calculate theta using the training data
theta_ml = max_lik_estimate(X, y)
# Define a range of test input values from -5 to 5
Xtest = np.linspace(-5, 5, 100).reshape(-1,1)
# Generate predictions for the test inputs
ml_prediction = predict_with_estimate(Xtest, theta_ml)

# Create a plot to display data and predictions
plt.figure()
# Plot original data points
plt.plot(X, y, '+', markersize=10, label='Data points')
# Plot line of best fit
plt.plot(Xtest, ml_prediction, label=f'y={theta_ml[0][0]:.2f}x' )
plt.xlabel("X-axis ($x$)")  # Label the x-axis
plt.ylabel("Y-axis ($y$)")  # Label the y-axis
plt.title("Plot of Training Data Set")  # Title for the plot
plt.xlim([-5, 5])  # Set the x-axis limits
plt.legend()  # Display legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot
