import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
 
x = np.array([0.5, 2.3, 2.9]).reshape(-1, 1)
y = np.array([1.4, 1.9, 3.2]).reshape(-1, 1)
 
def max_lik_estimate(x, y):
    theta_ml = np.linalg.inv(x.T @ x) @ x.T @ y
    return theta_ml
 
def predict_with_estimate(Xtest, theta):
    y_pred = Xtest @ theta
    return y_pred

N, D = x.shape
X_aug = np.hstack((np.ones((N, 1)), x))
print("X_aug: ", X_aug)
 
theta_ml = max_lik_estimate(X_aug, y)
print("Theta_ML: ", theta_ml)
 
# define test case
Xtest = np.linspace(-5, 5, 100).reshape(-1, 1)
Xtest_aug = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))
y_pred = predict_with_estimate(Xtest_aug, theta_ml)
 
plt.figure()
plt.plot(x, y, '+', markersize=10, label = 'Training data')
plt.plot(Xtest, y_pred, label = f'y = {theta_ml[0, 0]}')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
 
def RSME(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))
 
rsme_test = RSME(y, X_aug @ theta_ml)
print("RSME: ", rsme_test)
 
def loss_function(Phi, y, theta):
    return np.sum((y - Phi.dot(theta))**2)
 
def gradient_descent(Phi, y, theta, alpha, iterations):
    m = len(y)
    cost_history = [0] * iterations    
    for it in range(iterations):
        prediction = Phi.dot(theta)
        error = y - prediction
        gradient = -2 * Phi.T.dot(error)
        alpha_new = alpha / (1+0.00000001*it)
        theta -= alpha_new * gradient
        cost_history[it] = loss_function(Phi, y, theta)
        #Predict test outcomes
        if np.isnan(cost_history[it] or (it > 0 and cost_history[it] > cost_history[it-1])):
            print(f"Breaking at iteration {it} due to NaN or increasing cost.")
            break
    return theta, cost_history
 
np.random.seed(41)
numberTheta = 2
theta = np.random.uniform(-1, 1, (numberTheta, 1))
alpha = 0.0002 # learning rate
iterations = 10000 # number of iterations
 
# Run gradient descent
theta_gd, cost_history = gradient_descent(X_aug, y, theta, alpha, iterations)
print("theta_gd:", theta_gd)
 
plt.figure()
plt.plot(range(iterations), cost_history, 'b.')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost function over Time")
plt.show()