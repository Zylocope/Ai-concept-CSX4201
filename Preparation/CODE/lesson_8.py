import math
import random
 
def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + math.exp(-z))
 
def sigmoid_deriv(z):
    """
    Derivative of the sigmoid function wrt its input z.
 
    If you already have sigmoid(z) = s, you can do s * (1 - s).
    Here, for clarity, we compute it directly from z:
    s = sigmoid(z)
    ds/dz = s * (1 - s)
    """
    s = sigmoid(z)
    return s * (1 - s)
 
# ----------------------------------------------
# 1. Prepare the dataset
# ----------------------------------------------
# All possible inputs (4 bits) = 16 combinations
data_inputs = []
data_targets = []
 
for x1 in [0,1]:
    for x2 in [0,1]:
        for x3 in [0,1]:
            for x4 in [0,1]:
                # The logical function:
                # (x1 AND x2) OR (x3 AND x4)
                target = (x1 and x2) or (x3 and x4)
                data_inputs.append([x1, x2, x3, x4])
                data_targets.append(target)
 
 
for i in range(len(data_inputs)):
    print("Input:", data_inputs[i], " Output:", data_targets[i])
 
 
 
# Convert booleans to 0/1 (if needed)
# But in Python, True == 1, False == 0, so we can keep it as-is.
 
# ----------------------------------------------
# 2. Initialize weights and biases
#    (4 inputs -> 2 hidden -> 1 output)
# ----------------------------------------------
# random.seed(42)  # for reproducible results
 
# # Hidden layer neuron h1: w_{1,1}, w_{2,1}, w_{3,1}, w_{4,1}, b1
# w_11 = random.uniform(-0.5, 0.5)
# w_21 = random.uniform(-0.5, 0.5)
# w_31 = random.uniform(-0.5, 0.5)
# w_41 = random.uniform(-0.5, 0.5)
# b1   = random.uniform(-0.5, 0.5)
 
# # Hidden layer neuron h2: w_{1,2}, w_{2,2}, w_{3,2}, w_{4,2}, b2
# w_12 = random.uniform(-0.5, 0.5)
# w_22 = random.uniform(-0.5, 0.5)
# w_32 = random.uniform(-0.5, 0.5)
# w_42 = random.uniform(-0.5, 0.5)
# b2   = random.uniform(-0.5, 0.5)
 
# # Output neuron: w_{h1,o}, w_{h2,o}, b_o
# w_h1o = random.uniform(-0.5, 0.5)
# w_h2o = random.uniform(-0.5, 0.5)
# b_o   = random.uniform(-0.5, 0.5)
 
# ======================================= Setup =================================
# h1:
#    w_{1,1} = 0.10
#    w_{2,1} = 0.20
#    w_{3,1} = 0.30
#    w_{4,1} = 0.40
#    b_1     = 0.50
 
# h2:
#    w_{1,2} = 0.15
#    w_{2,2} = 0.25
#    w_{3,2} = 0.35
#    w_{4,2} = 0.45
#    b_2     = 0.55
 
# Output neuron:
#    w_{h1,o} = 0.60
#    w_{h2,o} = 0.70
#    b_o      = 0.80
 
w_11 = 0.1
w_21 = 0.2
w_31 = 0.3
w_41 = 0.4
b1   = 0.5
 
# Hidden layer neuron h2: w_{1,2}, w_{2,2}, w_{3,2}, w_{4,2}, b2
w_12 = 0.15
w_22 = 0.25
w_32 = 0.35
w_42 = 0.45
b2   = 0.55
 
# Output neuron: w_{h1,o}, w_{h2,o}, b_o
w_h1o = 0.60
w_h2o = 0.70
b_o   = 0.80
 
# Learning rate
eta = 0.1
 
# ----------------------------------------------
# 3. Training function (forward + backprop)
# ----------------------------------------------
def train_one_epoch():
    global w_11, w_21, w_31, w_41, b1
    global w_12, w_22, w_32, w_42, b2
    global w_h1o, w_h2o, b_o
 
    # Loop over all training examples (online/SGD update)
    for i, x in enumerate(data_inputs):
        # x is [x1, x2, x3, x4], target is data_targets[i]
        x1, x2, x3, x4 = x
        target = data_targets[i]  # 0 or 1
 
        # =========== Forward pass ===========
        # Hidden neuron h1
        z1 = (x1 * w_11) + (x2 * w_21) + (x3 * w_31) + (x4 * w_41) + b1
        h1 = sigmoid(z1)
 
        # Hidden neuron h2
        z2 = (x1 * w_12) + (x2 * w_22) + (x3 * w_32) + (x4 * w_42) + b2
        h2 = sigmoid(z2)
 
        # Output neuron
        zo = (h1 * w_h1o) + (h2 * w_h2o) + b_o
        y_hat = sigmoid(zo)
 
        # =========== Compute error (MSE) ===========
        # E = 1/2 (y_hat - target)^2
        # We'll need dE/dy_hat = (y_hat - target)
        # error = 0.5 * (y_hat - target)**2
        error = (y_hat - target)**2
 
        # =========== Backprop: Output layer ===========
        # dError/dy_hat = (y_hat - target)
        # dE_dy = (y_hat - target)
        dE_dy = 2*(y_hat - target)
        # dError/dzo = dE_dy * dy_hat/dzo
        # dy_hat/dzo = y_hat(1 - y_hat)
        dE_dzo = dE_dy * y_hat * (1 - y_hat)
 
        # Gradients for output weights
        dE_dw_h1o = dE_dzo * h1  # partial derivative wrt w_{h1,o}
        dE_dw_h2o = dE_dzo * h2  # partial derivative wrt w_{h2,o}
        dE_db_o   = dE_dzo * 1.0 # partial derivative wrt b_o
 
        # =========== Backprop: Hidden layer ===========
        # For h1:
        # dError/dw_11 = dError/dzo * dzo/dh1 * dh1/dz1 * dz1/dw_11
        # dzo/dh1 = w_h1o
        # dh1/dz1 = h1(1 - h1)
        # dz1/dw_11 = x1
        dzo_dh1     = w_h1o
        dh1_dz1     = h1 * (1 - h1)
 
        dE_dz1 = dE_dzo * dzo_dh1 * dh1_dz1
        dE_dw_11 = dE_dz1 * x1
        dE_dw_21 = dE_dz1 * x2
        dE_dw_31 = dE_dz1 * x3
        dE_dw_41 = dE_dz1 * x4
        dE_db1   = dE_dz1 * 1.0
 
        # For h2:
        # dzo/dh2 = w_h2o
        # dh2/dz2 = h2(1 - h2)
        dzo_dh2     = w_h2o
        dh2_dz2     = h2 * (1 - h2)
 
        dE_dz2 = dE_dzo * dzo_dh2 * dh2_dz2
        dE_dw_12 = dE_dz2 * x1
        dE_dw_22 = dE_dz2 * x2
        dE_dw_32 = dE_dz2 * x3
        dE_dw_42 = dE_dz2 * x4
        dE_db2   = dE_dz2 * 1.0
 
        # =========== Gradient Descent Update ===========
        # Output layer
        w_h1o -= eta * dE_dw_h1o
        w_h2o -= eta * dE_dw_h2o
        b_o   -= eta * dE_db_o
 
        # Hidden layer h1
        w_11 -= eta * dE_dw_11
        w_21 -= eta * dE_dw_21
        w_31 -= eta * dE_dw_31
        w_41 -= eta * dE_dw_41
        b1   -= eta * dE_db1
 
        # Hidden layer h2
        w_12 -= eta * dE_dw_12
        w_22 -= eta * dE_dw_22
        w_32 -= eta * dE_dw_32
        w_42 -= eta * dE_dw_42
        b2   -= eta * dE_db2
 
# ----------------------------------------------
# 4. Run Training
# ----------------------------------------------
num_epochs = 2000  # increase if needed
for epoch in range(num_epochs):
    train_one_epoch()
 
# ----------------------------------------------
# 5. Test the trained model
#    (Check all 16 combinations)
# ----------------------------------------------
print("Trained Weights and Biases:")
print(f" h1: w_11={w_11:.4f}, w_21={w_21:.4f}, w_31={w_31:.4f}, w_41={w_41:.4f}, b1={b1:.4f}")
print(f" h2: w_12={w_12:.4f}, w_22={w_22:.4f}, w_32={w_32:.4f}, w_42={w_42:.4f}, b2={b2:.4f}")
print(f" out: w_h1o={w_h1o:.4f}, w_h2o={w_h2o:.4f}, b_o={b_o:.4f}")
print()
 
def forward_pass(x1, x2, x3, x4):
    """Compute the output y_hat for a single 4-bit input."""
    z1 = (x1 * w_11) + (x2 * w_21) + (x3 * w_31) + (x4 * w_41) + b1
    h1 = sigmoid(z1)
    z2 = (x1 * w_12) + (x2 * w_22) + (x3 * w_32) + (x4 * w_42) + b2
    h2 = sigmoid(z2)
    zo = (h1 * w_h1o) + (h2 * w_h2o) + b_o
    return sigmoid(zo)
 
print("Truth Table (x1, x2, x3, x4) -> predicted_y vs. actual_y")
for x1 in [0,1]:
    for x2 in [0,1]:
        for x3 in [0,1]:
            for x4 in [0,1]:
                y_hat = forward_pass(x1, x2, x3, x4)
                target = (x1 and x2) or (x3 and x4)
                print(f"({x1},{x2},{x3},{x4}) -> {y_hat:.4f} vs. {target}")