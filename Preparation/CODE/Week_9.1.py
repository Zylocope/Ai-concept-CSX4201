import numpy as np
import torch
from torchvision import datasets, transforms
 
# 1. Load MNIST using PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='D:/Course/Ai concept CSX4201/Preparation/CODE/data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='D:/Course/Ai concept CSX4201/Preparation/CODE/data', train=False, transform=transform)
 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
 
x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))
 
# 2. Flatten images from (1, 28, 28) to (784,)
x_train = x_train.view(-1, 784)
x_test = x_test.view(-1, 784)
 
 
# 3. One-hot encode the labels
def one_hot_encode(y, num_classes=10):
    # y is a tensor of shape (batch_size,)
    # returns a one-hot encoded tensor of shape (batch_size, num_classes)
    return torch.zeros(len(y), num_classes).scatter_(1, y.unsqueeze(1), 1.)
 
y_train_oh = one_hot_encode(y_train, num_classes=10)
y_test_oh = one_hot_encode(y_test, num_classes=10)
 
 
 
# Convert to NumPy
x_train = x_train.numpy()
y_train = y_train.numpy()
x_test = x_test.numpy()
y_test = y_test.numpy()
y_train_oh = y_train_oh.numpy()
y_test_oh = y_test_oh.numpy()
 
 
# ================== SHOW ===================
import matplotlib.pyplot as plt
x_train_img = x_train[0].reshape(28, 28)  # First image in training set
x_test_img = x_test[0].reshape(28, 28)    # First image in testing set
 
# Plotting the first training image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.imshow(x_train_img, cmap='gray')
plt.title('First Training Image')
plt.colorbar()
plt.grid(False)
 
# Plotting the first testing image
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.imshow(x_test_img, cmap='gray')
plt.title('First Testing Image')
plt.colorbar()
plt.grid(False)
 
plt.show()
 
 
 
 
 
 
 
def init_params(input_dim=784, hidden_dim=128, output_dim=10, seed=42):
    np.random.seed(seed)
 
    # Weights and biases for hidden layer
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
 
    # Weights and biases for output layer
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))
 
    return W1, b1, W2, b2
 
 
 
def relu(x):
    return np.maximum(0, x)
 
def relu_derivative(x):
    # derivative of ReLU: 1 if x>0 else 0
    xtmp = (x > 0).astype(x.dtype)
    return xtmp
 
def softmax(logits):
    # Numerically stable softmax
    # shift by max logit to avoid overflow
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)
 
def cross_entropy_loss(predictions, targets):
    """
    predictions: shape (batch_size, 10), each row is a probability distribution
    targets: one-hot shape (batch_size, 10)
    """
    # Add a small epsilon to avoid log(0)
    eps = 1e-12
    return -np.sum(targets * np.log(predictions + eps)) / targets.shape[0]
 
def accuracy(predictions, targets):
    """
    predictions: shape (batch_size, 10) (probabilities)
    targets: one-hot shape (batch_size, 10)
    """
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    return np.mean(pred_labels == true_labels)
 
def forward_pass(X, W1, b1, W2, b2):
    """
    X: input data, shape (batch_size, 784)
    Returns:
      - hidden layer output
      - output layer logits
      - output layer probabilities (softmax)
    """
    # 1. Hidden layer
    z1 = X.dot(W1) + b1     # shape (batch_size, hidden_dim)
    a1 = relu(z1)           # ReLU activation
 
    # 2. Output layer
    z2 = a1.dot(W2) + b2    # shape (batch_size, 10)
    probs = softmax(z2)     # shape (batch_size, 10)
 
    return z1, a1, z2, probs
 
 
 
 
def backward_pass(X, y_true, z1, a1, z2, probs, W1, b1, W2, b2):
    """
    X: shape (batch_size, 784)
    y_true: one-hot shape (batch_size, 10)
    z1, a1: hidden layer pre/post-activation
    z2, probs: output layer pre/post-activation
    W1, b1, W2, b2: parameters
    Return:
      dW1, db1, dW2, db2 (gradients for each parameter)
    """
 
    batch_size = X.shape[0]
 
    # --- Grad w.r.t. output logits z2 ---
    # derivative of cross-entropy + softmax
    dz2 = (probs - y_true)  # shape (batch_size, 10)
 
    # --- Grad for W2 and b2 ---
    dW2 = a1.T.dot(dz2) / batch_size  # shape (hidden_dim, 10)
    db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
 
    # --- Backprop into hidden layer ---
    da1 = dz2.dot(W2.T)              # shape (batch_size, hidden_dim)
    dz1 = da1 * relu_derivative(z1)  # shape (batch_size, hidden_dim)
 
    # --- Grad for W1 and b1 ---
    dW1 = X.T.dot(dz1) / batch_size       # shape (784, hidden_dim)
    db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
 
    return dW1, db1, dW2, db2
 
 
 
def train_mnist(
    x_train, y_train,
    x_test, y_test,
    hidden_dim=128,
    num_epochs=5,
    batch_size=64,
    learning_rate=0.1
):
    # 1. Initialize parameters
    W1, b1, W2, b2 = init_params(
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10
    )
 
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size
 
    for epoch in range(num_epochs):
        # Shuffle the training data each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
 
        # Mini-batch training
        for i in range(num_batches):
            start = i * batch_size
            end   = start + batch_size
            X_batch = x_train[start:end]
            y_batch = y_train[start:end]
 
            # Forward pass
            z1, a1, z2, probs = forward_pass(X_batch, W1, b1, W2, b2)
 
            # Compute loss (optional for logging)
            loss = cross_entropy_loss(probs, y_batch)
 
            # Backward pass
            dW1, db1, dW2, db2 = backward_pass(
                X_batch, y_batch,
                z1, a1, z2, probs,
                W1, b1, W2, b2
            )
 
            # Update parameters
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
 
            # Print batch loss occasionally
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss:.4f}")
 
        # --- End of epoch: Evaluate on training & testing ---
        _, _, _, train_probs = forward_pass(x_train, W1, b1, W2, b2)
        train_loss = cross_entropy_loss(train_probs, y_train)
        train_acc  = accuracy(train_probs, y_train)
 
        _, _, _, test_probs = forward_pass(x_test, W1, b1, W2, b2)
        test_loss = cross_entropy_loss(test_probs, y_test)
        test_acc  = accuracy(test_probs, y_test)
 
        print(f"\n[Epoch {epoch+1} Summary]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}\n")
 
    return W1, b1, W2, b2
 
 
 
 
if __name__ == "__main__":
    # Hyperparameters
    hidden_dim = 128
    num_epochs = 5
    batch_size = 64
    # batch_size = 1
    learning_rate = 0.1
 
    # Train the network
    W1, b1, W2, b2 = train_mnist(
        x_train, y_train_oh,
        x_test,  y_test_oh,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
   
   
   
    # Using the first image from x_test
    first_test_image = x_test[20]  # This is already a numpy array flattened to shape (784,)
 
    # Convert to a 2D numpy array because forward_pass expects a batch, even if it's a batch of one
    first_test_image = first_test_image.reshape(1, -1)  # Reshape to (1, 784)
 
    # Run the forward pass using the trained parameters
    _, _, _, probabilities = forward_pass(first_test_image, W1, b1, W2, b2)
 
    # Display the probabilities
    print("Probabilities for each digit class:")
    print(probabilities)
 
    # Optionally, print the predicted digit
    predicted_digit = np.argmax(probabilities, axis=1)
    print(f"Predicted digit: {predicted_digit[0]}")
 
 
 
    # ================== SHOW ===================
    import matplotlib.pyplot as plt
    x_test_img = first_test_image.reshape(28, 28)    # First image in testing set
 
    # Plotting the first training image
    plt.figure(figsize=(10, 5))
    plt.imshow(x_test_img, cmap='gray')
    plt.title('First Testing Image')
    plt.colorbar()
    plt.grid(False)
 
    plt.show()