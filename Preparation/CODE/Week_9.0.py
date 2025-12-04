import torch
import torch.nn as nn
import torch.optim as optim
 
# Define the network structure
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden1 = nn.Linear(4, 2)  # 4 inputs -> 2 hidden nodes
        self.hidden2 = nn.Linear(2, 1)  # 2 hidden nodes -> 1 output
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
 
    def forward(self, x):
        x = self.sigmoid(self.hidden1(x))
        x = self.sigmoid(self.hidden2(x))
        return x
   
    def init_weights(self):
        # Initial weights and biases for hidden layer h1
        with torch.no_grad():
            self.hidden1.weight.data = torch.tensor([[0.10, 0.20, 0.30, 0.40],
                                                    [0.15, 0.25, 0.35, 0.45]], dtype=torch.float)
            self.hidden1.bias.data = torch.tensor([0.50, 0.55], dtype=torch.float)
 
        # Initial weights and biases for output layer
        with torch.no_grad():
            self.hidden2.weight.data = torch.tensor([[0.60, 0.70]], dtype=torch.float)  # Correct shape [1, 2]
            self.hidden2.bias.data = torch.tensor([0.80], dtype=torch.float)
 
 
# Prepare the dataset
inputs = torch.tensor([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
                       [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
                       [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1],
                       [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]], dtype=torch.float)
targets = torch.tensor([[int((x1 and x2) or (x3 and x4))] for x1, x2, x3, x4 in inputs], dtype=torch.float)
 
# Initialize the network
net = SimpleNN()
 
# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)
 
# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    for x, target in zip(inputs, targets):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(x)
        loss = criterion(output, target)
        loss.backward()         # backpropagation
        optimizer.step()        # update weights
 
# Test the trained model
print("Trained Weights and Biases:")
for name, param in net.named_parameters():
    print(f"{name}: {param.data.numpy()}")
 
print("\nTruth Table (x1, x2, x3, x4) -> predicted_y vs. actual_y")
for x, target in zip(inputs, targets):
    with torch.no_grad():
        pred = net(x)
    print(f"({x.numpy().astype(int)}) -> {pred.item():.4f} vs. {target.item():.0f}")
 