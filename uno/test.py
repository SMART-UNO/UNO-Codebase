import torch

# Define a simple neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

net = Net()
optim = torch.optim.Adam(net.parameters(), lr=0.01)

# Print the initial weights of the network
print(net.state_dict()['fc1.weight'])

# Run the training loop
for i in range(10):
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    optim.zero_grad()
    loss = torch.nn.functional.mse_loss(net(x), y)
    loss.backward()
    optim.step()

# Print the final weights of the network
print(net.state_dict()['fc1.weight'])
