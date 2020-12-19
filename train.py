import torch
import torchvision
import torchvision.transforms as transforms
from model import Net
import torch.optim as optim
import argparse
import torch.nn as nn
import torch.nn.functional as F

def load_data(download=True):
    # Load and transform CIFAR10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    return trainloader

def train(trainloader, net):

    # Define criterion and optimization method
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Start the NN training
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    return net

def main():
    trainloader = load_data()

    net = Net()
    net = train(trainloader, net)
    print('Finished Training')

    # Save the trained NN
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()
