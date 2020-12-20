import torch
import torchvision
import torchvision.transforms as transforms
from model import Net
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img, labels, n=4):
    fig, ax = plt.subplots(1,1)
    img = img / 2 + 0.5
    npimg = img.numpy()
    p = ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_xticks([35*i + 35/2 for i in range(n)])
    ax.set_xticklabels(labels)
    plt.show()

def load_data(download=True):
    # Load and transform CIFAR10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    return testloader

def show_random_results(testloader, net):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    imshow(torchvision.utils.make_grid(images), [classes[predicted[j]] for j in range(4)])

def main():
    net = Net()
    net.load_state_dict(torch.load("./cifar_net.pth"))

    y = []
    y_pred = []
    testloader = load_data()
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        y += labels
        y_pred += predicted

    print(classification_report(y, y_pred, target_names=classes))

if __name__ == '__main__':
    main()
