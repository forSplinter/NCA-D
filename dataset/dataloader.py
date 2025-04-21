import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def mnist_dataloader(batch_size=32, train=True, shuffle=True):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    mnist_data = datasets.MNIST(
        root="./data", train=train, download=True, transform=transform
    )
    data_loader = DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=shuffle)


def cifar10_dataloader(batch_size=32, train=True, shuffle=True):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    cifar_data = datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )

    data_loader = DataLoader(dataset=cifar_data, batch_size=batch_size, shuffle=shuffle)

    return data_loader
