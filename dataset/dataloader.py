import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


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

    return data_loader


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


def stl10_dataloaders(split="train", batch_size=1, targets_label=None):
    transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])

    dataset = datasets.STL10(
        root="./data", split=split, download=True, transform=transform
    )
    targets = torch.tensor(dataset.labels)

    class_indices = {
        label: (targets == label).nonzero(as_tuple=True)[0].tolist()
        for label in range(10)
    }

    class_dataloaders = {}
    for label, indices in class_indices.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        class_dataloaders[label] = loader

    if targets_label is not None:
        if targets_label in class_dataloaders:
            return class_dataloaders[targets_label]
        else:
            raise ValueError(f"Label {targets_label} not found in dataset.")

    return class_dataloaders
