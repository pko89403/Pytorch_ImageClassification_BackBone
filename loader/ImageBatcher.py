import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from config import config


def load_split_train_test(datadir, valid_size = .2):
    
    image_size = config().json_data["MODEL"]["IMAGE_AXIS_LENGTH"]
    bath_size = config().json_data["MODEL"]["BATCH_SIZE"]

    train_transform = transforms.Compose(transforms=[transforms.Resize(image_size),
                                                    transforms.CenterCrop(image_size),
                                                    transforms.ToTensor()])
    test_transform = transforms.Compose(transforms=[transforms.Resize(image_size),
                                                    transforms.CenterCrop(image_size),
                                                    transforms.ToTensor()])

    # generic data loader where the images are arranged root/label/_.png
    train_data = datasets.ImageFolder(root=datadir,
                                        transform=train_transform)

    test_data = datasets.ImageFolder(root=datadir,
                                        transform=test_transform)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    # samples elements randomly from a given list of indices
    train_sampler = SubsetRandomSampler(indices=train_idx)
    test_sampler = SubsetRandomSampler(indices=test_idx)

    train_loader = DataLoader(dataset=train_data,
                            sampler=train_sampler,
                            batch_size=bath_size)
    test_loader = DataLoader(dataset=test_data,
                            sampler=test_sampler,
                            batch_size=bath_size)
    return train_loader, test_loader

if __name__ == "__main__":
    
    BASE_PATH = os.path.join( os.getcwd(), "dataset/raw/")
    IMAGE_AXIS_LENGTH = 128
    BATCH_SIZE = 128
    VALID_SIZE = .2

    train_loader, test_loader = load_split_train_test(BASE_PATH, VALID_SIZE)
    print(train_loader.dataset.classes)    