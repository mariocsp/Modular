
'''
contains functionality  for creating pytorch dataloader for image
classification data
'''

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transform:transforms.Compose =  None,
    batch_size:int = 32,
    num_workers : int = NUM_WORKERS):

    """
    takes in an train and test path, then turn them into pytorch dataset
    and pytorch dataloader.

    Args:

        train_dir: Path to training directory
        test_dir : Path to training directory
        transform: Transformation apllied to dataset
        batch_size: Number of sample per batch in each dataloader
        num_workers : Number of cpu core use to load dataloader

    Return:
        A tupple of (train dataloader, test_dataloader, class_names)

    Example Usage:
        train dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir, test_dir, tranform, batch_size, num_workers
        )

    """

    train_data = datasets.ImageFolder(root =  train_dir,
                                  transform = transform, #trans fitur
                                  target_transform= None) #trans label
    test_data = datasets.ImageFolder(root =  test_dir,
                                  transform = transform, #trans fitur
                                  target_transform= None) #trans label

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last = True,
                                num_workers = num_workers,
                                pin_memory = True)

    test_dataloader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last= False,
                                num_workers=num_workers,
                                pin_memory = True)

    return train_dataloader, test_dataloader, class_names
