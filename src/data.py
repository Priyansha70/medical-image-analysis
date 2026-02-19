import os
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Expects directory like:
      data_dir/
        train/NORMAL, train/PNEUMONIA
        val/NORMAL, val/PNEUMONIA   (optional; if missing, use test as val)
        test/NORMAL, test/PNEUMONIA
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, train=True))

    if os.path.isdir(val_dir):
        val_ds = datasets.ImageFolder(val_dir, transform=get_transforms(img_size, train=False))
    else:
        val_ds = datasets.ImageFolder(test_dir, transform=get_transforms(img_size, train=False))

    test_ds = datasets.ImageFolder(test_dir, transform=get_transforms(img_size, train=False))

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, idx_to_class
