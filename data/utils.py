from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(config, norm_mean, norm_std):
    transform_train = transforms.Compose(
        [
            # transforms.Resize(config["img_size"]),
            # transforms.RandomCrop(config["img_size"], padding=4),
            # transforms.RandomResizedCrop(config["img_size"]),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize(norm_mean, norm_std),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
            ),
            transforms.CenterCrop(config["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(config["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    return transform_train, transform_test


def torch_to_hf_transforms(examples, trans, img_key="image"):
    examples["pixel_values"] = [
        trans(image.convert("RGB")) for image in examples[img_key]
    ]
    return examples


def hf_collate_fn(examples):
    images = []
    labels = []

    for example in examples:
        images.append(example["pixel_values"])
        labels.append(example["label" if "label" in example else "labels"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)

    return {"pixel_values": pixel_values, "labels": labels}


def get_dataset(config):
    if config["dataset"] == "cifar10":
        config["n_classes"] = 10
        config["img_size"] = (32, 32)
        config["patch_size"] = 4
        trans_train, trans_test = get_transforms(
            config, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        )
        ds_train = datasets.CIFAR10(
            train=True,
            root=Path.home() / "data" / "pytorch",
            download=True,
            transform=trans_train,
        )
        ds_test = datasets.CIFAR10(
            train=False, root=Path.home() / "data" / "pytorch", transform=trans_test
        )
    elif config["dataset"] == "monkeys":
        config["n_classes"] = 10
        config["img_size"] = (224, 224)
        config["patch_size"] = 16
        trans_train, trans_test = get_transforms(
            config, (0.4363, 0.4328, 0.3291), (0.2464, 0.2419, 0.2454)
        )
        ds_train = datasets.ImageFolder(
            root=Path.home() / "data" / "monkeys" / "training", transform=trans_train
        )
        ds_test = datasets.ImageFolder(
            root=Path.home() / "data" / "monkeys" / "validation", transform=trans_test
        )
    elif config["dataset"] == "tiny-imagenet":
        config["n_classes"] = 200
        config["img_size"] = (64, 64)
        config["patch_size"] = 8
        # Input seq will be of shape (B, 65, D) 
        trans_train, trans_test = get_transforms(config, _IMAGENET_MEAN, _IMAGENET_STD)
        trans_train = partial(torch_to_hf_transforms, trans=trans_train)
        trans_test = partial(torch_to_hf_transforms, trans=trans_test)
        ds = load_dataset("Maysee/tiny-imagenet")
        ds_train = ds["train"].with_transform(trans_train)
        ds_test = ds["valid"].with_transform(trans_test)
        print(ds_train)
        print(ds_test)
    elif config["dataset"] == "cifar10-hf":
        config["n_classes"] = 10
        config["img_size"] = (32, 32)
        config["patch_size"] = 4
        trans_train, trans_test = get_transforms(
            config, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        )
        trans_train = partial(torch_to_hf_transforms, trans=trans_train, img_key="img")
        trans_test = partial(torch_to_hf_transforms, trans=trans_test, img_key="img")
        ds = load_dataset("cifar10")
        ds_train = ds["train"].with_transform(trans_train)
        ds_test = ds["test"].with_transform(trans_test)
    elif config["dataset"] == "beans":
        config["n_classes"] = 3
        config["img_size"] = (224, 224)
        config["patch_size"] = 16
        trans_train, trans_test = get_transforms(config, _IMAGENET_MEAN, _IMAGENET_STD)
        trans_train = partial(
            torch_to_hf_transforms, trans=trans_train, img_key="image"
        )
        trans_test = partial(torch_to_hf_transforms, trans=trans_test, img_key="image")
        ds = load_dataset("beans")
        ds_train = ds["train"].with_transform(trans_train)
        ds_test = ds["validation"].with_transform(trans_test)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    train_loader = DataLoader(
        ds_train,
        collate_fn=hf_collate_fn,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        ds_test,
        collate_fn=hf_collate_fn,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_loader, test_loader


def get_ds_stats(ds):
    # return the per-channel mean and std of the dataset
    # ds: dataset object
    # return: mean, std
    x = np.stack([np.asarray(ds[i][0]) for i in range(len(ds))])
    return x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3))

