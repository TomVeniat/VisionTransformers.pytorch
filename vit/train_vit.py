import os
from pathlib import Path
import torch
import torchvision
from torchvision import datasets, transforms
import argparse
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from vit import VisionTransformer


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='monkeys')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--exp-name', type=str, default=None)

    return vars(parser.parse_args())

def get_transforms(config, norm_mean, norm_std):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(config['img_size'], padding=4),
        transforms.Resize(config['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    return transform_train, transform_test

def load_dataset(config):

    if config['dataset'] == 'cifar10':
        config['n_classes'] = 10
        config['img_size'] = (32, 32)
        trans_train, trans_test = get_transforms(config, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ds_train = datasets.CIFAR10(train=True, root=Path.home()/'data'/'pytorch', download=True, transform=trans_train)
        ds_test = datasets.CIFAR10(train=False, root=Path.home()/'data'/'pytorch', transform=trans_test)
    elif config['dataset'] == 'monkeys':
        config['n_classes'] = 10
        config['img_size'] = (224, 224)
        trans_train, trans_test = get_transforms(config, (0.4363, 0.4328 , 0.3291), (0.2464, 0.2419, 0.2454))
        ds_train = datasets.ImageFolder(root=Path.home()/'data'/'monkeys'/'training', transform=trans_train)
        ds_test = datasets.ImageFolder(root=Path.home()/'data'/'monkeys'/'validation', transform=trans_test)
    else: 
        raise ValueError(f"Unknown dataset: {config['dataset']}")


    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'])
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=config['batch_size'])

    return train_loader, test_loader

def train(config):

    train_loader, test_loader = load_dataset(config)

    model = VisionTransformer(config['img_size'], config['patch_size'], config['embed_dim'], config['n_layers'], config['n_heads'], n_classes=config['n_classes'], dropout=config['dropout'])
    model.to(config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # last_commit_msg = os.popen('git log -1 --pretty=%B').read().strip()
    # writer = SummaryWriter(comment=f"_{last_commit_msg}")
    layout = {
        "metrics": {
            "loss": ["Multilinfe", ["loss/train", "loss/test"]],
            "accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
        },
    }

    writer = SummaryWriter(comment=f"_{config['exp_name']}" if config['exp_name'] is not None else "")
    writer.add_custom_scalars(layout)



    for epoch in trange(config['epochs'], desc="Training progress"):
        model.train()
        total = 0
        correct = 0
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(config['device']), y.to(config['device'])

            optimizer.zero_grad()
            y_hat = model(x)[:,0,:]
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_loss = 0

            for x, y in test_loader:
                x, y = x.to(config['device']), y.to(config['device'])
                y_hat = model(x)[:,0,:]
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                total_loss += torch.nn.functional.cross_entropy(y_hat, y).item()

            test_acc = correct / total
            test_loss = total_loss / len(test_loader)
            tqdm.write(f"Epoch {epoch}: Accuracy: {test_acc}")
            # print(f"Epoch {epoch}: Accuracy: {test_acc}")
        # writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc}, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)
        # writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)

if __name__ == '__main__':
    config = get_args()
    train(config)