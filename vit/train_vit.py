from pathlib import Path
import torch
import torchvision
from torchvision import datasets
import argparse
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from vit import VisionTransformer


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return vars(parser.parse_args())

def load_dataset(config):

    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    if config['dataset'] == 'cifar10':
        ds_train = datasets.CIFAR10(train=True, root=Path.home()/'data'/'pytorch', download=True, transform=transform)
        ds_test = datasets.CIFAR10(train=False, root=Path.home()/'data'/'pytorch', transform=transform)
        config['n_classes'] = 10
        config['img_size'] = (32, 32)
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
    writer = SummaryWriter()

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
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
        writer.add_scalars('Loss', {'Train': train_loss, 'Test': test_loss}, epoch)

if __name__ == '__main__':
    config = get_args()
    train(config)