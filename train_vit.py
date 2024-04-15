from collections import defaultdict
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from data.utils import get_dataset
from model.vit import VisionTransformer


def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="tiny-imagenet")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    # parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--exp-name", type=str, default=None)

    return vars(parser.parse_args())


def train(config):
    train_loader, test_loader = get_dataset(config)

    model = VisionTransformer(
        config["img_size"],
        config["patch_size"],
        config["embed_dim"],
        config["n_layers"],
        config["n_heads"],
        n_classes=config["n_classes"],
        dropout=config["dropout"],
    )
    model.to(config["device"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # last_commit_msg = os.popen('git log -1 --pretty=%B').read().strip()
    # writer = SummaryWriter(comment=f"_{last_commit_msg}")
    layout = {
        "metrics": {
            "loss": ["Multiline", ["loss/train", "loss/test"]],
            "accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
        },
    }

    writer = SummaryWriter(
        comment=f"_{config['exp_name']}" if config["exp_name"] is not None else ""
    )
    writer.add_custom_scalars(layout)

    for epoch in trange(config["epochs"], desc="Training progress"):
        model.train()
        total = 0
        correct = 0
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            x, y = (
                batch["pixel_values"].to(config["device"]),
                batch["labels"].to(config["device"]),
            )
            # x, y = x.to(config['device']), y.to(config['device'])

            optimizer.zero_grad()
            y_hat = model(x)[:, 0, :]
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
            for batch in test_loader:
                x, y = (
                    batch["pixel_values"].to(config["device"]),
                    batch["labels"].to(config["device"]),
                )
                # x, y = x.to(config['device']), y.to(config['device'])
                y_hat = model(x)[:, 0, :]
                _, predicted = torch.max(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                total_loss += torch.nn.functional.cross_entropy(y_hat, y).item()
            test_acc = correct / total
            test_loss = total_loss / len(test_loader)
            tqdm.write(f"Epoch {epoch}: Accuracy: {test_acc}")
            # print(f"Epoch {epoch}: Accuracy: {test_acc}")
        # writer.add_scalars('accuracy', {'train': train_acc, 'test': test_acc}, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/test", test_acc, epoch)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        # writer.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)


if __name__ == "__main__":
    config = get_args()
    train(config)
