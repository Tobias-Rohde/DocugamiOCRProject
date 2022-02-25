import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from data import CharacterDataset
import tqdm


# Simple CNN with three conv blocks (conv,maxpool,batchnorm,relu)
class CharacterClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 16
        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1)
        self.bs1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels // 2, 3, 1, 1)
        self.bs2 = nn.BatchNorm2d(channels // 2)
        self.conv3 = nn.Conv2d(channels // 2, channels, 3, 1, 1)
        self.output = nn.Linear(channels * 4 * 4, 2)

    def forward(self, x):
        bsz = x.shape[0]

        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bs1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bs2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = x.view(bsz, -1)

        return self.output(x)


def compute_tp_fp_fn(preds, y):
    return torch.logical_and(preds == 1, y == 1).sum(dim=0).cpu(),\
           torch.logical_and(preds == 1, y == 0).sum(dim=0).cpu(),\
           torch.logical_and(preds == 0, y == 1).sum(dim=0).cpu()


def main():
    train_data = CharacterDataset(10000, 0)
    val_data = CharacterDataset(1000, 10000)
    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)

    device = torch.device("cuda:0")
    model = CharacterClassifier().to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    epoch_range = tqdm.trange(100)
    for _ in epoch_range:
        total_loss = 0
        num_batches = 0

        train_tp, train_fp, train_fn = 0, 0, 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            preds = logits >= 0

            tp, fp, fn = compute_tp_fp_fn(preds, y)
            train_tp += tp
            train_fp += fp
            train_fn += fn

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().item()
            num_batches += 1
        train_p = train_tp / (train_tp + train_fp)
        train_r = train_tp / (train_tp + train_fn)
        train_f1 = 2 * (train_p * train_r) / (train_p + train_r)

        # Validation: compute precision, recall and f1
        val_tp, val_fp, val_fn = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                preds = logits >= 0

                tp, fp, fn = compute_tp_fp_fn(preds, y)
                val_tp += tp
                val_fp += fp
                val_fn += fn
        val_p = val_tp / (val_tp + val_fp)
        val_r = val_tp / (val_tp + val_fn)
        val_f1 = 2 * (val_p * val_r) / (val_p + val_r)

        epoch_range.set_description(f"Loss: {total_loss / num_batches:.4f} "
                                    f"Train Bold: {train_p[0]:.4f},{train_r[0]:.4f},{train_f1[0]:.4f} Train Italic: {train_p[1]:.4f},{train_r[1]:.4f},{train_f1[1]:.4f} "
                                    f"Val Bold: {val_p[0]:.4f},{val_r[0]:.4f},{val_f1[0]:.4f} Val Italic: {val_p[1]:.4f},{val_r[1]:.4f},{val_f1[1]:.4f}")

    torch.save(model, "model.pt")


if __name__ == "__main__":
    main()
