"""
train_ascii_cnn.py
Treina uma CNN pequena para classificar blocos 8x8 (intensidade + DoG/Sobel)
nos conjuntos de 8, 16 ou 32 caracteres.

Requer: pip install torch numpy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------

class AsciiBlockDataset(Dataset):
    def __init__(self, bin_path="dataset.bin"):
        raw = np.fromfile(bin_path, dtype=np.uint8).reshape(-1, 1 + 64 + 64)
        self.labels = raw[:, 0].astype(np.int64)
        intensity = raw[:, 1:65].astype(np.float32).reshape(-1, 8, 8) / 255.0
        edge = raw[:, 65:129].astype(np.float32).reshape(-1, 8, 8) / 255.0
        self.x = np.stack([intensity, edge], axis=1)  # (N, 2, 8, 8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), self.labels[idx]


def load_label_names(path="labels.txt"):
    with open(path, encoding="utf-8") as f:
        n = int(f.readline())
        names = {}
        for _ in range(n):
            line = f.readline().rstrip("\n")
            idx_str, ch = line.split(" ", 1)
            names[int(idx_str)] = ch
    return names


# ---------------------------------------------------------------------
# 2. Modelo: CNN pequena (entrada 2x8x8)
# ---------------------------------------------------------------------

class AsciiCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 2x8x8 -> 16x8x8
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 16x8x8 -> 32x8x8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # -> 32x4x4

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> 64x4x4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------
# 3. Treino
# ---------------------------------------------------------------------

def train(bin_path="dataset.bin", labels_path="labels.txt",
          epochs=30, batch_size=128, lr=1e-3, val_split=0.15):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    dataset = AsciiBlockDataset(bin_path)
    names = load_label_names(labels_path)
    num_classes = len(names)
    print(f"Dataset: {len(dataset)} amostras, {num_classes} classes")

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # shuffle=True aqui é o que resolve o dataset.bin não estar embaralhado
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AsciiCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validação
        model.eval()
        val_correct, val_total = 0, 0
        confusion_errors = {}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                val_correct += (pred == y).sum().item()
                val_total += x.size(0)

                # rastreia pares confundidos (útil pra você ver se seus
                # pares "delicados" como . / · / : estão se confundindo)
                wrong = pred != y
                for p, t in zip(pred[wrong].tolist(), y[wrong].tolist()):
                    key = (names[t], names[p])
                    confusion_errors[key] = confusion_errors.get(key, 0) + 1

        val_acc = val_correct / val_total

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nMelhor acurácia de validação: {best_val_acc:.4f}")
    print("Modelo salvo em: best_model.pt")

    print("\nPares mais confundidos (real -> previsto):")
    top_confusions = sorted(confusion_errors.items(), key=lambda kv: -kv[1])[:10]
    for (real, pred), count in top_confusions:
        print(f"  '{real}' -> '{pred}': {count}x")


if __name__ == "__main__":
    train()
