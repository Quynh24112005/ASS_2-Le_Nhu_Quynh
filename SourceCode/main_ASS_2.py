import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

def get_data_loaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_len = int(0.8 * len(full_train))
    val_len = len(full_train) - train_len
    train_set, val_set = random_split(full_train, [train_len, val_len])

    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_set, batch_size=BATCH_SIZE),
        DataLoader(test_set, batch_size=BATCH_SIZE)
    )

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

#Training
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    acc = 100 * correct / total
    return avg_loss, acc

def evaluate(model, loader, loss_fn=None, return_preds=False):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    true, pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(dim=1)

            if loss_fn:
                loss_sum += loss_fn(out, y).item() * x.size(0)
                correct += (preds == y).sum().item()
                total += y.size(0)

            if return_preds:
                true.extend(y.cpu().numpy())
                pred.extend(preds.cpu().numpy())

    if loss_fn:
        return loss_sum / total, 100 * correct / total
    return true, pred

def plot_curves(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Tổn thất huấn luyện')
    plt.plot(epochs, val_loss, label='Tổn thất xác thực')
    plt.xlabel('Epochs')
    plt.ylabel('Tổn thất')
    plt.title('Đường cong tổn thất (CNN)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Độ chính xác huấn luyện')
    plt.plot(epochs, val_acc, label='Độ chính xác xác thực')
    plt.xlabel('Epochs')
    plt.ylabel('Độ chính xác (%)')
    plt.title('Đường cong độ chính xác (CNN)')
    plt.legend()
    plt.grid(True)

    plt.suptitle("Đường cong học tập cho CNN")
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=300)
    plt.show()


def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data_loaders()
    model = ConvNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_loss_vals, val_loss_vals = [], []
    train_acc_vals, val_acc_vals = [], []

    best_val_loss = float('inf')
    patience = 5
    counter = 0

    with open("training_log.txt", "w") as log_file:
        log_file.write("Epoch\tTrainLoss\tTrainAcc\tValLoss\tValAcc\n")

        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            train_loss_vals.append(tr_loss)
            val_loss_vals.append(val_loss)
            train_acc_vals.append(tr_acc)
            val_acc_vals.append(val_acc)

            log_line = f"{epoch+1}\t{tr_loss:.4f}\t{tr_acc:.2f}\t{val_loss:.4f}\t{val_acc:.2f}\n"
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}% - Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            log_file.write(log_line)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    log_file.write("Early stopping triggered.\n")
                    print("Early stopping triggered.")
                    break

    model.load_state_dict(torch.load("best_model.pt"))

    y_true, y_pred = evaluate(model, test_loader, return_preds=True)
    test_acc = 100 * np.mean((np.array(y_true) == np.array(y_pred)).astype(np.float32))
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    with open("training_log.txt", "a") as log_file:
        log_file.write(f"\nTest Accuracy: {test_acc:.2f}%\n")

    plot_conf_matrix(y_true, y_pred)
    plot_curves(train_loss_vals, val_loss_vals, train_acc_vals, val_acc_vals)
