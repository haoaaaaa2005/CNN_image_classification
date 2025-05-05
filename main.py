import os
import glob
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchmetrics.classification import Precision, Recall, F1Score, MulticlassROC

# ===== Global Hyperparameters and Paths =====
DATA_DIR = r"D:\CNN_image_classification\data\chinese_medicine_dataset"
CLASSES = [
    'dangshen','gouqi','huaihua','jiangcan','niubangzi','tiannanxing','mudanpi','zhuling','gancao','baihe',
    'baibu','zhuye','zhuru','zicao','hongteng','aiye','jingjie','jinyinhua','huangbai','huangqi'
]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
IMAGE_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== Dataset Definition =====
class ChineseMedicineDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.samples = []
        self.transform = transform
        for idx, cls in enumerate(classes):
            folder = os.path.join(root_dir, cls)
            for img_path in glob.glob(os.path.join(folder, "*.jpg")):
                self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ===== Transforms and DataLoader =====
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = ChineseMedicineDataset(DATA_DIR, CLASSES, transform)
total = len(dataset)
if total == 0:
    raise RuntimeError(f"未在 {DATA_DIR} 下找到任何图像，请检查路径和文件名格式。")

train_size = int(TRAIN_RATIO * total)
val_size   = int(VAL_RATIO * total)
test_size  = total - train_size - val_size
train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# ===== Model Definition =====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE[0]//8) * (IMAGE_SIZE[1]//8), 256),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model     = SimpleCNN(NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ===== Metrics =====
precision_metric = Precision(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
recall_metric    = Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
f1_metric        = F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
roc_metric       = MulticlassROC(num_classes=NUM_CLASSES).to(DEVICE)

# ===== Training and Evaluation =====
def train_one_epoch(loader):
    model.train(); running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(); outputs = model(images)
        loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(loader, compute_roc=False):
    model.eval(); all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs  = torch.softmax(model(images), dim=1)
            all_probs.append(probs); all_labels.append(labels)
    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels).to(DEVICE)
    preds      = torch.argmax(all_probs, dim=1)

    acc  = (preds == all_labels).float().mean().item()
    prec = precision_metric(preds, all_labels).item()
    rec  = recall_metric(preds, all_labels).item()
    f1   = f1_metric(preds, all_labels).item()
    roc_data = None
    if compute_roc:
        fpr, tpr, _ = roc_metric(all_probs, all_labels)
        roc_data    = (fpr, tpr)
    return acc, prec, rec, f1, roc_data

# ===== Main Training Loop =====
train_losses, val_metrics = [], []
for epoch in range(1, NUM_EPOCHS+1):
    t_loss = train_one_epoch(train_loader)
    v_acc, v_prec, v_rec, v_f1, _ = evaluate(val_loader)
    train_losses.append(t_loss); val_metrics.append((v_acc, v_prec, v_rec, v_f1))
    print(f"Epoch {epoch}/{NUM_EPOCHS} "
          f"Train Loss:{t_loss:.4f} "
          f"Val Acc:{v_acc:.4f} "
          f"Val Prec:{v_prec:.4f} "
          f"Val Rec:{v_rec:.4f} "
          f"Val F1:{v_f1:.4f}")

# ===== Final Test Evaluation =====
t_acc, t_prec, t_rec, t_f1, roc_data = evaluate(test_loader, compute_roc=True)
print(f"Test Acc:{t_acc:.4f} Prec:{t_prec:.4f} Rec:{t_rec:.4f} F1:{t_f1:.4f}")

# ===== Visualization =====
plt.figure(); plt.plot(range(1,NUM_EPOCHS+1), train_losses, label='Train Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss Curve'); plt.legend(); plt.show()

val_accs, val_precs, val_recs, val_f1s = zip(*val_metrics)
plt.figure();
plt.plot(range(1,NUM_EPOCHS+1), val_accs, label='Val Acc')
plt.plot(range(1,NUM_EPOCHS+1), val_precs, label='Val Prec')
plt.plot(range(1,NUM_EPOCHS+1), val_recs, label='Val Rec')
plt.plot(range(1,NUM_EPOCHS+1), val_f1s, label='Val F1')
plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Metrics'); plt.legend(); plt.show()

if roc_data: #
    fpr, tpr = roc_data
    plt.figure(figsize=(10, 8))
    for idx, name in enumerate(CLASSES):
        plt.plot(fpr[idx].cpu(), tpr[idx].cpu(), label=name)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # 对角线
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right', fontsize='small', ncol=2)  # 可调整图例样式
    plt.grid(True)
    plt.tight_layout()
    plt.show()

