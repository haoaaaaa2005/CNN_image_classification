import os
import glob
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from torchmetrics.classification import Precision, Recall, F1Score, MulticlassROC

# ===== Global Hyperparameters and Paths =====
DATA_DIR = r"D:\CNN_image_classification\data\chinese_medicine_dataset"
CLASSES = [
    'dangshen','gouqi','huaihua','jiangcan','niubangzi','tiannanxing',
    'mudanpi','zhuling','gancao','baihe','baibu','zhuye','zhuru','zicao',
    'hongteng','aiye','jingjie','jinyinhua','huangbai','huangqi'
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
PATIENCE = 5  # Early stopping patience
MODEL_SAVE_PATH = 'best_resnet50.pth'

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
if len(dataset) == 0:
    raise RuntimeError(f"No images found in {DATA_DIR}. Please check the path.")

total = len(dataset)
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

# ===== Model Definition: ResNet50 =====
model = models.resnet50(pretrained=True)
# Replace the final fully-connected layer
def set_resnet_classifier(model, num_classes):
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

model = set_resnet_classifier(model, NUM_CLASSES).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ===== Metrics =====
precision_metric = Precision(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
recall_metric    = Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
f1_metric        = F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(DEVICE)
roc_metric       = MulticlassROC(num_classes=NUM_CLASSES).to(DEVICE)

# ===== Training and Evaluation =====
def train_one_epoch(loader):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(loader, compute_roc=False):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs  = torch.softmax(model(images), dim=1)
            all_probs.append(probs)
            all_labels.append(labels)
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

# ===== Main Training Loop with Early Stopping =====
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []
val_metrics = []

for epoch in range(1, NUM_EPOCHS+1):
    t_loss = train_one_epoch(train_loader)
    train_losses.append(t_loss)

    # Compute validation loss separately
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
    v_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(v_loss)

    v_acc, v_prec, v_rec, v_f1, _ = evaluate(val_loader)
    val_metrics.append((v_acc, v_prec, v_rec, v_f1))

    print(f"Epoch {epoch}/{NUM_EPOCHS} "
          f"Train Loss: {t_loss:.4f} "
          f"Val Loss: {v_loss:.4f} "
          f"Val Acc: {v_acc:.4f} "
          f"Val F1: {v_f1:.4f}")

    # Early Stopping & Save Best Model
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  > New best model saved with Val Loss: {v_loss:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered. No improvement in {PATIENCE} epochs.")
            break

# ===== Load Best Model and Final Test Evaluation =====
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
t_acc, t_prec, t_rec, t_f1, roc_data = evaluate(test_loader, compute_roc=True)
print(f"Test Acc: {t_acc:.4f} Prec: {t_prec:.4f} Rec: {t_rec:.4f} F1: {t_f1:.4f}")

# ===== Visualization =====
plt.figure(); plt.plot(range(1,len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1,len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
plt.legend(); plt.show()

val_accs, val_precs, val_recs, val_f1s = zip(*val_metrics)
plt.figure();
plt.plot(range(1,len(val_accs)+1), val_accs, label='Val Acc')
plt.plot(range(1,len(val_precs)+1), val_precs, label='Val Prec')
plt.plot(range(1,len(val_recs)+1), val_recs, label='Val Rec')
plt.plot(range(1,len(val_f1s)+1), val_f1s, label='Val F1')
plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Metrics')
plt.legend(); plt.show()

if roc_data:
    fpr, tpr = roc_data
    plt.figure(figsize=(10, 8))
    for idx, name in enumerate(CLASSES):
        plt.plot(fpr[idx].cpu(), tpr[idx].cpu(), label=name)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
