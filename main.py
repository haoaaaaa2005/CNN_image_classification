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
import gradio as gr  # 引入 Gradio
from io import BytesIO
from torchmetrics.classification import Precision, Recall, F1Score, MulticlassROC

# ===== Global Hyperparameters and Paths =====
DATA_DIR = r"D:\CNN_image_classification\data\chinese_medicine_dataset"
CLASSES = [
    'dangshen','gouqi','huaihua','jiangcan','niubangzi','tiannanxing',
    'mudanpi','zhuling','gancao','baihe','baibu','zhuye','zhuru','zicao',
    'hongteng','aiye','jingjie','jinyinhua','huangbai','huangqi'
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== Available Model Types =====
MODEL_TYPES = ['SimpleCNN', 'ResNet18', 'ResNet50', 'EfficientNet_B1', 'AlexNet']

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

# ===== Model Factory =====
def get_model(model_type, num_classes):
    if model_type == 'SimpleCNN':
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * (IMAGE_SIZE[0]//8) * (IMAGE_SIZE[1]//8), 256),
                    nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            def forward(self, x): return self.classifier(self.features(x))
        return SimpleCNN(num_classes)
    else:
        name_map = {'ResNet18':'resnet18','ResNet50':'resnet50','EfficientNet_B1':'efficientnet_b1','AlexNet':'alexnet'}
        name = name_map.get(model_type)
        model = getattr(models, name)(pretrained=True)
        if model_type == 'AlexNet':
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        elif model_type == 'EfficientNet_B1':
            in_f = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_f, num_classes)
        else:
            in_f = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
        return model

# ===== Main function for Gradio =====
def train_and_plot(batch_size, lr, epochs, train_ratio, val_ratio, test_ratio,
                   image_size, patience,
                   mean_r, mean_g, mean_b,
                   std_r, std_g, std_b,
                   model_type):
    global IMAGE_SIZE
    IMAGE_SIZE = (int(image_size), int(image_size))
    NORMALIZE_MEAN = [float(mean_r), float(mean_g), float(mean_b)]
    NORMALIZE_STD  = [float(std_r),  float(std_g),  float(std_b)]
    MODEL_SAVE_PATH = 'best_model.pth'

    # 数据准备
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    dataset = ChineseMedicineDataset(DATA_DIR, CLASSES, transform)
    total = len(dataset)
    tr, vr = float(train_ratio), float(val_ratio)
    train_size = int(tr * total)
    val_size   = int(vr * total)
    test_size  = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
                                                 generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=int(batch_size), shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=int(batch_size), shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=int(batch_size), shuffle=False)

    # 模型、优化器、损失
    device = DEVICE
    model = get_model(model_type, NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss()

    # Early Stopping
    best_val_loss = float('inf'); patience_counter = 0

    # 训练与验证
    train_losses, val_losses = [], []
    for epoch in range(1, int(epochs) + 1):
        model.train(); running_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(); out = model(imgs)
            loss = criterion(out, lbls); loss.backward(); optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_losses.append(running_loss / len(train_loader.dataset))

        model.eval(); val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                val_loss += criterion(model(imgs), lbls).item() * imgs.size(0)
        val_losses.append(val_loss / len(val_loader.dataset))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(patience):
                break

    # 绘图
    buf = BytesIO()
    plt.figure(); plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss'); plt.legend()
    plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

# ===== Gradio UI =====
with gr.Blocks() as demo:
    gr.Markdown("## 中药图像分类模型训练与可视化")
    with gr.Row():
        with gr.Column():
            model_box = gr.Dropdown(choices=MODEL_TYPES, value='ResNet50', label='Model Type')
            batch_box = gr.Slider(1, 128, value=32, label='Batch Size')
            lr_box    = gr.Number(value=1e-3, label='Learning Rate')
            epochs_box= gr.Slider(1, 100, value=20, step=1, label='Num Epochs')
            train_r   = gr.Slider(0.1, 0.9, value=0.7, step=0.05, label='Train Ratio')
            val_r     = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label='Val Ratio')
            test_r    = gr.Slider(0.05, 0.5, value=0.1, step=0.05, label='Test Ratio')
            img_size  = gr.Slider(64, 512, value=224, step=32, label='Image Size')
            patience  = gr.Slider(1, 20, value=5, step=1, label='Early Stopping Patience')
            # 一级菜单展示但不调整
            with gr.Accordion("Normalize Parameters", open=True):
                gr.Markdown("**Current Mean:** R={0}, G={1}, B={2}".format(0.485,0.456,0.406))
                gr.Markdown("**Current Std:** R={0}, G={1}, B={2}".format(0.229,0.224,0.225))
            # 二级菜单可调整
            with gr.Accordion("Adjust Normalize Mean/Std", open=False):
                mean_r    = gr.Slider(0.0, 1.0, value=0.485, step=0.001, label='Mean R')
                mean_g    = gr.Slider(0.0, 1.0, value=0.456, step=0.001, label='Mean G')
                mean_b    = gr.Slider(0.0, 1.0, value=0.406, step=0.001, label='Mean B')
                std_r     = gr.Slider(0.0, 1.0, value=0.229, step=0.001, label='Std R')
                std_g     = gr.Slider(0.0, 1.0, value=0.224, step=0.001, label='Std G')
                std_b     = gr.Slider(0.0, 1.0, value=0.225, step=0.001, label='Std B')
            run_btn = gr.Button('Run Training', loading='Running')  # Changed: 点击按钮后显示 'Running'
        with gr.Column():
            output_img = gr.Image(type='pil', label='Loss Curve')
    run_btn.click(fn=train_and_plot,
                  inputs=[batch_box, lr_box, epochs_box, train_r, val_r, test_r,
                          img_size, patience,
                          mean_r, mean_g, mean_b,
                          std_r, std_g, std_b,
                          model_box],
                  outputs=output_img)

    demo.launch()  # 启动 Gradio UI
