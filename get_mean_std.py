import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# 自定义数据集类，用于加载特定文件名格式的图片
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 获取所有符合dangshen_1.jpg到dangshen_75.jpg的图片
        self.image_files = [f"dangshen_{i}.jpg" for i in range(1, 76) if os.path.exists(os.path.join(self.root_dir, f"dangshen_{i}.jpg"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 确保所有图片都是RGB格式

        if self.transform:
            image = self.transform(image)

        return image

# 定义数据预处理转换，将图片转换为张量
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建自定义数据集和数据加载器
dataset_path = r'D:\CNN_image_classification\data\chinese_medicine_dataset\dangshen'  # Windows路径
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 计算均值和标准差
mean = torch.zeros(3)
std = torch.zeros(3)
num_images = 0

for images in dataloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    num_images += batch_samples

mean /= num_images
std /= num_images

print(f"Mean: {mean}")
print(f"Std: {std}")

#Mean: tensor([0.6823, 0.5950, 0.5036])
#Std: tensor([0.2168, 0.2449, 0.2687])