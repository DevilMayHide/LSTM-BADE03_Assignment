# 计算均值和标准差
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch

train_data = MNIST(root="MNIST", train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))
train_features, _ = next(iter(train_loader))
mean = train_features.mean().item()  # 均值
std = train_features.std().item()    # 标准差
print(mean, std)

# 得到的输出: 0.1307, 0.3081(保留四位小数)
