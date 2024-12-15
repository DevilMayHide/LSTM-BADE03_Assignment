import torch
from model import LSTMModel
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import torch.nn.functional as F

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载测试数据
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 初始化模型
model = LSTMModel(input_dim=28, hidden_dim=128, layer_dim=2, output_dim=10, seq_dim=28, dropout_rate=0.2)

# 加载权重
state_dict = torch.load("Our_Best_model\Our_best_model.pth") # 修改路径
model.load_state_dict(state_dict)
model.eval()

# 测试模型
all_preds = []
all_labels = []
test_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x = x.view(x.size(0), 28, -1)  # 调整输入形状
        outputs = model(x)
        test_loss += F.cross_entropy(outputs, y, reduction='sum').item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

test_loss /= len(test_loader.dataset)
f1 = f1_score(all_labels, all_preds, average="macro")

print(f"Test Loss: {test_loss:.4f}")
print(f"Final F1 Score: {f1:.4f}")
