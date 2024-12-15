import torch
from model import LSTMModel

# 加载 .ckpt 文件
ckpt_path = "Our_Best_model\lstm-epoch=24-val_loss=0.03.ckpt"
checkpoint = torch.load(ckpt_path)

# 提取模型权重
state_dict = checkpoint["state_dict"]

# 保存为 .pth 文件
torch.save(state_dict, "Our_best.model.pth")
