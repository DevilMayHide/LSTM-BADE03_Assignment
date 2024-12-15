from model import LSTMModel
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import lightning as L

def main():
    # 数据增强和进行预处理
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # 归一化处理
    ])
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    
    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) # shuffle=True，可以提高鲁棒性
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2, pin_memory=True) # 提高主机内存到GPU内存的传输
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    # automatically save the best model
    checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints", 
            filename="lstm-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1, 
            mode="min", 
        )
    
    # 增加早停机制, 提升训练效率
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min"
    )
    
    # load model from checkpoint and evaluate
    if(args.eval != ""):
        trainer = L.Trainer(devices=1, num_nodes=1)
        LSTM = LSTMModel.load_from_checkpoint(args.eval)
        trainer.test(model=LSTM, dataloaders=test_loader)
    
    # train model
    else:
        LSTM = LSTMModel(input_dim=28, hidden_dim=128, layer_dim=2, output_dim=10, seq_dim=28, dropout_rate=0.2)
        
        trainer = L.Trainer(
            max_epochs=35, 
            log_every_n_steps=50,
            callbacks=[checkpoint_callback, early_stop],
            default_root_dir="checkpoints",
            devices=1,          
            accelerator='gpu')
        
        trainer.fit(model=LSTM, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.test(model=LSTM, dataloaders=test_loader)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval", type=str, default="", help="Path to model to evaluate")
    args = parser.parse_args()
    main()