"""
Inspired by: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#step-3-create-model-class
"""
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import lightning as L

import torch.nn.functional as F

class LSTMModel(L.LightningModule):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim, seq_dim, dropout_rate):
        super().__init__()
        self.save_hyperparameters()
        # 设为双向LSTM，添加dropout防止过拟合
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_rate, bidirectional=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate) # 添加dropout 层
        self.criterion = nn.CrossEntropyLoss()
        self.preds = []
        self.labels = []

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.hparams.layer_dim * 2, x.size(0), self.hparams.hidden_dim, device=self.device) # *2 for bidirectional
        # Initialize cell state
        c0 = torch.zeros(self.hparams.layer_dim * 2, x.size(0), self.hparams.hidden_dim, device=self.device)
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        #out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5) # 增加权重衰减
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            },
        }

    def step(self, batch, mode="train"):
        x, y = batch
        x = x.view(x.size(0), self.hparams.seq_dim, -1)
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        if mode != "train":
            self.preds.append(y_pred.argmax(dim=1).cpu())
            self.labels.append(y.cpu())
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)

        # Calculate F1 score
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
        self.log(f"Final F1 Score", f1, prog_bar=True)

        # Clear the stored predictions and labels for the next epoch
        self.preds.clear()
        self.labels.clear()
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), self.seq_dim, -1)
        pred = self.forward(x)
        return pred