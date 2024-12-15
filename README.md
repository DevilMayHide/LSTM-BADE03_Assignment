<div align="center">
<h1>LSTM-BADE03_Assignment</h1>
LSTM Model Optimization on MNIST Dataset
<br>
<br>
  
</div>


## About
This project focuses on optimizing a baseline LSTM model provided by our instructor **Davison Wang**. We modified the network architecture and parameters to achieve significant performance improvements on the MNIST dataset. 🚀

**Our Final Best Results🎉**:
- **Final F1 Score**: `0.9931`
- **Test Loss**: `0.0216`

### Group Members
- **RUIHAO ZHANG🤠**
  - Faculty: FIT
  - Major: Computer Science
- **HAIXIN YI🕵️‍♂️**
  - Faculty: FIT
  - Major: Software Engineering
- **BENHUANG LIU🧑‍💻**
  - Faculty: FIT
  - Major: Computer Science
- **WENQIAN XU👨‍🎓**
  - Faculty: FIT
  - Major: Software Engineering

---

## Requirements
The following environment setup is necessary to replicate our results:

- torch==2.5.1+cu124
- numpy
- matplotlib
- conda
- scikit-learn
- lightning==2.4.0

---

## Quick Installation
First, you may use `git` command to clone our project to your local computer.

`git clone https://github.com/DevilMayHide/LSTM-BADE03_Assignment.git`

Follow these steps to set up the environment and install the required dependencies:
```bash
conda env create -f environment.yml LSTM
conda activate LSTM
```

---

## Manual Installation
You can try the following instructions to manually configure the environment if you encounter errors when trying to **Quick Install** the environment.
```bash
conda create -n lstm python=3.11
conda activate lstm
pip install -r requirements.txt # We provide the requirements.txt file to help you quickly configure dependencies
```
If you still find some errors, then you may follow the instructions below to configure your environment step-by-step. Dont' be lazy.😜
```bash
conda create -n lstm python=3.11
conda activate lstm
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pytorch-lightning
pip install lightning
pip install matplotlib
pip install numpy
pip install scikit-learn
```

---

## Train 
To train the model from scratch using the provided code and data:
```bash
python main.py
```
This will train the LSTM model on the MNIST dataset and save the best checkpoints in the checkpoints directory.

---

## Test
To test the model using a pre-trained checkpoint and reproduce the results:
```bash
python main.py --eval checkpoints/xxx
```
Replace `XXX` with your trained **best weights file** `.ckpt`.💪

---

## Replication of ours results
To reproduce our results, you can try the following steps:
```bash
python main.py --eval Our_Best_model\lstm-epoch=24-val_loss=0.03.ckpt
```
For this project we are using the `PyTorch Lightning` framework, so we store information such as weights, optimiser state, etc. as `.ckpt` format files, and you can find the best model we have trained in the folder `Our_Best_model`.✊

We also provide a `.pth` file to help those who are not using `PyTorch Lightning`. It can also be found in the `Our_Best_model` folder, and you can run `test_pth.py` to reproduce our results🤞.

---

## Architecture Enhancements and Performance Improvements
We made several critical modifications to the original LSTM model to achieve **significant performance gains**👏. The following summarizes our changes:
1. **Added a Bidirectional LSTM**:
   - Improved the model’s ability to understand context in both forward and backward directions.
   - Enhanced feature extraction from sequential data.
  
2. **Introduced Dropout Layers**:
   - Added Dropout to regularise the model to prevent overfitting.
     ```python
     """model.py"
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
     ```
 3. **Optimized Network Parameters**:
    - Increased the hidden dimension to `128` and used `2` LSTM layers to boost the model’s capacity.
      ```python
      """main.py"""
      LSTM = LSTMModel(input_dim=28, hidden_dim=128, layer_dim=2, output_dim=10, seq_dim=28, dropout_rate=0.2)
      ```
 4. **Implemented Early Stopping**:
    - Early stopping prevents unnecessary training and helps to avoid overfitting.
      ```python
      """main.py"""
      # 增加早停机制, 提升训练效率
      early_stop = EarlyStopping(
          monitor="val_loss",
          patience=10,
          verbose=True,
          mode="min"
      )
      ```
 5. **Improved Data Augmentation**:
    - Used `RandomRotation` and `RandomAffine` transformations to enhance the diversity of the training data, making the model more robust.
      ```python
      """main.py"""
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
      ```
    You may be wondering `Normalize((0.1307,), (0.3081,))` where the two figures here come from😮.
    
    That's because we wrote a program to calculate the **mean and standard deviation** of all the pixel values in the entire dataset for the **MNIST dataset**.
    
    You may check this file and run it `calculate_MINIST.py`👈. 
   
 6. **Switched to AdamW Optimizer**:
    - Improved convergence speed and stability by incorporating weight decay.
    - Learning rate scheduler (ReduceLROnPlateau): Dynamically adjusts the learning rate according to the validation loss to prevent training stagnation.
      ```python
      """model.py"""
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
      ```
  These enhancements collectively contributed to achieving a **Final F1 Score** of `0.9931` and a **Test Loss** of `0.0216`, representing significant performance improvements over the baseline model🙌.

  ---

  ## References
  1. https://zh.d2l.ai/chapter_preface/index.html
  2. https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#steps_3
  3. https://www.sciencedirect.com/science/article/pii/S0893608021003439

          
      
