import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np

class LogisticClassifier(nn.Module):
    def __init__(self, input_features, classes, device, lr=0.001):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(input_features, classes).to(self.device)

        self.optimizer = torch.optim.AdamW(self.linear.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.linear(x)
        return x

    def acc(self, y_pred, target):
        return torch.mean(torch.eq(y_pred, target).float()).item()

    def fit(self, dataloader, val_dataloader=None, encoder=None, label='number', epochs=50, evaluate_every=1):
        all_losses = []
        val_losses = []
        pbar = tqdm(range(int(epochs)), desc='Training Epochs')
        val_loss = np.nan
        val_acc = np.nan
        for i in pbar:
            for image_batch, labels in dataloader:
                image_batch = image_batch.to(self.device)
                if encoder is not None: 
                    image_batch = encoder(image_batch)
                    
                labels = labels[label]
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                y_pred = self.forward(image_batch)
                loss = self.loss_fn(y_pred, labels)
                loss.backward()
                self.optimizer.step()
                all_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item(), val_loss=val_loss, val_acc=val_acc)
            if i % evaluate_every == 0 and val_dataloader is not None:
                acc_batches = []
                loss_batches = []
                for image_batch, labels in val_dataloader:
                    image_batch = image_batch.to(self.device)
                    if encoder is not None: 
                        image_batch = encoder(image_batch)

                    labels = labels[label]
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        y_pred = self.forward(image_batch)
                        val_loss = self.loss_fn(y_pred, labels)
                        val_acc = self.acc(y_pred.argmax(dim=1), labels)
                    loss_batches.append(val_loss.item())
                    acc_batches.append(val_acc)
                val_loss = np.mean(loss_batches)
                val_acc = np.mean(acc_batches)
                
                val_losses.append(val_loss)
                pbar.set_postfix(loss=loss.item(), val_loss=val_loss, val_acc=val_acc)
                pbar.refresh(0.01)
        return all_losses, val_losses, val_acc