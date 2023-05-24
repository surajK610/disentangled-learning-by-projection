import torch.nn as nn
import torch
import numpy as np

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        ### Convolutional section
        self.cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            
            nn.LeakyReLU(True),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

        )
        
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.lin = nn.Sequential(

            nn.Linear(64, 128),
            nn.ReLU(True),

            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Apply convolutions
        #print("cnn input dim:", x.shape)
        x = self.cnn(x)
#         print("cnn output dim", x.shape)
        # Flatten
        x = self.flatten(x)
#         print("flatten shape", x.shape)
        # # Apply linear layers
        x = self.lin(x)
        return x


def train_epoch_classifer(classifier, device, dataloader, loss_fn_class, optimizer_class, label='number', encoder=None):
    
    classifier.train()
    train_loss = []
    count = 0

    for image_batch, labels in dataloader:
        labels = labels[label]
        image_batch = image_batch.to(device)
        if encoder is not None:
            image_batch = encoder(image_batch)
        labels = labels.to(device)

        predictions = classifier(image_batch)
        loss = loss_fn_class(predictions, labels)
        
        optimizer_class.zero_grad()
        loss.backward()
        optimizer_class.step()
        
        count = count + 1

        if count % 100 == 0: 
            print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch_classifier(classifier, device, dataloader, loss_fn_class, label='number', encoder=None):

    classifier.eval()
    with torch.no_grad(): 
        
        conc_out = []
        conc_label = []
        for image_batch, labels in dataloader:
            labels = labels[label]
            image_batch = image_batch.to(device)
            if encoder is not None:
                image_batch = encoder(image_batch)
            predictions = classifier(image_batch)
            
            conc_out.append(predictions.cpu())
            conc_label.append(labels)

        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 

        val_loss = loss_fn_class(conc_out, conc_label)
        val_accuracy = np.mean(conc_out.argmax(dim=1).numpy() == conc_label.numpy())
    return val_loss.data, val_accuracy

def setup_classifier(num_units, device, lr=0.0001, weight_decay=1e-05):

    classifier = Classifier(num_units)
    loss_fn_class = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    classifier.to(device)

    return classifier, loss_fn_class, optimizer
