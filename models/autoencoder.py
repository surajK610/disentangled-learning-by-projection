import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):

    encoder.train()
    decoder.train()
    train_loss = []
    count = 0

    for image_batch, labels in dataloader: 
        image_batch = image_batch.to(device)
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)
        loss = loss_fn(decoded_data, image_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = count + 1
        
        if count % 100 == 0: 
            print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad(): 
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)

            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())

        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 

        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs(encoder,decoder, test_dataset, device, idx, n=5, fig_path='outputs/ae_outputs'):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = torch.tensor(test_dataset[i][0]).unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        
        with torch.no_grad():
            rec_img  = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy().transpose((2,1,0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Original images')
            
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy().transpose((2,1,0)))  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        
        if i == n//2:
            ax.set_title('Reconstructed images')
    plt.savefig(os.path.join(fig_path, f'reconstructed_{idx}.png'))

def linearly_interpolate(decoder, device, point1, point2, point3, point4,
                        n=5, fig_path='outputs/interpolation.png'):
    plt.figure(figsize=(8,8))
    
    def subprocess(a, b, idx_fun):
        difference = b - a
        jump_size = difference/(n+1)

        for i in range(0, n+2):
            plt.subplot(n+2, n+2, idx_fun(i, n))
            interpolate_coord = a + jump_size * i
            img = decoder(interpolate_coord.to(device))
            plt.imshow(img.detach().cpu().numpy().squeeze(0).transpose(2, 1, 0))
        
    subprocess(point1, point2, lambda i, n: (i)*(n+2) + 1)
    subprocess(point2, point3, lambda i, n: (n+2)*(n+1)+i+1)
    subprocess(point4, point3, lambda i, n: (i+1)*(n+2))
    subprocess(point1, point4, lambda i, n: i+1)

    plt.savefig(fig_path)
    
def linearly_interpolate_test_idxs(encoder, decoder, test_dataset, device, i1, i2, i3, i4, 
                                n=5, fig_path='outputs/interpolation.png'):
    img1 = torch.from_numpy(test_dataset[i1][0]).unsqueeze(0).to(device)
    encoded1 = encoder(img1)
    img2 = torch.from_numpy(test_dataset[i2][0]).unsqueeze(0).to(device)
    encoded2 = encoder(img2)
    img3 = torch.from_numpy(test_dataset[i3][0]).unsqueeze(0).to(device)
    encoded3 = encoder(img3)
    img4 = torch.from_numpy(test_dataset[i4][0]).unsqueeze(0).to(device)
    encoded4 = encoder(img4)
    linearly_interpolate(decoder, device, encoded1, encoded2, encoded3, encoded4, n=n, fig_path=fig_path)