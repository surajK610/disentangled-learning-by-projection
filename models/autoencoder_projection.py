import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.autoencoder import Encoder, Decoder
from interventions.rlace_torch import rlace
from tqdm import tqdm 
from dataset.dataset_utils import get_data_loaders, ColoredMNIST

class RLACE_AE:
  def __init__(self, test_dataset, rank=1, device="cuda:0", out_iters=75000, in_iters_adv=1, in_iters_clf=1, epsilon=0.0015, batch_size=128, 
              evaluate_every=1000, labels_1=3, labels_2=10, d=4):
    """
    :param X: The input (np array)
    :param Y: the lables (np array)
    :param X_dev: Dev set (np array)
    :param Y_dev: Dev labels (np array)
    :param rank: Number of dimensions to neutralize from the input.
    :param device:
    :param out_iters: Number of batches to run
    :param in_iters_adv: number of iterations for adversary's optimization
    :param in_iters_clf: number of iterations from the predictor's optimization
    :param epsilon: stopping criterion .Stops if abs(acc - majority) < epsilon.
    :param batch_size:
    :param evalaute_every: After how many batches to evaluate the current adversary.
    :return:
    """
    
    assert (d%2 == 0)
    
    self.rank = rank
    self.device = device
    self.out_iters = out_iters
    self.in_iters_adv = in_iters_adv
    self.in_iters_clf = in_iters_clf
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.evaluate_every = evaluate_every
    
    self.optimizer_class = torch.optim.AdamW
    self.optimizer_params = {'lr':0.005, 'weight_decay':1e-4}
    
    self.encoder = Encoder(encoded_space_dim=d).to(device)
    self.decoder = Decoder(encoded_space_dim=d).to(device)
    
    self.params_to_optimize = [
        {'params': self.encoder.parameters()},
        {'params': self.decoder.parameters()},
    ]
    self.test_dataset = test_dataset


  def solve_adv_game(self, d, dataloader, train_dataset, valid_loader, o_epochs=100, a_epochs=10, evaluate_every=1):
      
    self.P1 = 1e-1*torch.randn(int(d/2), int(d/2)).to(self.device)
    self.P2 = 1e-1*torch.randn(int(d/2), int(d/2)).to(self.device)
    
    
    
    optimizer_autoencoder = self.optimizer_class(self.params_to_optimize, **self.optimizer_params)

    self.train_ae_losses = []
    
    n = 20000
    dataloader_rlace = DataLoader(train_dataset, batch_size=n, shuffle=True)
    
    for o_step in tqdm(range(o_epochs)):
      self.P1.requires_grad = True
      self.P2.requires_grad = True
      # train autoencoder for n steps
      
      for a_step in tqdm(range(a_epochs)):
          train_ae_loss = train_epoch_with_projection(self.encoder, self.decoder, self.P1, self.P2, self.device, 
                                                      dataloader, torch.nn.MSELoss(), optimizer_autoencoder, d=d)
          
          self.train_ae_losses.append(train_ae_loss)
          plot_ae_outputs(self.encoder,self.decoder, self.test_dataset, self.device, self.P1, self.P2, d=d, n=5)
      
      
      subset = next(iter(dataloader_rlace))
      
      X = self.encoder(subset[0].to(self.device)).detach()
      X1 = X[:,:int(d/2)]
      X2 = X[:,int(d/2):]
      Y1 = subset[1]['color'].to(self.device)
      Y2 = subset[1]['number'].to(self.device)
      
      rlace_output1 = rlace(X1, Y1, n, rank=1)
      rlace_output2 = rlace(X2, Y2, n, rank=1)
      
      
      self.P1 = rlace_output1.best_P
      self.P2 = rlace_output2.best_P
      
      plot_ae_outputs(self.encoder,self.decoder, self.P1, self.P2, d=d, n=5)
      
    return self.prepare_output() 

  def prepare_output(self):
    return {'P1':self.P1, 'P2':self.P2, 'encoder':self.encoder, 'decoder':self.decoder, 'train_losses_ae': self.train_ae_losses}

def plot_ae_outputs(encoder,decoder, test_dataset, device, p1, p2, d, n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = torch.tensor(test_dataset[i][0]).unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        
        I = torch.eye(int(d/2)).to(device)
        with torch.no_grad():
            encoded_data = encoder(img)
            latent_space_1 = encoded_data[:,:int(d/2)]
            latent_space_2 = encoded_data[:,int(d/2):]

            projected_1 = latent_space_1 @ (I - p1)
            projected_2 = latent_space_2 @ (I - p2)
            
            projected_latent_space = torch.cat((projected_1, projected_2), dim=1)
            
            rec_img  = decoder(projected_latent_space)
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
    plt.show()   
    

def train_epoch_with_projection(encoder, decoder, p1, p2, device, dataloader, loss_fn, optimizer, d):
    p1.requires_grad = False
    p2.requires_grad = False
    
    encoder.train()
    decoder.train()
    train_loss = []
    count = 0
    I = torch.eye(int(d/2)).to(device)
    for image_batch, _ in dataloader: 
        image_batch = image_batch.to(device)
        encoded_data = encoder(image_batch)
        
        # should be a batch x 4 matrix. We want two batch x 2 matrices
        latent_space_1 = encoded_data[:,:int(d/2)]
        latent_space_2 = encoded_data[:,int(d/2):]
        
        projected_1 = latent_space_1 @ (I - p1)
        projected_2 = latent_space_2 @ (I - p2)
#         else:
#             projected_1 = latent_space_1 @ (p1)
#             projected_2 = latent_space_2 @ (p2)
        
        projected_latent_space = torch.cat((projected_1, projected_2), dim=1)
        
        decoded_data = decoder(projected_latent_space)
        loss = loss_fn(decoded_data, image_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = count + 1
        
        if count % 100 == 0: 
            print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def encoder_with_projection(output, device, batch, dim):
  encoded_data = output['encoder'](batch)
  
  I = torch.eye(int(dim/2)).to(device)
  latent_space_1 = encoded_data[:,:int(dim/2)]
  latent_space_2 = encoded_data[:,int(dim/2):]
      
  projected_1 = latent_space_1 @ (I - output['P1'])
  projected_2 = latent_space_2 @ (I - output['P2'])

  projected_latent_space = torch.cat((projected_1, projected_2), dim=1)
  return projected_latent_space

def encoder_proj1(output, device, batch, dim):
  encoded_data = output['encoder'](batch)
  I = torch.eye(int(dim/2)).to(device)
  latent_space_1 = encoded_data[:,:int(dim/2)]    
  projected_1 = latent_space_1 @ (I - output['P1'])

  return projected_1

def encoder_proj2(output, device, batch, dim):
  encoded_data = output['encoder'](batch)
  I = torch.eye(int(dim/2)).to(device)
  latent_space_2 = encoded_data[:,int(dim/2):]
  projected_2 = latent_space_2 @ (I - output['P2'])

  return projected_2


def main():
  test_dataset = ColoredMNIST()
  train_loader, valid_loader, test_loader = get_data_loaders(batch_size=64)
  outputs = {}
  num_epochs = 10
  dim = 6

  autoencoder = RLACE_AE(test_dataset, device="cuda:0", d=dim)
  output = autoencoder.solve_adv_game(dim, train_loader, o_epochs=num_epochs, a_epochs=5)
  outputs[dim] = output