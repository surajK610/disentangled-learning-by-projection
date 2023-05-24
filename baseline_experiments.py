import argparse
import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt

from dataset.dataset_utils import ColoredMNIST, get_data_loaders
from models.autoencoder import Encoder, Decoder, plot_ae_outputs, linearly_interpolate_test_idxs, test_epoch, train_epoch
from models.classifier import Classifier, train_epoch_classifer, test_epoch_classifier


def baseline_classification(FLAGS, train_loader, valid_loader):

  os.makedirs('outputs/classifiers', exist_ok=True)
  device = FLAGS.device

  ## Number Classifier -- Runs 5 Epochs
  classifier_number = Classifier(10)
  loss_fn_number = torch.nn.CrossEntropyLoss()

  lr_number = FLAGS.lr_number
  optim_class_layout = torch.optim.Adam(classifier_number.parameters(), lr=lr_number, weight_decay=1e-05)
  classifier_number.to(device)

  num_epochs = 5
  history_layout={'class_loss':[],'val_loss':[]}


  for epoch in range(num_epochs):
      train_loss = train_epoch_classifer(classifier_number,device,train_loader,loss_fn_number, optim_class_layout)
      val_loss, val_accuracy = test_epoch_classifier(classifier_number,device,valid_loader,loss_fn_number)
      
      print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}  \t val acc {:.3f} '.format(epoch + 1, num_epochs,train_loss,val_loss,val_accuracy))
      history_layout['class_loss'].append(train_loss)
      history_layout['val_loss'].append(val_loss)


  ## Color Classifier -- Runs 1 Epoch
  classifier_color = Classifier(3)
  loss_fn_color = torch.nn.CrossEntropyLoss()

  lr_class = 0.0005

  optim_class_shape = torch.optim.Adam(classifier_color.parameters(), lr=lr_class, weight_decay=1e-05)

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  classifier_color.to(device)

  num_epochs = 1
  history_shape={'class_loss':[],'val_loss':[]}


  for epoch in range(num_epochs):
      train_loss = train_epoch_classifer(classifier_color,device,train_loader,loss_fn_color,optim_class_shape, label='color')
      val_loss, val_accuracy = test_epoch_classifier(classifier_color,device,valid_loader,loss_fn_color, label='color')
      
      print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}  \t val acc {:.3f} '.format(epoch + 1, num_epochs,train_loss,val_loss,val_accuracy))
      history_shape['class_loss'].append(train_loss)
      history_shape['val_loss'].append(val_loss)

  torch.save(classifier_number.state_dict(), 'outputs/classifiers/classifier_number.pt')
  torch.save(classifier_color.state_dict(), 'outputs/classifiers/classifier_color.pt')
    

def baseline_autoencoder(FLAGS, train_loader, valid_loader, test_loader, test_dataset):

  os.makedirs('outputs/autoencoder', exist_ok=True)
  os.makedirs('outputs/autoencoder/training_images', exist_ok=True)
  d = FLAGS.d

  encoder = Encoder(encoded_space_dim=d)
  decoder = Decoder(encoded_space_dim=d)
  loss_fn = torch.nn.MSELoss()
  lr= FLAGS.lr_autoencoder


  params_to_optimize = [
      {'params': encoder.parameters()},
      {'params': decoder.parameters()},
  ]

  optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

  device = FLAGS.device

  encoder.to(device)
  decoder.to(device)

  num_epochs = 100
  history={'train_loss':[],'val_loss':[]}

  for epoch in range(num_epochs):
      train_loss = train_epoch(encoder,decoder,device,train_loader,loss_fn,optim)
      val_loss = test_epoch(encoder,decoder,device,valid_loader,loss_fn)
      print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
      history['train_loss'].append(train_loss)
      history['val_loss'].append(val_loss)
      plot_ae_outputs(encoder,decoder,test_dataset, device, idx=epoch, n=5, fig_path='outputs/autoencoder/training_images')
  
  test_error = test_epoch(encoder,decoder,device,test_loader,loss_fn).item()
  print(f'Test error: {test_error}')

  
  torch.save(encoder.state_dict(), f'outputs/autoencoder/encoder_d={d}.pt')
  torch.save(decoder.state_dict(), f'outputs/autoencoder/decoder_d={d}.pt')

  plt.figure(figsize=(10,8))
  plt.plot(history['train_loss'], label='Train Loss')
  plt.plot(history['val_loss'], label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Average Loss')
  #plt.grid()
  plt.legend()
  plt.title(f'Loss Curves Autoencoder d={d}')
  plt.savefig(f'outputs/autoencoder/loss_curves_autoencoder_d={d}.png')

  linearly_interpolate_test_idxs(decoder, 1, 15, 2, 14, fig_path='outputs/autoencoder/interpolation.png')


def main(FLAGS):
  test_dataset = ColoredMNIST()
  train_loader, valid_loader, test_loader = get_data_loaders(batch_size=FLAGS.batch_size)
  print(FLAGS.baseline_classification)
  if FLAGS.baseline_classification:
    baseline_classification(FLAGS, train_loader, valid_loader)
  if FLAGS.baseline_autoencoder:
    baseline_autoencoder(FLAGS, train_loader, valid_loader, test_loader, test_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='Baseline Experiments',
                description='trains autoencoder for Colored MNIST and assess classification performance as well as generates some figures')
    parser.add_argument("--baseline-classification", default=False, type=bool, help="run baseline classification experiment")
    parser.add_argument("--baseline-autoencoder", default=False, type=bool, help="run baseline autoencoder experiment")

    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--device", default="cuda:0", type=str, help="device to run on")
    parser.add_argument("--lr-number", default=0.001, type=float, help="learning rate for number classifier")
    parser.add_argument("--lr-color", default=0.0005, type=float, help="learning rate for color classifier")
    parser.add_argument("--lr-autoencoder", default=0.001, type=float, help="learning rate for autoencoder")
    parser.add_argument("--d", default=4, type=int, help="number of latent dimensions for autoencoder")
    
    FLAGS = parser.parse_args()

    main(FLAGS)
    