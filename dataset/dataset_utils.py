import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,random_split, Dataset
import numpy as np
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt

class ColoredMNIST(datasets.VisionDataset):
  def __init__(self, root='./data/colored_mnist', transform=None, label='both'):
    super(ColoredMNIST, self).__init__(root, transform=transform)

    self.label = label
    self.data = torch.load(os.path.join(root, 'data_tuples.pt'))

  def __getitem__(self, index):
    img, target_num, target_col  = self.data[index]

    if self.transform is not None:
        img = self.transform(img)
        
    if self.label == 'number':
        return img, target_num
    elif self.label == 'color':
        return img, target_col
    else:
        return img, {'number':target_num, 'color':target_col}

  def __len__(self):
    return len(self.data)        
    
def color_grayscale_arr(arr, color='red'):
  arr = arr.squeeze().numpy()
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if color == 'red':
    arr = np.concatenate([arr,
                        np.zeros((h, w, 2), dtype=dtype)], axis=2)
  elif color == 'green':
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                        arr,
                        np.zeros((h, w, 1), dtype=dtype)], axis=2)
  else: 
    arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype),
                        arr], axis=2)
  return arr

def randomly_convert_colors(dataset):
  data_tuples = []
  for i in tqdm(range(len(dataset))):
    img, num_label = dataset[i]
    choice = random.random()
    if choice <= 1/3: 
      data_tuples.append((color_grayscale_arr(img, color='red').transpose(2, 1, 0), num_label, 0))
      ## 0 = red
    elif choice <= 2/3: 
      data_tuples.append((color_grayscale_arr(img, color='green').transpose(2, 1, 0), num_label, 1))
      ## 1 = green
    else:
      data_tuples.append((color_grayscale_arr(img, color='blue').transpose(2, 1, 0), num_label, 2))
      ## 2 = blue
  
  return data_tuples


def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 8))
  columns = 6
  rows = 3
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label = dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label['number']))  # set title
    plt.imshow(img.transpose(2, 1, 0))

  plt.show()  # finally, render the plot

def get_data_loaders(batch_size=64):
  dataset = ColoredMNIST()
  # plot_dataset_digits(dataset)

  train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.8, 0.10, 0.10])

  train_loader = DataLoader(train_dataset, batch_size=batch_size)
  valid_loader = DataLoader(validation_dataset, batch_size=batch_size)
  test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
  return train_loader, valid_loader, test_loader


def main():
  ### Set the random seed for reproducible results
  torch.manual_seed(0)

  preprocess = transforms.ToTensor()

  dataset = torchvision.datasets.MNIST('./data/mnist', download=True, transform=preprocess)
  colored_dataset = randomly_convert_colors(dataset)

  os.makedirs('./data/colored_mnist', exist_ok=True)
  torch.save(colored_dataset, './data/colored_mnist/data_tuples.pt')


if __name__ == '__main__':
  main()