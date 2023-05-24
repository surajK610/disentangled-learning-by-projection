import argparse
import torch
import os
import numpy as np
from dataset.dataset_utils import ColoredMNIST, get_data_loaders
from models.disentangled_autoencoder import RLACE_AE , encoder_proj1, encoder_proj2   
from models.logistic_classifier import LogisticClassifier
import json

NUM_CLASSES_NUMBER = 10
NUM_CLASSES_COLOR = 3

def main(FLAGS):
    
    test_dataset = ColoredMNIST() ## this is only used for visualizing does not matter that includes everything
    train_loader, valid_loader, test_loader = get_data_loaders(batch_size=FLAGS.batch_size)

    num_epochs = FLAGS.epochs
    a_epochs = FLAGS.a_epochs
    d = FLAGS.d
    rank = FLAGS.rank

    os.makedirs('outputs/disentangled_autoencoder/training_figures', exist_ok=True)
    autoencoder = RLACE_AE(test_dataset, rank=rank, d=d, batch_size=FLAGS.batch_size,
                            device=FLAGS.device, fig_path='outputs/disentangled_autoencoder/training_figures')
    print(f"Solving adversarial game for dimension {d}")
    output = autoencoder.solve_adv_game(train_loader, valid_loader, o_epochs=num_epochs, a_epochs=a_epochs)
    torch.save(output, f'outputs/disentangled_autoencoder/outputs_dim_{d}.pt')

    lc_num = LogisticClassifier(d, NUM_CLASSES_NUMBER, lr=0.0001, device=FLAGS.device) # 10 = number classes
    _, _, val_acc_num = lc_num.fit(train_loader, valid_loader, output['encoder'], label='number', epochs=20)

    lc_col = LogisticClassifier(d, NUM_CLASSES_COLOR, lr=0.001, device=FLAGS.device)
    _, _, val_acc_col =lc_col.fit(train_loader, valid_loader, output['encoder'], label='color', epochs=10)

    #encoder_with_projection_d = lambda batch: encoder_with_projection(batch, dim=dim)
    encoder_proj1_d = lambda batch: encoder_proj1(output, FLAGS.device, batch, dim=d)
    encoder_proj2_d = lambda batch: encoder_proj2(output, FLAGS.device, batch, dim=d)

    lc_num_proj1 = LogisticClassifier(int(d/2), NUM_CLASSES_NUMBER, lr=0.001, device=FLAGS.device)
    _, _, val_acc_num_proj1 = lc_num_proj1.fit(train_loader, valid_loader, encoder_proj1_d, 
                                            label='number', epochs=20)
    lc_col_proj1 = LogisticClassifier(int(d/2), NUM_CLASSES_COLOR, lr=0.001, device=FLAGS.device)
    _, _, val_acc_col_proj1 =  lc_col_proj1.fit(train_loader, valid_loader, encoder_proj1_d, 
                                            label='color', epochs=10)

    lc_num_proj2 = LogisticClassifier(int(d/2), NUM_CLASSES_NUMBER, lr=0.001, device=FLAGS.device)
    _, _, val_acc_num_proj2 = lc_num_proj2.fit(train_loader, valid_loader, encoder_proj2_d, 
                                            label='number', epochs=20)
    lc_col_proj2 = LogisticClassifier(int(d/2), NUM_CLASSES_COLOR, lr=0.001, device=FLAGS.device)
    _, _, val_acc_col_proj2 =  lc_col_proj2.fit(train_loader, valid_loader, encoder_proj2_d, 
                                            label='color', epochs=10)

    results = {
        'before_projection': {'number_accuracy': val_acc_num, 'color_accuracy': val_acc_col},
        'number_dimensions' : {'number_accuracy': val_acc_num_proj1, 'color_accuracy': val_acc_col_proj1},
        'color_dimensions': {'number_accuracy': val_acc_num_proj2, 'color_accuracy': val_acc_col_proj2},
    }
    json.dump(results, open(f'outputs/disentangled_autoencoder/results_dim_{d}.json', 'w'),  indent = 6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='Disentangled Learning Experiments',
                description='trains disentangled autoencoder by projecting classifiers in the latent space using R-LACE')
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--device", default="cuda:0", type=str, help="device to run on")
    parser.add_argument("--d", default=6, type=int, help="number of latent dimensions for autoencoder")
    parser.add_argument("--rank", default=1, type=int, help="rank of linear subspace to project with RLACE")

    parser.add_argument("--epochs", default=10, type=int, help="number of epochs for training")
    parser.add_argument("--a-epochs", default=5, type=int, help="number of autoencoder epochs per rlace epoch")

    FLAGS = parser.parse_args()

    main(FLAGS)
    