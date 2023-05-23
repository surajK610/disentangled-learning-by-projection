import torch
from tqdm import trange
import numpy as np


class CounterFactual:
  def __init__(self, classifier, device):
    self.classifier = classifier
    for param in self.classifier.parameters():
        param.requires_grad = False
    self.device = device
    
    self.prev = 1e9

    self.progress_patience = 0
    self.threshold_patience = 0
    self.patience = 50

    # I selected, 1e-2; saw the original paper used a higher value.
    self.loss_tolerance = 0.05

  def compute_counterfactual(self, latent_vector_batch, new_label_index_batch, device='cuda:0', max_steps=10_000):

    self.x_nudge = latent_vector_batch.clone().to(device)
    self.x_nudge.requires_grad_(True)

    self.optimizer = torch.optim.AdamW([self.x_nudge], lr=1e-3, eps=1e-4)
    self.target = new_label_index_batch.to(device)
    self.target.requires_grad_(False)

    prog_bar = trange(max_steps, desc=f'Epoch 0')

    for i in prog_bar:
      batch_at_step, done, loss_across_max, accuracy, gradnorm = self.step()
      prog_bar.set_postfix(loss_across_max=loss_across_max)
      prog_bar.set_description(f'Epoch {i}')

      if done:
        return self.x_nudge, "converged"

    return self.x_nudge, "timed-out"

  def step(self):
      
    self.optimizer.zero_grad()
    logits = self.classifier(self.x_nudge)
    
    loss_across = torch.nn.functional.cross_entropy(
      logits, self.target, reduction="none"
    )
    
    loss = loss_across.sum()
    loss.backward()
    
    accuracy = (logits.argmax(1) == self.target).float().mean().item()
    gradnorm = self.x_nudge.grad.detach().pow(2).sum().sqrt().cpu().item()
    
    self.optimizer.step()
    
    loss_across_max = loss_across.max().cpu().item()
    
    if loss_across_max >= self.prev:
      self.progress_patience += 1
    else:
      self.progress_patience = 0

    if (loss_across_max < self.loss_tolerance) or (self.progress_patience >= self.patience):
      done = True
    else:
      done = False

    self.prev = loss_across_max

    if np.isnan(loss_across_max):
      raise Exception("Loss is nan bruv.")

    return self.x_nudge.clone().detach(), done, loss_across_max, accuracy, gradnorm