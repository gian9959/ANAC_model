import json

import numpy as np
import torch
from alive_progress import alive_bar
from torch.utils.data import DataLoader

from anac_model.anac_dataset import AnacDataset
from anac_model.anac_matching_model import AnacMatchingModel
from anac_model.collate_normalization import collate_fn
from anac_model.learning_functions import validation


def guesses(all_scores, all_labels):
    right_guesses = 0
    for i, scores in enumerate(all_scores):
        labels = all_labels[i]
        for j, s in enumerate(scores):
            if s >= t:
                s = 1
            else:
                s = 0
            if s == int(labels[j]):
                right_guesses += 1
    return right_guesses

with open('config.json', 'r') as f:
    config = json.load(f)

db_params = config['db_params']
model_params = config['model_params']

print('Loading validation dataset...')
val_dataset = AnacDataset('./data/anac_validation.csv', db_params)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

print('Loading training dataset...')
tr_dataset = AnacDataset('./data/anac_training.csv', db_params)
tr_loader = DataLoader(tr_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

print()
print(f'Checkpoint: {model_params["checkpoint"]}')
val_loss = validation(val_loader, model_params)

print()
print('Finding best threshold...')

hidden_layers = model_params['hidden_layers']
dropout = model_params['dropout']

checkpoint = torch.load(model_params['checkpoint'])

if checkpoint['hidden_layers'] is not None:
    hidden_layers = checkpoint['hidden_layers']

if checkpoint['dropout'] is not None:
    dropout = checkpoint['dropout']

model = AnacMatchingModel(hidden_layers=hidden_layers, dropout=dropout)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

thresholds = np.linspace(0, 1, 100)

best_threshold = 0.0
best_guesses = 0

total_labels = 0

all_scores = list()
all_labels = list()

with torch.no_grad():
    for tend, comp, lab in val_loader:
        s = model(tend, comp)
        all_scores.append(s)
        all_labels.append(lab)
        total_labels += len(lab)

for t in thresholds:
    right_guesses = guesses(all_scores, all_labels)
    if right_guesses > best_guesses:
        best_guesses = right_guesses
        best_threshold = t

print(f'Total labels: {total_labels}')
print(f'Right guesses: {best_guesses}')
print(f'Accuracy: {best_guesses / total_labels}')
print(f'Best threshold: {best_threshold}')

print()
print('Trying threshold on training set...')

right_guesses = 0

total_labels = 0

all_scores = list()
all_labels = list()

with torch.no_grad():
    for tend, comp, lab in tr_loader:
        scores = model(tend, comp)
        all_scores.append(scores)
        all_labels.append(lab)
        total_labels += len(lab)

right_guesses = guesses(all_scores, all_labels)

print(f'Total labels: {total_labels}')
print(f'Right guesses: {right_guesses}')
print(f'Accuracy: {right_guesses / total_labels}')
