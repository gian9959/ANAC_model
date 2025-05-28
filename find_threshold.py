import json

import numpy as np
import torch
from alive_progress import alive_bar
from torch.utils.data import DataLoader

from anac_model.anac_dataset import AnacDataset
from anac_model.anac_matching_model import AnacMatchingModel
from anac_model.collate_normalization import collate_fn
from anac_model.learning_functions import validation

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
    for tender, companies, labels in val_loader:
        scores = model(tender, companies)
        all_scores.append(scores)
        all_labels.append(labels)
        total_labels += len(labels)

for t in thresholds:
    right_guesses = 0
    for i, s in enumerate(all_scores):
        s = np.array(s)
        labs = all_labels[i]
        guesses = (s >= t).astype(int)
        for j, g in enumerate(guesses):
            if g == labs[j]:
                right_guesses += 1
    if right_guesses > best_guesses:
        best_guesses = right_guesses
        best_threshold = t

print(f'Total labels: {total_labels}')
print(f'Right guesses: {best_guesses}')
print(f'Accuracy: {best_guesses/total_labels}')
print(f'Best threshold: {best_threshold}')

print()
print('Trying threshold on training set...')

right_guesses = 0

total_labels = 0

all_scores = list()
all_labels = list()

with torch.no_grad():
    for tender, companies, labels in tr_loader:
        scores = model(tender, companies)
        all_scores.append(scores)
        all_labels.append(labels)
        total_labels += len(labels)

for i, s in enumerate(all_scores):
    s = np.array(s)
    labs = all_labels[i]
    guesses = (s >= best_threshold).astype(int)
    for j, g in enumerate(guesses):
        if g == labs[j]:
            right_guesses += 1

print(f'Total labels: {total_labels}')
print(f'Right guesses: {right_guesses}')
print(f'Accuracy: {right_guesses/total_labels}')
