import os

import numpy as np
import torch
import torch.nn as nn
from alive_progress import alive_bar

from anac_model.anac_matching_model import AnacMatchingModel


def training(loader, model_params):
    starting_epoch = 1
    train_length = model_params['train_length']
    hidden_layers = model_params['hidden_layers']
    dropout = model_params['dropout']

    dr_string = ''

    if os.path.isfile(model_params['checkpoint']):
        print(f'Loading checkpoint: {model_params["checkpoint"]}')
        checkpoint = torch.load(model_params['checkpoint'])

        if checkpoint['epoch'] is not None:
            starting_epoch = checkpoint['epoch'] + 1

        if checkpoint['hidden_layers'] is not None:
            hidden_layers = checkpoint['hidden_layers']

        if checkpoint['dropout'] is not None:
            dropout = checkpoint['dropout']

        model = AnacMatchingModel(hidden_layers, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Checkpoint NOT loaded")
        model = AnacMatchingModel(hidden_layers, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Hidden layers: {hidden_layers}")

    if dropout:
        dr_string = '_dr'
    print(f"Dropout: {dropout}")

    print('Training...')
    loss_fn = nn.BCELoss()
    model.train()

    for epoch in range(starting_epoch, starting_epoch + train_length):
        tr_loss = 0.0
        with alive_bar(len(loader)) as bar:
            for tender, companies, labels in loader:
                scores = model(tender, companies)
                loss = loss_fn(scores, labels)
                tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar()

        tr_loss = tr_loss / len(loader)
        print(f"Epoch {epoch}: Loss {tr_loss:.4f}")

        checkpoint = {'epoch': epoch, 'hidden_layers': hidden_layers, 'dropout': dropout,
                      'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'tr_loss': tr_loss}

        save_path = f"./checkpoints/{hidden_layers}H{dr_string}/Epoch{epoch}_checkpoint.pth"
        torch.save(checkpoint, save_path)

    return save_path


def guesses(scores_list, labels_list, threshold):
    rg = 0
    for i, values in enumerate(scores_list):
        labels = labels_list[i]
        for j, v in enumerate(values):
            if v >= threshold:
                v = 1
            else:
                v = 0
            if v == int(labels[j]):
                rg += 1
    return rg


def validation(loader, model_params):
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

    loss_fn = nn.BCELoss()
    val_loss = 0.0

    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.0
    best_guesses = 0
    total_labels = 0
    all_scores = list()
    all_labels = list()

    with torch.no_grad():
        with alive_bar(len(loader)) as bar:
            for tender, companies, labels in loader:
                scores = model(tender, companies)

                loss = loss_fn(scores, labels)
                val_loss += loss.item()

                all_scores.append(scores)
                all_labels.append(labels)
                total_labels += len(labels)
                bar()

    with alive_bar(len(thresholds)) as bar:
        for t in thresholds:
            right_guesses = guesses(all_scores, all_labels, t)
            if right_guesses > best_guesses:
                best_guesses = right_guesses
                best_threshold = t
            bar()

    print(f"Validation Loss: {val_loss/len(loader):.4f}")

    print()
    print(f'Best threshold: {best_threshold}')
    print(f'Accuracy: {best_guesses / total_labels}')
    print(f'{best_guesses} right guesses out of {total_labels}')

    return val_loss / len(loader), best_threshold, best_guesses / total_labels
