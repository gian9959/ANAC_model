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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Checkpoint NOT loaded")
        model = AnacMatchingModel(hidden_layers, dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"Hidden layers: {hidden_layers}")

    if dropout:
        dr_string = '_dr'
    print(f"Dropout: {dropout}")

    print('Training...')
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(starting_epoch, starting_epoch + train_length):
        tr_loss = 0.0
        with alive_bar(len(loader)) as bar:
            for tender, companies, mask in loader:
                scores = model(tender, companies)
                masked_scores = scores[mask.bool()]
                labels = (companies['label'])[mask.bool()]
                loss = loss_fn(masked_scores, labels)
                tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar()

        tr_loss = tr_loss / len(loader)
        print(f"Epoch {epoch}: Loss {tr_loss:.4f}")

        checkpoint = {'epoch': epoch, 'hidden_layers': hidden_layers, 'dropout': dropout,
                      'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'tr_loss': tr_loss}

        save_path = f"../../checkpoints/{hidden_layers}H{dr_string}/Epoch{epoch}_checkpoint.pth"
        torch.save(checkpoint, save_path)

    return save_path


def guesses(scores_list, labels_list, threshold):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, values in enumerate(scores_list):
        labels = labels_list[i]
        for j, v in enumerate(values):
            if v >= threshold:
                r = 1
            else:
                r = 0

            if r == labels[j]:
                if r == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if r == 1:
                    fp += 1
                else:
                    fn += 1
    return tp, tn, fp, fn


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

    loss_fn = nn.BCEWithLogitsLoss()
    val_loss = 0.0

    with torch.no_grad():
        with alive_bar(len(loader)) as bar:
            for tender, companies, mask in loader:
                scores = model(tender, companies)
                masked_scores = scores[mask.bool()]
                labels = (companies['label'])[mask.bool()]

                loss = loss_fn(masked_scores, labels)
                val_loss += loss.item()

                bar()

    print(f"Validation Loss: {val_loss / len(loader):.4f}")

    return val_loss / len(loader)


def f1_and_accuracy(loader, model_params, threshold=0.5):
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
    sig = nn.Sigmoid()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        with alive_bar(len(loader)) as bar:
            for tender, companies, mask in loader:
                scores = model(tender, companies)
                masked_scores = scores[mask.bool()]
                sig_scores = sig(masked_scores)
                labels = (companies['label'])[mask.bool()]

                all_scores.append(sig_scores)
                all_labels.append(labels)
                bar()

    tp, tn, fp, fn = guesses(all_scores, all_labels, threshold)
    guess = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * ((precision * recall) / (precision + recall))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL: {recall:.4f}")
    print(f"F1 SCORE: {f_score:.4f}")
    print(f"ACCURACY: {accuracy:.4f}")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")

    return precision, recall, f_score, accuracy, guess


def find_best_threshold(loader, model_params):
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
    sig = nn.Sigmoid()

    all_scores = []
    all_labels = []

    max_correct = 0

    thresholds = np.linspace(0.01, 1, 100)
    print(thresholds)
    best_t = thresholds[0]

    with torch.no_grad():
        with alive_bar(len(loader)) as bar:
            for tender, companies, mask in loader:
                scores = model(tender, companies)
                masked_scores = scores[mask.bool()]
                sig_scores = sig(masked_scores)
                labels = (companies['label'])[mask.bool()]

                all_scores.append(sig_scores)
                all_labels.append(labels)
                bar()

    print('Finding best threshold...')
    with alive_bar(len(thresholds)) as bar:
        for t in thresholds:
            tp, tn, fp, fn = guesses(all_scores, all_labels, t)
            if (tp + tn) > max_correct:
                max_correct = tp + tn
                best_t = t
            bar()

    print(f"BEST THRESHOLD: {best_t:.4f}")
    print(f"MAX CORRECT GUESSES: {max_correct}")

    return best_t, max_correct
