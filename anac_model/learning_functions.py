import os
import torch
import torch.nn as nn
from alive_progress import alive_bar

from anac_model.anac_matching_model import AnacMatchingModel


def training(loader, checkpoint_path='', hidden_layers=0, dropout=False, epoch_length=10):

    loss_fn = nn.BCEWithLogitsLoss()
    starting_epoch = 1

    dr_string = ''

    if os.path.isfile(checkpoint_path):
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)

        if checkpoint['epoch'] is not None:
            starting_epoch = checkpoint['epoch'] + 1

        if checkpoint['hidden_layers'] is not None:
            hidden_layers = checkpoint['hidden_layers']

        if checkpoint['dropout'] is not None:
            dropout = checkpoint['dropout']

        model = AnacMatchingModel(hidden_layers, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Checkpoint NOT loaded")
        model = AnacMatchingModel(hidden_layers, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Hidden layers: {hidden_layers}")

    if dropout:
        dr_string = '_dr'
    print(f"Dropout: {dropout}")

    print('Training...')
    model.train()

    for epoch in range(starting_epoch, starting_epoch + epoch_length):
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


def validation(loader, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = AnacMatchingModel(checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        with alive_bar(len(loader)) as bar:
            for tender, companies, labels in loader:
                scores = model(tender, companies)
                loss = loss_fn(scores, labels)
                val_loss += loss.item()
                bar()

    val_loss = val_loss / len(loader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss
