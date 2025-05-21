import os
import torch
import torch.nn as nn
from alive_progress import alive_bar

from anac_model.anac_matching_model import AnacMatchingModel


def training(loader, checkpoint_path='', hl=0, epoch_length=10):
    loss_fn = nn.BCEWithLogitsLoss()

    if os.path.isfile(checkpoint_path):
        print('Loading weights...')
        checkpoint = torch.load(checkpoint_path)

        hidden_layers = checkpoint['hidden_layers']
        starting_epoch = checkpoint['epoch'] + 1

        model = AnacMatchingModel(hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Weights not initialized")
        hidden_layers = hl
        starting_epoch = 1
        model = AnacMatchingModel(hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Hidden layers: {hidden_layers}")

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
        checkpoint = {'epoch': epoch, 'hidden_layers': hidden_layers, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'tr_loss': tr_loss}
        save_path = f"./checkpoints/{hidden_layers}HiddenLayers/Epoch{epoch}_checkpoint.pth"
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
