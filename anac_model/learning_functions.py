import os
import torch
import torch.nn as nn
from alive_progress import alive_bar
from torch.utils.data import DataLoader

from anac_model.anac_matching_model import AnacMatchingModel
from anac_model.anac_dataset import AnacDataset
from anac_model.collate_normalization import collate_fn


def training(checkpoint_path='', hl=0, epoch_length=10):
    print('Loading dataset...')
    dataset = AnacDataset('./data/anac_training.csv')
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    loss_fn = nn.BCEWithLogitsLoss()

    print('Loading weights...')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        hidden_layers = checkpoint['hidden_layers']
        starting_epoch = checkpoint['epoch']

        model = AnacMatchingModel(hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        hidden_layers = hl
        starting_epoch = 1
        model = AnacMatchingModel(hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print('Training...')
    model.train()

    for epoch in range(starting_epoch, starting_epoch+epoch_length):
        with alive_bar(len(loader)) as bar:
            for tender, companies, labels in loader:
                scores = model(tender, companies)
                loss = loss_fn(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar()

        print(f"Epoch {epoch}: Loss {loss.item():.4f}")
        checkpoint = {'epoch': epoch, 'hidden_layers': hidden_layers, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'loss': loss}
        save_path = f"./checkpoints/{hidden_layers}HiddenLayers/Epoch{epoch}_checkpoint.pth"
        torch.save(checkpoint, save_path)

    return save_path


def validation(checkpoint_path):
    print('Loading dataset...')
    dataset = AnacDataset('./data/anac_validation.csv')
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

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

    val_loss /= len(loader)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss
