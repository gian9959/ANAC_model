import os
import re
import torch
import torch.nn as nn
from alive_progress import alive_bar
from torch.utils.data import DataLoader

from anac_model.anac_matching_model import AnacMatchingModel
from anac_model.anac_dataset import AnacDataset
from anac_model.collate_normalization import collate_fn


def training(checkpoint_path=''):
    print('Loading dataset...')
    dataset = AnacDataset('./data/anac_training.csv')
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    loss_fn = nn.BCEWithLogitsLoss()

    print('Loading weights...')
    if os.path.isfile(checkpoint_path):
        hidden_layers = int(re.findall('[0-9]+', checkpoint_path)[0])

        model = AnacMatchingModel(hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        starting_epoch = checkpoint['epoch']
    else:
        hidden_layers = 0
        model = AnacMatchingModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        starting_epoch = 1

    print('Training...')
    model.train()

    # Training loop
    for epoch in range(starting_epoch, 10 + starting_epoch):
        with alive_bar(len(loader)) as bar:
            for tender, companies, labels in loader:
                scores = model(tender, companies)
                loss = loss_fn(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar()

        print(f"Epoch {epoch}: Loss {loss.item():.4f}")
        checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, f"./checkpoints/{hidden_layers}HiddenLayers/Epoch{epoch}_checkpoint.pth")

    return epoch


def validation(checkpoint_path):
    print('Loading dataset...')
    dataset = AnacDataset('./data/anac_validation.csv')
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    model = AnacMatchingModel()
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint)
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
