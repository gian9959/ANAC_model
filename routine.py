import json
from torch.utils.data import DataLoader

from anac_model.anac_dataset import AnacDataset
from anac_model.collate_normalization import collate_fn
from anac_model.learning_functions import training
from anac_model.learning_functions import validation

print('Loading training dataset...')
tr_dataset = AnacDataset('./data/anac_training.csv')
tr_loader = DataLoader(tr_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

print('Loading validation dataset...')
val_dataset = AnacDataset('./data/anac_validation.csv')
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

with open('config.json', 'r') as f:
    config = json.load(f)

checkpoint_path = config['checkpoint']
hidden_layers = config['hidden_layers']

for i in range(10):
    checkpoint_path = training(tr_loader, checkpoint_path=checkpoint_path, hl=hidden_layers)
    val_loss = validation(val_loader, checkpoint_path=checkpoint_path)
