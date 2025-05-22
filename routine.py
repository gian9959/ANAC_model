import json
from torch.utils.data import DataLoader

from anac_model.anac_dataset import AnacDataset
from anac_model.collate_normalization import collate_fn
from anac_model.learning_functions import training
from anac_model.learning_functions import validation

with open('config.json', 'r') as f:
    config = json.load(f)

db_params = config['db_params']

print('Loading training dataset...')
tr_dataset = AnacDataset('./data/anac_training.csv', db_params)
tr_loader = DataLoader(tr_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

print('Loading validation dataset...')
val_dataset = AnacDataset('./data/anac_validation.csv', db_params)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

model_params = config['model_params']

for i in range(10):
    model_params['checkpoint'] = training(tr_loader, model_params)
    val_loss = validation(val_loader, model_params)
