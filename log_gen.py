import os
import json
import pandas
import torch
from torch.utils.data import DataLoader

from anac_model.anac_dataset import AnacDataset
from anac_model.collate_normalization import collate_fn
from anac_model.learning_functions import validation

print('Loading validation dataset...')
val_dataset = AnacDataset('./data/anac_validation.csv')
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

with open('config.json', 'r') as f:
    config = json.load(f)

epochs = []
tr_loss = []
val_loss = []

file_list = os.listdir(config["log_source"])
file_list.sort()
file_list.sort(key=len)

for check_path in file_list:
    path = config["log_source"] + '/' + check_path

    checkpoint = torch.load(path)
    print()
    print(f"Epoch: {checkpoint['epoch']}")

    epochs.append(checkpoint['epoch'])
    tr_loss.append(checkpoint['tr_loss'])

    val_loss.append(validation(val_loader, checkpoint_path=path))

csv_dict = {'EPOCH': epochs, 'TR_LOSS': tr_loss, 'VAL_LOSS': val_loss}
csv_df = pandas.DataFrame(csv_dict)
csv_df.to_csv(config['log_dest'], sep=';', index=False)
