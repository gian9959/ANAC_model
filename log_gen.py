import os
import json
import pandas
import torch
from torch.utils.data import DataLoader

from anac_model.anac_dataset import AnacDataset
from anac_model.collate_normalization import collate_fn
from anac_model.learning_functions import validation

with open('config.json', 'r') as f:
    config = json.load(f)

db_params = config['db_params']

print('Loading validation dataset...')
val_dataset = AnacDataset('./data/anac_validation.csv', db_params)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

epochs = []
tr_loss = []
val_loss = []

log_params = config['log_params']

file_list = os.listdir(log_params["source"])
file_list.sort()
file_list.sort(key=len)

model_params = config['model_params']
print(f'Source directory: {log_params["source"]}')

for check_path in file_list:
    path = log_params["source"] + '/' + check_path

    checkpoint = torch.load(path)
    print()
    print(f"Epoch: {checkpoint['epoch']}")

    epochs.append(checkpoint['epoch'])
    tr_loss.append(checkpoint['tr_loss'])

    model_params['checkpoint'] = path
    val_loss.append(validation(val_loader, model_params))

csv_dict = {'EPOCH': epochs, 'TR_LOSS': tr_loss, 'VAL_LOSS': val_loss}
csv_df = pandas.DataFrame(csv_dict)
csv_df.to_csv(log_params['dest'], sep=';', index=False)
print(f'Csv file saved: {log_params["dest"]}')
