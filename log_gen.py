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

tr_dataset = AnacDataset('./data/anac_training.csv', db_params)
tr_loader = DataLoader(tr_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

epochs = []
tr_loss = []
val_loss = []
best_thresh = []
val_acc = []
tr_acc = []

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
    v_l, b_t, v_acc = validation(val_loader, model_params)
    val_loss.append(v_l)
    best_thresh.append(b_t)
    val_acc.append(v_acc)

    _, _, t_acc = validation(tr_loader, model_params)
    tr_acc.append(t_acc)

csv_dict = {'EPOCH': epochs, 'TRAINING LOSS': tr_loss, 'VALIDATION LOSS': val_loss, 'THRESHOLD (BASED ON VALIDATION SET)': best_thresh, 'VALIDATION ACCURACY': val_acc, 'TRAINING ACCURACY': tr_acc}
csv_df = pandas.DataFrame(csv_dict)
csv_df.to_csv(log_params['dest'], sep=';', index=False)
print(f'Csv file saved: {log_params["dest"]}')
