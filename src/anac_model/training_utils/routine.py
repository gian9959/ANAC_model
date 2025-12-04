import pandas
from alive_progress import alive_bar
from torch.utils.data import DataLoader

from anac_model.data_utils.main_utils import extract_from_index, load_from_index, normalization
from anac_model.data_utils.other_utils import load_config, connect_to_db
from anac_model.datasets.anac_dataset import AnacDataset, collate
from anac_model.training_utils.learning_functions import training, validation

config = load_config()

print('Loading indexes...')
training_csv = pandas.read_csv('../../data/indexes/training_index.csv', sep=';')
validation_csv = pandas.read_csv('../../data/indexes/validation_index.csv', sep=';')

conn = connect_to_db(config['db_params'])

print('Loading normalization constants...')
const = config['constants']
print(f'CONSTANTS LOADED: {const}')

print('Loading training data...')
training_data = []
with alive_bar(len(training_csv)) as bar:
    for i, row in training_csv.iterrows():
        cig, cf_list, labels, revs, emps = extract_from_index(row)
        d = load_from_index(cig, cf_list, labels, revs, emps, conn)
        norm_d = normalization(d, const)
        training_data.append(norm_d)
        bar()
print('Creating dataset...')
training_set = AnacDataset(training_data)
training_loader = DataLoader(training_set, batch_size=32, collate_fn=collate, shuffle=True)

print('Loading validation data...')
validation_data = []
with alive_bar(len(validation_csv)) as bar:
    for i, row in validation_csv.iterrows():
        cig, cf_list, labels, revs, emps = extract_from_index(row)
        d = load_from_index(cig, cf_list, labels, revs, emps, conn)
        norm_d = normalization(d, const)
        validation_data.append(norm_d)
        bar()
print('Creating dataset...')
validation_set = AnacDataset(validation_data)
validation_loader = DataLoader(validation_set, batch_size=1, collate_fn=collate, shuffle=False)

model_params = config['model_params']

print("Starting training")
for i in range(model_params['sessions']):
    model_params['checkpoint'] = training(training_loader, model_params)
    validation(validation_loader, model_params)
