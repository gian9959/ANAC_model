import random
import torch
import torch.nn.functional as func
from torch.utils.data import Dataset


class AnacDataset(Dataset):
    def __init__(self, raw_data):
        self.data = []

        for d in raw_data:
            tender = dict(d)
            tender.update({'lat': torch.tensor(d['lat'], dtype=torch.float32)})
            tender.update({'lon': torch.tensor(d['lon'], dtype=torch.float32)})
            tender.update({'budget': torch.tensor(d['budget'], dtype=torch.float32)})
            tender.update({'cat': torch.tensor(d['cat'], dtype=torch.float32)})

            lat_list = []
            lon_list = []
            found_list = []
            rev_list = []
            ateco_list = []
            lab_list = []
            random.shuffle(tender["companies"])
            for c in tender['companies']:
                lat_list.append(c['lat'])
                lon_list.append(c['lon'])
                found_list.append(c['foundation'])
                rev_list.append(c['revenue'])
                ateco_list.append(c['ateco'])
                lab_list.append(c['label'])

            companies = dict()
            companies.update({'lat': torch.tensor(lat_list, dtype=torch.float32)})
            companies.update({'lon': torch.tensor(lon_list, dtype=torch.float32)})
            companies.update({'foundation': torch.tensor(found_list, dtype=torch.float32)})
            companies.update({'revenue': torch.tensor(rev_list, dtype=torch.float32)})
            companies.update({'ateco': torch.tensor(ateco_list)})
            companies.update({'label': torch.tensor(lab_list, dtype=torch.float32)})

            tender.update({'companies': companies})
            self.data.append(tender)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    max_comp = max(len(b['companies']) for b in batch)

    t_lat = []
    t_lon = []
    budgets = []
    cats = []

    c_lat = []
    c_lon = []
    foundations = []
    revenues = []
    atecos = []
    labels = []

    masks = []
    for b in batch:
        t_lat.append(b['lat'])
        t_lon.append(b['lon'])
        budgets.append(b['budget'])
        cats.append(b['cat'])

        tmp_lat = []
        tmp_lon = []
        tmp_foundations = []
        tmp_revenues = []
        tmp_atecos = []
        tmp_labels = []
        for c in b['companies']:
            tmp_lat.append(c['lat'])
            tmp_lon.append(c['lon'])
            tmp_foundations.append(c['foundation'])
            tmp_revenues.append(c['revenue'])
            tmp_atecos.append(c['ateco'])
            tmp_labels.append(c['label'])

        c_lat.append(func.pad(torch.stack(tmp_lat), (0, max_comp - len(b['companies'])), value=0))
        c_lon.append(func.pad(torch.stack(tmp_lon), (0, max_comp - len(b['companies'])), value=0))
        foundations.append(func.pad(torch.stack(tmp_foundations), (0, max_comp - len(b['companies'])), value=0))
        revenues.append(func.pad(torch.stack(tmp_revenues), (0, max_comp - len(b['companies'])), value=0))
        atecos.append(func.pad(torch.stack(tmp_atecos), (0, max_comp - len(b['companies'])), value=0))
        labels.append(func.pad(torch.stack(tmp_labels), (0, max_comp - len(b['companies'])), value=0))

        masks.append(func.pad(torch.ones(len(b['companies'])), (0, max_comp - len(b['companies'])), value=0))

    t_lat = torch.stack(t_lat)
    t_lon = torch.stack(t_lon)
    budgets = torch.stack(budgets)
    cats = torch.stack(cats)

    c_lat = torch.stack(c_lat)
    c_lon = torch.stack(c_lon)
    foundations = torch.stack(foundations)
    revenues = torch.stack(revenues)
    atecos = torch.stack(atecos)
    labels = torch.stack(labels)

    masks = torch.stack(masks)

    tenders = {
        'lat': t_lat,
        'lon': t_lon,
        'budget': budgets,
        'cat': cats
    }

    companies = {
        'lat': c_lat,
        'lon': c_lon,
        'foundation': foundations,
        'revenue': revenues,
        'ateco': atecos,
        'label': labels
    }

    return tenders, companies, masks
