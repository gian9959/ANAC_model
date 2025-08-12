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
            tender.update({'cpv': torch.tensor(d['cpv'], dtype=torch.float32)})
            tender.update({'ogg': torch.tensor(d['ogg'], dtype=torch.float32)})

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
                ateco_list.append(torch.tensor(c['ateco'], dtype=torch.float32))
                lab_list.append(c['label'])

            companies = dict()
            companies.update({'lat': torch.tensor(lat_list, dtype=torch.float32)})
            companies.update({'lon': torch.tensor(lon_list, dtype=torch.float32)})
            companies.update({'foundation': torch.tensor(found_list, dtype=torch.float32)})
            companies.update({'revenue': torch.tensor(rev_list, dtype=torch.float32)})
            companies.update({'ateco': torch.stack(ateco_list)})
            companies.update({'label': torch.tensor(lab_list, dtype=torch.float32)})

            tender.update({'companies': companies})
            self.data.append(tender)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    max_comp = max(len(b['companies']['lat']) for b in batch)

    t_lat = []
    t_lon = []
    budgets = []
    cats = []
    cpvs = []
    oggs = []

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
        cpvs.append(b['cpv'])
        oggs.append(b['ogg'])

        c = b['companies']
        c_lat.append(func.pad(c['lat'], (0, max_comp - len(c['lat']))))
        c_lon.append(func.pad(c['lon'], (0, max_comp - len(c['lat']))))
        foundations.append(func.pad(c['foundation'], (0, max_comp - len(c['lat']))))
        revenues.append(func.pad(c['revenue'], (0, max_comp - len(c['lat']))))
        atecos.append(func.pad(c['ateco'], (0, 0, 0, max_comp - len(c['lat']))))
        labels.append(func.pad(c['label'], (0, max_comp - len(c['lat']))))

        masks.append(func.pad(torch.ones(len(c['lat'])), (0, max_comp - len(c['lat']))))

    t_lat = torch.stack(t_lat)
    t_lon = torch.stack(t_lon)
    budgets = torch.stack(budgets)
    cats = torch.stack(cats)
    cpvs = torch.stack(cpvs)
    oggs = torch.stack(oggs)

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
        'cat': cats,
        'cpv': cpvs,
        'oggs': oggs
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
