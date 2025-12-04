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
            tender.update({'cat': torch.tensor(d['cat'], dtype=torch.long)})
            tender.update({'cpv': torch.tensor(d['cpv'], dtype=torch.float32)})
            tender.update({'ogg': torch.tensor(d['ogg'], dtype=torch.float32)})

            id_list = []
            lat_list = []
            lon_list = []
            found_list = []
            rev_list = []
            emp_list = []
            ateco_list = []
            lab_list = []
            reg_list = []
            random.shuffle(tender["companies"])
            for c in tender['companies']:
                id_list.append(c['id'])
                lat_list.append(c['lat'])
                lon_list.append(c['lon'])
                found_list.append(c['foundation'])
                rev_list.append(c['revenue'])
                emp_list.append(c['employees'])
                ateco_list.append(torch.tensor(c['ateco'], dtype=torch.float32))
                lab_list.append(c['label'])
                reg_list.append(c['region'])

            companies = dict()
            companies.update({'id': id_list})
            companies.update({'lat': torch.tensor(lat_list, dtype=torch.float32)})
            companies.update({'lon': torch.tensor(lon_list, dtype=torch.float32)})
            companies.update({'foundation': torch.tensor(found_list, dtype=torch.float32)})
            companies.update({'revenue': torch.tensor(rev_list, dtype=torch.float32)})
            companies.update({'employees': torch.tensor(emp_list, dtype=torch.float32)})
            companies.update({'ateco': torch.stack(ateco_list)})
            companies.update({'label': torch.tensor(lab_list, dtype=torch.float32)})
            companies.update({'region': reg_list})

            tender.update({'companies': companies})
            self.data.append(tender)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    max_comp = max(len(b['companies']['lat']) for b in batch)

    t_ids = []
    t_lat = []
    t_lon = []
    budgets = []
    cats = []
    cpvs = []
    oggs = []

    c_ids = []
    c_lat = []
    c_lon = []
    foundations = []
    revenues = []
    employees = []
    atecos = []
    labels = []
    regions = []

    masks = []
    for b in batch:
        t_ids.append(b['id'])
        t_lat.append(b['lat'])
        t_lon.append(b['lon'])
        budgets.append(b['budget'])
        cats.append(b['cat'])
        cpvs.append(b['cpv'])
        oggs.append(b['ogg'])

        c = b['companies']
        c_ids.append(c['id'])
        c_lat.append(func.pad(c['lat'], (0, max_comp - len(c['lat']))))
        c_lon.append(func.pad(c['lon'], (0, max_comp - len(c['lat']))))
        foundations.append(func.pad(c['foundation'], (0, max_comp - len(c['lat']))))
        revenues.append(func.pad(c['revenue'], (0, max_comp - len(c['lat']))))
        employees.append(func.pad(c['employees'], (0, max_comp - len(c['lat']))))
        atecos.append(func.pad(c['ateco'], (0, 0, 0, max_comp - len(c['lat']))))
        labels.append(func.pad(c['label'], (0, max_comp - len(c['lat']))))
        regions.append(c['region'])

        masks.append(func.pad(torch.ones(len(c['lat'])), (0, max_comp - len(c['lat']))))

    t_lat = torch.stack(t_lat)
    t_lon = torch.stack(t_lon)
    budgets = torch.stack(budgets).unsqueeze(-1)
    cats = torch.stack(cats)
    cpvs = torch.stack(cpvs)
    oggs = torch.stack(oggs)

    c_lat = torch.stack(c_lat)
    c_lon = torch.stack(c_lon)
    foundations = torch.stack(foundations).unsqueeze(-1)
    revenues = torch.stack(revenues).unsqueeze(-1)
    employees = torch.stack(employees).unsqueeze(-1)
    atecos = torch.stack(atecos)
    labels = torch.stack(labels)

    masks = torch.stack(masks)

    tenders = {
        'id': t_ids,
        'lat': t_lat,
        'lon': t_lon,
        'budget': budgets,
        'cat': cats,
        'cpv': cpvs,
        'ogg': oggs
    }

    companies = {
        'id': c_ids,
        'lat': c_lat,
        'lon': c_lon,
        'foundation': foundations,
        'revenue': revenues,
        'employees': employees,
        'ateco': atecos,
        'label': labels,
        'region': regions
    }

    return tenders, companies, masks
