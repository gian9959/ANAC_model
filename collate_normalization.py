import torch

MIN_LAT = 36.9269
MAX_LAT = 46.4983
MIN_LON = 7.3170
MAX_LON = 18.1750

AVG_BUDGET = 11639055
STD_BUDGET = 211413033

MAX_FOUNDATION = 2023
MIN_FOUNDATION = 1856

AVG_REVENUE = 15966221
STD_REVENUE = 279661459

CATEGORIES = {'LAVORI': 0, 'FORNITURE': 1, 'SERVIZI': 2}


def collate_fn(batch):
    t = dict()
    c = dict()
    labs = []

    if len(batch) == 1:
        data = batch[0]

        lat = (float(data.get('lat')) - MIN_LAT) / (MAX_LAT - MIN_LAT)
        lon = (float(data.get('lon')) - MIN_LON) / (MAX_LON - MIN_LON)

        budget = (float(data.get('budget')) - AVG_BUDGET) / STD_BUDGET

        cat = CATEGORIES.get(data.get('cat'))

        t.update({'lat': torch.tensor(lat, dtype=torch.float32).unsqueeze(0)})
        t.update({'lon': torch.tensor(lon, dtype=torch.float32).unsqueeze(0)})
        t.update({'budget': torch.tensor(budget, dtype=torch.float32).unsqueeze(0)})
        t.update({'cat': torch.tensor(cat, dtype=torch.int).unsqueeze(0)})
        t.update({'cpv_desc': torch.tensor(data.get('cpv_desc'), dtype=torch.float32)})
        t.update({'ogg_desc': torch.tensor(data.get('ogg_desc'), dtype=torch.float32)})

        lat_list = []
        lon_list = []
        foundation_list = []
        revenue_list = []
        ateco_list = []

        comp_list = data.get('companies')
        for comp in comp_list:
            lat_c = (float(comp.get('lat')) - MIN_LAT) / (MAX_LAT - MIN_LAT)
            lat_list.append(lat_c)

            lon_c = (float(comp.get('lon')) - MIN_LON) / (MAX_LON - MIN_LON)
            lon_list.append(lon_c)

            foundation = (float(comp.get('foundation')) - MIN_FOUNDATION) / (MAX_FOUNDATION - MIN_FOUNDATION)
            foundation_list.append(foundation)

            revenue = (float(comp.get('revenue')) - AVG_REVENUE) / AVG_REVENUE
            revenue_list.append(revenue)

            ateco = comp.get('ateco_desc')
            ateco_list.append(ateco)

            labs.append(int(comp.get('label')))

        c.update({'lat': torch.tensor(lat_list, dtype=torch.float32)})
        c.update({'lon': torch.tensor(lon_list, dtype=torch.float32)})
        c.update({'foundation': torch.tensor(foundation_list, dtype=torch.float32).unsqueeze(1)})
        c.update({'revenue': torch.tensor(revenue_list, dtype=torch.float32).unsqueeze(1)})
        c.update({'ateco_desc': torch.tensor(ateco_list, dtype=torch.float32)})

        labs = torch.tensor(labs, dtype=torch.float32)

    else:
        print('ERROR - BATCH SIZE ERROR')

    return t, c, labs
