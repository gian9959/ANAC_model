import numpy as np
import pymysql as pymysql
from torch.utils.data import Dataset
from alive_progress import alive_bar

import pandas

hostname = 'localhost'
username = 'root'
password = ''
database = 'anac'

conn = pymysql.connect(host=hostname, user=username, passwd=password, db=database)
cur = conn.cursor()


class AnacDataset(Dataset):
    def __init__(self):
        self.data = []

        data_csv = pandas.read_csv('./data/dataset.csv', sep=';')
        d_length = len(data_csv)
        prov_csv = pandas.read_csv('./data/province_italiane.csv', sep=';')
        with alive_bar(d_length) as bar:
            for i, cig in enumerate(data_csv.get('CIG')):
                query = 'SELECT * FROM gare WHERE CIG = %s'
                params = (cig,)
                cur.execute(query, params)
                res = cur.fetchall()[0]
                prov = res[2]
                for j, p in enumerate(prov_csv.get('Sigla')):
                    if p == prov:
                        lat = prov_csv.get('Latitudine')[j]
                        lon = prov_csv.get('Longitudine')[j]
                        break
                budget = res[3]
                cat = res[4]
                cpv_desc = np.frombuffer(res[6], dtype=np.float32)
                ogg_desc = np.frombuffer(res[7], dtype=np.float32)

                companies = []
                cf_list = data_csv.get('CF')[i].replace("[", "").replace("]", "").replace("'", "").split(", ")
                targets = data_csv.get('AGG')[i].replace("[", "").split(", ")
                for k, cf in enumerate(cf_list):
                    query = 'SELECT * FROM imprese WHERE CODICE_FISCALE = %s'
                    params = (cf,)
                    cur.execute(query, params)
                    res = cur.fetchall()[0]
                    prov_c = res[2]
                    for j, p in enumerate(prov_csv.get('Sigla')):
                        if p == prov_c:
                            lat_c = prov_csv.get('Latitudine')[j]
                            lon_c = prov_csv.get('Longitudine')[j]
                            break
                    foundation = res[5]
                    revenue = res[6]
                    ateco_desc = np.frombuffer(res[7], dtype=np.float32)
                    label = targets[k]

                    company = dict()
                    company.update({'lat': lat_c})
                    company.update({'lon': lon_c})
                    company.update({'foundation': foundation})
                    company.update({'revenue': revenue})
                    company.update({'ateco_desc': ateco_desc})
                    company.update({'label': label})
                    companies.append(company)

                tender = dict()
                tender.update({'lat': lat})
                tender.update({'lon': lon})
                tender.update({'budget': budget})
                tender.update({'cat': cat})
                tender.update({'cpv_desc': cpv_desc})
                tender.update({'ogg_desc': ogg_desc})
                tender.update({'companies': companies})
                self.data.append(tender)
                bar()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

