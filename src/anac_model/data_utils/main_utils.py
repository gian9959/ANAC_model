import numpy as np
import pandas
from alive_progress import alive_bar

MIN_LAT = 36.9269
MAX_LAT = 46.4983
MIN_LON = 7.3170
MAX_LON = 18.1750

MAX_FOUNDATION = 2025

MIN_BUDGET = 1000

CATEGORIES = {'LAVORI': 0, 'FORNITURE': 1, 'SERVIZI': 2}

prov_csv = pandas.read_csv('csv/province_italiane.csv', sep=';', keep_default_na=False)
area_csv = pandas.read_csv('csv/regioni.csv', sep=';', keep_default_na=False)
istat_csv = pandas.read_csv('csv/Elenco-comuni-istat.csv', sep=',', keep_default_na=False, dtype=str)
excep_csv = pandas.read_csv('csv/prov_eccezioni.csv', sep=',', keep_default_na=False, dtype=str)


def gen_data_indexes(conn, start_date='2000-01-01', end_date='2025-12-31'):
    cur = conn.cursor()
    out_list = []

    print('LOADING FROM DATABASE...')
    query = "SELECT DISTINCT CIG, DATA_PUBBLICAZIONE, OGGETTO_PRINCIPALE_CONTRATTO FROM gare WHERE DATA_PUBBLICAZIONE >= %s AND DATA_PUBBLICAZIONE <= %s AND IMPORTO_COMPLESSIVO_GARA >= 1000 ORDER BY DATA_PUBBLICAZIONE ASC"
    params = (start_date, end_date)
    cur.execute(query, params)
    res = cur.fetchall()
    with alive_bar(len(res)) as bar:
        for row in res:
            cig = row[0]
            if cig is None:
                bar()
                continue

            year = int(str(row[1]).split('-')[0])
            if year is None or year == 0:
                bar()
                continue

            cat = CATEGORIES.get(row[2].upper())
            if cat is None:
                bar()
                continue

            query = "SELECT CODICE_FISCALE, AGGIUDICATO FROM partecipazioni WHERE CIG = %s"
            params = (cig,)
            cur.execute(query, params)
            parts = cur.fetchall()

            ex_parts = []
            revs = []
            emps = []
            for p in parts:
                query = "SELECT * FROM imprese WHERE CODICE_FISCALE = %s"
                params = (p[0],)
                cur.execute(query, params)
                c = cur.fetchall()
                if len(c) > 0:
                    c = c[0]
                    check_prov = c[2] is not None and c[2] != ''
                    check_ateco = c[3] is not None and c[3] != ''
                    check_foundation = c[4] is not None and c[4] != 0 and c[4] < year
                    if check_prov and check_ateco and check_foundation:
                        start_index = year - 2016

                        query = "SELECT * FROM ricavi WHERE CODICE_FISCALE = %s"
                        params = (p[0],)
                        cur.execute(query, params)
                        rev = cur.fetchall()
                        found_rev = None

                        query = "SELECT * FROM dipendenti WHERE CODICE_FISCALE = %s"
                        params = (p[0],)
                        cur.execute(query, params)
                        emp = cur.fetchall()
                        found_emp = None

                        if len(rev) > 0 and len(emp) > 0:
                            rev = rev[0]
                            for i in range(start_index, -1, -1):
                                if rev[i + 1] is not None:
                                    found_rev = rev[i + 1]
                                    break

                            emp = emp[0]
                            for i in range(start_index, -1, -1):
                                if emp[i + 1] is not None and emp[i + 1] >= 0:
                                    found_emp = emp[i + 1]
                                    break

                        if found_rev is not None and found_emp is not None:
                            revs.append(found_rev)
                            emps.append(found_emp)
                            ex_parts.append(p)

            if len(ex_parts) > 0:
                cf_list = [p[0] for p in ex_parts]
                lab_list = [p[1] for p in ex_parts]

                row_dict = dict()
                row_dict.update({"CIG": cig})
                row_dict.update({"CF": cf_list})
                row_dict.update({"REVS": revs})
                row_dict.update({"EMPS": emps})
                row_dict.update({"LABELS": lab_list})

                out_list.append(row_dict)
            bar()

    return pandas.DataFrame(out_list)


def extract_from_index(row):
    cig = row.get('CIG')
    cf_list = row.get('CF').replace("[", "").replace("]", "").replace("'", "").split(", ")
    labels = row.get('LABELS').replace("[", "").replace("]", "").split(", ")
    labels = [int(l) for l in labels]
    revs = row.get('REVS').replace("[", "").replace("]", "").split(", ")
    revs = [float(r) for r in revs]
    emps = row.get('EMPS').replace("[", "").replace("]", "").split(", ")
    emps = [int(e) for e in emps]
    return cig, cf_list, labels, revs, emps


def load_from_index(cig, cf_list, labels, revs, emps, conn):
    cur = conn.cursor()
    query = "SELECT * FROM gare WHERE CIG = %s"
    params = (cig,)
    cur.execute(query, params)
    res = cur.fetchall()[0]

    prov = res[3].upper()
    lat = None
    lon = None
    location = prov_csv.loc[prov_csv["Sigla"] == prov]
    if not location.empty:
        lat = location.get('Latitudine').iloc[0]
        lon = location.get('Longitudine').iloc[0]

    budget = res[4]
    cat = res[5]
    cpv = np.frombuffer(res[7], dtype=np.float32)
    ogg = np.frombuffer(res[8], dtype=np.float32)

    companies = []
    for k, cf in enumerate(cf_list):
        query = "SELECT * FROM imprese WHERE CODICE_FISCALE = %s"
        params = (cf,)
        cur.execute(query, params)
        res = cur.fetchall()[0]

        prov_c = res[2].upper()
        lat_c = None
        lon_c = None
        region= None
        location_c = prov_csv.loc[prov_csv["Sigla"] == prov_c]
        if not location_c.empty:
            lat_c = location_c.get('Latitudine').iloc[0]
            lon_c = location_c.get('Longitudine').iloc[0]
            region = location_c.get('Regione').iloc[0]

        foundation = res[4]
        ateco = np.frombuffer(res[5], dtype=np.float32)
        label = labels[k]
        revenue = revs[k]
        employees = emps[k]

        company = dict()
        if lat_c is not None and lon_c is not None and region is not None:
            company.update({'id': cf})
            company.update({'lat': lat_c})
            company.update({'lon': lon_c})
            company.update({'foundation': foundation})
            company.update({'revenue': revenue})
            company.update({'employees': employees})
            company.update({'ateco': ateco})
            company.update({'label': label})
            company.update({'region': region})
            companies.append(company)
        else:
            raise Exception(f'INVALID PROVINCE {prov_c}')

    tender = dict()
    if lat is not None and lon is not None:
        tender.update({'id': cig})
        tender.update({'lat': lat})
        tender.update({'lon': lon})
        tender.update({'budget': budget})
        tender.update({'cat': cat})
        tender.update({'cpv': cpv})
        tender.update({'ogg': ogg})
        tender.update({'companies': companies})
    else:
        raise Exception(f'INVALID PROVINCE: "{prov}"')

    return tender


def signed_log1p(num):
    return np.sign(num) * np.log1p(np.abs(num))


def normalization(data, constants):
    MIN_FOUNDATION = constants['min_foundation']
    MAX_BUDGET = constants['max_budget']
    MIN_REVENUE = constants['min_revenue']
    MAX_REVENUE = constants['max_revenue']
    MAX_EMPLOYEES = constants['max_employees']

    try:
        norm_data = dict(data)

        lat = (float(data.get('lat')) - MIN_LAT) / (MAX_LAT - MIN_LAT)
        norm_data.update({'lat': lat})

        lon = (float(data.get('lon')) - MIN_LON) / (MAX_LON - MIN_LON)
        norm_data.update({'lon': lon})

        budget = (np.log1p(float(data.get('budget'))) - np.log1p(MIN_BUDGET)) / (np.log1p(MAX_BUDGET) - np.log1p(MIN_BUDGET))
        norm_data.update({'budget': budget})

        cat = CATEGORIES.get(data.get('cat').upper())
        if cat is None:
            raise Exception(f"INVALID CAT: <{data.get('cat')}>")
        norm_data.update({'cat': cat})

        comp_list = list()
        for comp in data.get('companies'):
            c = dict(comp)

            lat_c = (float(comp.get('lat')) - MIN_LAT) / (MAX_LAT - MIN_LAT)
            c.update({'lat': lat_c})

            lon_c = (float(comp.get('lon')) - MIN_LON) / (MAX_LON - MIN_LON)
            c.update({'lon': lon_c})

            foundation = (float(comp.get('foundation')) - MIN_FOUNDATION) / (MAX_FOUNDATION - MIN_FOUNDATION)
            c.update({'foundation': foundation})

            revenue = (signed_log1p(float(comp.get('revenue'))) - signed_log1p(MIN_REVENUE)) / (signed_log1p(MAX_REVENUE) - signed_log1p(MIN_REVENUE))
            c.update({'revenue': revenue})

            employees = np.log1p(int(comp.get('employees'))) / np.log1p(MAX_EMPLOYEES)
            c.update({'employees': employees})

            comp_list.append(c)

        norm_data.update({'companies': comp_list})

        return norm_data

    except Exception as e:
        raise e
