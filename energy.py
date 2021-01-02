import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

date, injection, consommation =[], [], []


import json
import requests

#response = requests.get("https://data.enedis.fr/api/records/1.0/search/?dataset=bilan-electrique-demi-heure&q=&rows=10000&sort=horodate&facet=horodate")
json_file = open("bilan-electrique-demi-heure.json", "r")
json_data = json.loads(json_file.read())
for record in json_data :
    json_date = record['fields']['horodate'].replace('T',' ')
    json_date = json_date.replace('00+01:00','00')
    date.append(json_date)
    injection.append(record['fields']['injection_rte'])
    consommation.append(record['fields']['consommation_totale'])
date = pd.DatetimeIndex(pd.to_datetime(date, utc=True)).tz_convert('Europe/Paris')
data = pd.DataFrame(list(zip(injection, consommation)), index = date,columns=['Injection', 'Consommation'])

#data = pd.read_csv("bilan-electrique-demi-heure.csv")


print(data.head())
y = data['Consommation'].resample('MS').mean()
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()