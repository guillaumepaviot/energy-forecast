import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

response = requests.get("https://data.enedis.fr/api/records/1.0/search/?dataset=bilan-electrique-demi-heure&q=&rows=10000&sort=horodate&facet=horodate")
json_data = json.loads(response.text)
date, injection, consommation =[], [], []
for record in json_data['records'] :
    json_date = record['fields']['horodate'].replace('T',' ')
    json_date = json_date.replace('00+00:00','00')
    date.append(json_date)
    injection.append(record['fields']['injection_rte'])
    consommation.append(record['fields']['consommation_totale'])
date = pd.to_datetime(date)
data = pd.DataFrame(list(zip(injection, consommation)), index = date,columns=['Injection', 'Consommation'])


decomposition = sm.tsa.seasonal_decompose(data['Consommation'], model='additive', period='30T')
fig = decomposition.plot()
plt.show()