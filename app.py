import pickle
import streamlit as st
import pandas as pd

with open("models/prophet_model", "rb") as f:
    forecasting_model = pickle.load(f)

df = pd.read_csv('data_cleaned.csv')


st.title('Prévision de la consommation électrique française')

date = ["Une journée", "Une semaine", "Un mois"]

option = st.selectbox('Sur quelle durée voulez-vous prédire ?', date)

if option == "Une journée" :
    period = 24
    next_date = '2020-12-20 00:30:00'
elif option == "Une semaine":
    period = 168
    next_date = '2020-12-26 00:30:00'
else :
    period = 744
    next_date = '2021-01-19 00:30:00'

future_dates = forecasting_model.make_future_dataframe(periods=period, freq='H')
forecast = forecasting_model.predict(future_dates)

fig = forecasting_model.plot(forecast, uncertainty=False)
ax = fig.gca()
ax.set_xlim(pd.to_datetime(['2020-12-12 00:30:00', next_date]))
ax.set_ylim([3e10, 6e10])
ax.set_xlabel("Temps")
ax.set_ylabel("Puissance consommée (W)")
fig



st.markdown("Les données proviennent de l'[Open Data d'ENEDIS](https://data.enedis.fr/explore/dataset/bilan-electrique-demi-heure/information/).")
st.markdown("Les modélisations sont faites grâce à l'outil [Prophet](https://facebook.github.io/prophet/) de Facebook.")
st.markdown("Vous pouvez retrouver plus d'informations sur ce modèle ainsi que les changements à venir sur mon [site](https://guillaumepaviot.github.io/projects/prevision.html).")
