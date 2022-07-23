import numpy as np
import pandas as pd
from pycaret.regression import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def Haversine(lat1: float, lon1: float, lat2: float, lon2: float)-> float:
    lon1, lon2, lat1, lat2 = map(np.radians, [lon1, lon2, lat1, lat2])
   
    diffLon = lon2 - lon1
    diffLat = lat2 - lat1

    Distance = 2 * 6371 * np.arcsin(
        np.sqrt(
            np.sin(diffLat/2)**2 +
            np.cos(lat1)*np.cos(lat2)*
            np.sin(diffLon/2)**2
        )
    )

    return Distance

dfUber = pd.read_csv('uber.csv')

del dfUber['Unnamed: 0']
del dfUber['key']

dfUber['Distance'] = Haversine(
    dfUber['pickup_latitude'],
    dfUber['pickup_longitude'],
    dfUber['dropoff_latitude'],
    dfUber['dropoff_longitude']
    )

del dfUber['pickup_latitude']
del dfUber['pickup_longitude']
del dfUber['dropoff_latitude']
del dfUber['dropoff_longitude']

dfUber = dfUber[dfUber['fare_amount'] >= 1]
dfUber = dfUber[dfUber['fare_amount'] <=80]

dfUber = dfUber[dfUber['Distance'] <= 50]
dfUber = dfUber[dfUber['Distance'] >= 0.1]

dfUber = dfUber[dfUber['passenger_count'] >= 1]
dfUber = dfUber[dfUber['passenger_count'] <= 4]

dfUber['pickup_datetime'] = pd.to_datetime(dfUber['pickup_datetime'] )
dfUber['Hour'] = dfUber['pickup_datetime'].apply(lambda x: x.hour)
dfUber['Minute'] = dfUber['pickup_datetime'].apply(lambda x: x.minute)
dfUber['Day'] = dfUber['pickup_datetime'].apply(lambda x: x.dayofweek)

del dfUber['pickup_datetime']


fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Box(
        x=dfUber['Distance'],
        name='Distância'
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Box(
        x=dfUber['fare_amount'],
        name='Preço'
    ),
    row=2,
    col=1
)

fig.add_trace(
    go.Bar(
        y=dfUber['passenger_count'].value_counts(),
        name='Quantidade de Passageiros',
        x=[1,2,3,4]
    ),
    row=3,
    col=1
)

fig.show()

EnvRegr = setup(dfUber, target='fare_amount', normalize=True)
best = create_model('lightgbm')

UberMLTunned = tune_model(best)

plot_model(UberMLTunned)
plot_model(UberMLTunned, plot ='error')
plot_model(UberMLTunned, plot ='feature_all')
plot_model(UberMLTunned, plot ='learning')

UberMLFinal = finalize_model(UberMLTunned)

jsCode = convert_model(UberMLTunned, 'javascript')
arquivoJs = open('scoreModel.js', 'a+')
arquivoJs.write('export '+jsCode)
arquivoJs.close()

save_model(UberMLFinal, 'UberPrice')

dfUberSam = dfUber.sample(frac=0.001)
dtLoad = load_model('UberPrice')
predict_model(dtLoad, data=dfUberSam)