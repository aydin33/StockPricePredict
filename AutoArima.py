import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoader as dl
from datetime import datetime
from pmdarima import auto_arima

dLoad = dl.DataLoader()
df = dLoad.PrepareDataSet()
print(df)

data = df.sort_index(ascending=True, axis=0)

train = data[:770]
valid = data[770:]

training = train['Close']
validation = valid['Close']

model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)

forecast = model.predict(n_periods=61)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])),2)))
rms

plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(forecast['Prediction'])
plt.show()