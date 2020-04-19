import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoader as dl
from sklearn.linear_model import LinearRegression
from datetime import datetime

dLoad = dl.DataLoader()
df = dLoad.PrepareDataSet()
print(df)

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i].strftime('%Y%m%d')
    new_data['Close'][i] = data['Close'][i]

train = new_data[:3028]
valid = new_data[3028:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

model = LinearRegression()
model.fit(x_train,y_train)

preds = model.predict(x_valid)
print(preds)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms

valid['Predictions'] = 0
valid['Predictions'] = preds
valid.index = new_data[3028:].index
train.index = new_data[:3028].index
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])

plt.show()