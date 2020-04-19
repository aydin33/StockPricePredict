import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoader as dl

dLoad = dl.DataLoader()
df = dLoad.PrepareDataSet()
print(df)

plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close price history')
plt.show()

#preparing moving average graphic
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)), columns=['Date','Close'])
new_data2 = pd.DataFrame(index=range(0,len(df)), columns=['Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    new_data2['Close'][i] = data['Close'][i]

train = new_data[:987]
train2 = new_data2[:987]
valid = new_data[987:]
print('\n Shape of training set:')
print(train.shape)
print('\n Shape of validation set:')
print(valid.shape)

preds = []
for i in range(0,valid.shape[0]):
    a = np.sum(train2[len(train)-248+i:], axis=0) + sum(preds)
    b = a/248
    preds.append(b)

print(preds)
print(valid)

valid['Predictions'] = 0
valid['Predictions'] = preds

plt.plot(train2['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.show()