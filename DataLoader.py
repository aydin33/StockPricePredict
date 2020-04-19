class DataLoader:
    def __init__(self):
        pass

    def PrepareDataSet(self):
        import pandas as pd

        from matplotlib.pylab import rcParams
        rcParams['figure.figsize'] = 20,10

        from sklearn.preprocessing import MinMaxScaler
        #scaler = MinMaxScaler(feature_range=(0,1))

        df = pd.read_csv('ASELSAN.csv')
        df.head()

        df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
        df.index= df['Date']
        return df
