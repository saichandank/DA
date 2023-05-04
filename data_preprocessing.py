import pandas as pd
import pandas_datareader.data as web
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler

###################################Q1

class data_processing(object):

    def __init__(self, data):
        self.data = data

    def normalized(self):
        df_scaled = self.data.copy()
        for column in df_scaled.columns:
            df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (df_scaled[column].max() - df_scaled[column].min())
        return df_scaled

    def standardize(self):
        df_scaled = self.data.copy()
        for column in df_scaled.columns:
            df_scaled[column] = (df_scaled[column] - df_scaled[column].mean()) / df_scaled[column].std()
        return df_scaled

    def IQR(self):
        df_scaled = self.data.copy()
        transformer = RobustScaler().fit(df_scaled)
        df = pd.DataFrame(transformer.transform(df_scaled),
             columns=["High", "Low", "Open", "Close", "Volume", "Adj Close"])
        df.insert(0, "Date", self.data.index, True)
        return df

    def show_originals(self):
        self.data.plot(y=["High", "Low", "Open", "Close", "Volume", "Adj Close"],
                       kind="line", title="Original AAPL dataset")

    def show_normalized(self):
        scaled_df = self.normalized()
        scaled_df.plot(y=["High", "Low", "Open", "Close", "Volume", "Adj Close"],
                       kind="line", title="Normalized AAPL dataset")

    def show_standardized(self):
        std_df = self.standardize()
        std_df.plot(y=["High", "Low", "Open", "Close", "Volume", "Adj Close"],
                    kind="line", title="Standardized AAPL dataset")

    def show_IQR(self):
        iqr_df = self.IQR()
        iqr_df.plot(x="Date", y=["High", "Low", "Open", "Close", "Volume", "Adj Close"],
                    kind="line", title="IQR for AAPL data set")

