import pandas as pd
import numpy as np

df = pd.read_csv('Analytics.csv', header=5)
df = df.rename(columns={'Väg till konvertering per källa': 'path',
                        'Konverteringar': 'total_conversions',
                        'Konverteringsvärde': 'total_conversion_value'})
df.insert(len(df.columns),'total_null',np.zeros(len(df), dtype=np.int))

df['total_conversions'] = df['total_conversions'].str.replace('\s+','').astype(int)
df['total_conversion_value'] = df['total_conversion_value'].str.rstrip('kr').\
    str.replace(',','.').str.replace('\s+','').astype(float).astype(int)

df.to_csv('data_processed.csv', index = False, sep=';')