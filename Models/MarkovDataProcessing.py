import pandas as pd

df = pd.read_csv('Analytics_raw_data_sample.csv', header=5)
df = df.rename(columns={'Väg till konvertering per källa': 'path',
                        'Konverteringar': 'total_conversions',
                        'Konverteringsvärde': 'total_conversion_value'})

df['total_null'] = 0

df['total_conversions'] = df['total_conversions'].str.replace('\s+','').astype(int)
df['total_conversion_value'] = df['total_conversion_value'].str.rstrip('kr').\
    str.replace(',','.').str.replace('\s+','').astype(float).astype(int)

df.to_csv('data_processed.csv', index = False, sep=';')