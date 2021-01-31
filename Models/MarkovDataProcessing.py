import pandas as pd

def process_GA_data(file_path, save_to_path):
    df = pd.read_csv(file_path, header=5)
    df = df.rename(columns={'Väg till konvertering per källa': 'path',
                            'Konverteringar': 'total_conversions',
                            'Konverteringsvärde': 'total_conversion_value'})

    df['total_null'] = 0

    df['total_conversions'] = df['total_conversions'].str.replace('\s+','').astype(int)
    df['total_conversion_value'] = df['total_conversion_value'].str.rstrip('kr').\
        str.replace(',','.').str.replace('\s+','').astype(float).astype(int)

    df.to_csv(save_to_path, index = False, sep=';')

if __name__ == '__main__':
    file_path = '../Data/Analytics_raw_data_sample.csv'
    save_to_path = '../Data/GA_data_processed.csv'
    process_GA_data(file_path, save_to_path)